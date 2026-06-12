from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
import wave
from pathlib import Path

import numpy as np
import websockets


DEFAULT_URL = "ws://127.0.0.1:8000/api/v1/model/ws/conversations"
TEXT_KEEP_RE = re.compile(r"[0-9A-Za-z가-힣]")


def read_text(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return path.read_text(encoding=encoding).strip()
        except UnicodeDecodeError:
            continue
    return path.read_text().strip()


def read_wav_as_pcm16_mono(path: Path, target_rate: int = 16000) -> tuple[bytes, float]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())

    if sample_width == 2:
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
        audio /= np.iinfo(np.int16).max
    elif sample_width == 4:
        audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32)
        audio /= np.iinfo(np.int32).max
    else:
        raise ValueError(f"Unsupported sample width: {sample_width} bytes")

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    if sample_rate != target_rate:
        audio = resample_linear(audio, sample_rate, target_rate)

    int16 = np.clip(audio, -1.0, 1.0)
    int16 = (int16 * np.iinfo(np.int16).max).astype(np.int16)
    duration_ms = len(int16) / target_rate * 1000
    return int16.tobytes(), duration_ms


def resample_linear(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate or audio.size == 0:
        return audio
    duration = audio.shape[0] / source_rate
    target_length = int(round(duration * target_rate))
    if target_length <= 0:
        return np.array([], dtype=np.float32)
    target_indices = np.linspace(0, audio.shape[0] - 1, num=target_length)
    return np.interp(target_indices, np.arange(audio.shape[0]), audio).astype(np.float32)


def chunk_bytes(data: bytes, chunk_size: int) -> list[bytes]:
    return [data[index : index + chunk_size] for index in range(0, len(data), chunk_size)]


async def send_audio(
    websocket,
    audio_bytes: bytes,
    *,
    chunk_ms: int,
    sample_rate: int,
    trailing_silence_ms: int,
) -> float:
    bytes_per_ms = sample_rate * 2 / 1000
    chunk_size = max(2, int(bytes_per_ms * chunk_ms))
    if chunk_size % 2:
        chunk_size += 1

    chunks = chunk_bytes(audio_bytes, chunk_size)
    silence_bytes = b"\x00\x00" * int(sample_rate * trailing_silence_ms / 1000)
    if silence_bytes:
        chunks.extend(chunk_bytes(silence_bytes, chunk_size))

    started = time.perf_counter()
    for chunk in chunks:
        await websocket.send(chunk)
        await asyncio.sleep(chunk_ms / 1000)
    return (time.perf_counter() - started) * 1000


async def receive_input_transcriptions(
    websocket,
    *,
    send_done: asyncio.Event,
    timeout_s: float,
    post_send_wait_s: float,
) -> list[str]:
    transcripts: list[str] = []
    deadline = time.perf_counter() + timeout_s
    last_message_at = time.perf_counter()
    while time.perf_counter() < deadline:
        if send_done.is_set() and time.perf_counter() - last_message_at >= post_send_wait_s:
            break
        remaining = max(0.1, deadline - time.perf_counter())
        try:
            message = await asyncio.wait_for(websocket.recv(), timeout=min(0.5, remaining))
        except asyncio.TimeoutError:
            continue
        except websockets.ConnectionClosed:
            break

        last_message_at = time.perf_counter()
        if isinstance(message, bytes):
            continue

        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            continue

        event_type = payload.get("type")
        text = payload.get("transcription")
        if event_type == "input" and text:
            transcripts.append(str(text).strip())
    return transcripts


def normalize_for_cer(text: str) -> str:
    return "".join(TEXT_KEEP_RE.findall(text)).lower()


def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (char_a != char_b)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def calculate_cer(reference: str, hypothesis: str) -> tuple[float, int, int]:
    normalized_ref = normalize_for_cer(reference)
    normalized_hyp = normalize_for_cer(hypothesis)
    print(f"Normalized Reference: {normalized_ref}")
    print(f"Normalized Hypothesis: {normalized_hyp}")
    distance = levenshtein_distance(normalized_ref, normalized_hyp)
    length = len(normalized_ref)
    cer = distance / length * 100 if length else 0.0
    return cer, distance, length


async def measure(args: argparse.Namespace) -> None:
    reference = read_text(Path(args.reference))
    audio_bytes, duration_ms = read_wav_as_pcm16_mono(
        Path(args.audio), target_rate=args.sample_rate
    )

    url = args.url
    if args.user_id is not None:
        separator = "&" if "?" in url else "?"
        url = f"{url}{separator}user_id={args.user_id}"

    started = time.perf_counter()
    async with websockets.connect(url, max_size=None, ping_interval=None) as websocket:
        send_done = asyncio.Event()

        async def send_and_mark_done() -> float:
            try:
                return await send_audio(
                    websocket,
                    audio_bytes,
                    chunk_ms=args.chunk_ms,
                    sample_rate=args.sample_rate,
                    trailing_silence_ms=args.trailing_silence_ms,
                )
            finally:
                send_done.set()

        send_task = asyncio.create_task(
            send_and_mark_done()
        )
        receive_task = asyncio.create_task(
            receive_input_transcriptions(
                websocket,
                send_done=send_done,
                timeout_s=args.timeout,
                post_send_wait_s=args.post_send_wait,
            )
        )
        send_duration_ms, transcripts = await asyncio.gather(send_task, receive_task)
    elapsed_ms = (time.perf_counter() - started) * 1000

    hypothesis = "".join(transcripts).strip()
    cer, distance, reference_length = calculate_cer(reference, hypothesis)

    print("=== Gemini Live STT CER Measurement ===")
    print(f"audio: {args.audio}")
    print(f"reference: {args.reference}")
    print(f"audio_duration_ms: {duration_ms:.1f}")
    print(f"send_duration_ms: {send_duration_ms:.1f}")
    print(f"elapsed_ms: {elapsed_ms:.1f}")
    print(f"input_transcription_events: {len(transcripts)}")
    print(f"reference_chars_normalized: {reference_length}")
    print(f"edit_distance: {distance}")
    print(f"cer: {cer:.2f}%")
    print()
    print("=== Reference ===")
    print(reference)
    print()
    print("=== Hypothesis ===")
    print(hypothesis if hypothesis else "[no input transcription received]")
    print()
    print("=== Input Transcription Events ===")
    for index, transcript in enumerate(transcripts, start=1):
        print(f"{index}. {transcript}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure CER for Gemini Live input transcription."
    )
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--audio", default="tests/sample_long.wav")
    parser.add_argument("--reference", default="tests/sample_long.txt")
    parser.add_argument("--user-id", type=int, default=None)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--chunk-ms", type=int, default=100)
    parser.add_argument("--trailing-silence-ms", type=int, default=2500)
    parser.add_argument("--timeout", type=float, default=80)
    parser.add_argument(
        "--post-send-wait",
        type=float,
        default=10,
        help="Seconds to keep collecting input transcriptions after audio send completes.",
    )
    args = parser.parse_args()
    asyncio.run(measure(args))


if __name__ == "__main__":
    main()
