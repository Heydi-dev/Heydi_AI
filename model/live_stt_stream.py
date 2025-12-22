"""
Utility script for running real-time STT with Whisper using microphone input.
This mirrors the sample provided but wraps it with a reusable class and CLI hook.
"""
import time
import wave
from pathlib import Path
from typing import Optional

import pyaudio
import whisper


class LiveSTTStreamer:
    def __init__(
        self,
        model_name: str = "base",
        chunk: int = 1024,
        channels: int = 1,
        rate: int = 16000,
        record_seconds: int = 5,
        temp_wave_file: str = "temp.wav",
    ):
        self.chunk = chunk
        self.channels = channels
        self.rate = rate
        self.record_seconds = record_seconds
        self.temp_wave_file = Path(temp_wave_file)

        self.model = whisper.load_model(model_name)
        self.audio = pyaudio.PyAudio()
        self.stream: Optional[pyaudio.Stream] = None

    def start(self):
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk,
        )

    def stop(self):
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()
        if self.temp_wave_file.exists():
            self.temp_wave_file.unlink()

    def record_chunk(self) -> bytes:
        frames = []
        for _ in range(0, int(self.rate / self.chunk * self.record_seconds)):
            frames.append(self.stream.read(self.chunk))
        return b"".join(frames)

    def save_temp_wave(self, data: bytes):
        with wave.open(str(self.temp_wave_file), "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(self.rate)
            wf.writeframes(data)

    def transcribe(self, audio_bytes: bytes) -> str:
        self.save_temp_wave(audio_bytes)
        result = self.model.transcribe(str(self.temp_wave_file))
        return result.get("text", "").strip()

    def run(self):
        self.start()
        print("실시간 음성 인식 시작 (Ctrl+C로 종료)")
        try:
            while True:
                chunk_bytes = self.record_chunk()
                text = self.transcribe(chunk_bytes)
                print("📢 인식된 텍스트:", text)
        except KeyboardInterrupt:
            print("종료합니다.")
        finally:
            self.stop()


if __name__ == "__main__":
    LiveSTTStreamer().run()
