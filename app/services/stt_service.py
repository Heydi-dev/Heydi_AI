import asyncio
from io import BytesIO
import wave

import numpy as np
import torch
import whisper
from whisper import DecodingOptions, decode
from whisper.audio import log_mel_spectrogram, pad_or_trim


def _load_model(model_name: str):
    return whisper.load_model(model_name)


class WhisperSTTService:
    def __init__(self, model_name: str = "base", target_sample_rate: int = 16000):
        self.model_name = model_name
        self._model = None
        self.target_sample_rate = target_sample_rate

    @property
    def model(self):
        if self._model is None:
            self._model = _load_model(self.model_name)
        return self._model

    async def transcribe_wav_bytes(self, audio_bytes: bytes) -> str:
        if not audio_bytes:
            return ""
        return await asyncio.to_thread(self._transcribe, audio_bytes)

    def _transcribe(self, audio_bytes: bytes) -> str:
        audio_array = self._bytes_to_audio_array(audio_bytes)
        if audio_array.size == 0:
            return ""

        audio_tensor = torch.from_numpy(audio_array)
        audio_tensor = pad_or_trim(audio_tensor)
        mel = log_mel_spectrogram(audio_tensor).to(self.model.device)

        options = DecodingOptions(language="ko", fp16=torch.cuda.is_available())
        result = decode(self.model, mel, options)
        return result.text.strip()

    def _bytes_to_audio_array(self, audio_bytes: bytes) -> np.ndarray:
        with wave.open(BytesIO(audio_bytes), "rb") as wf:
            sample_rate = wf.getframerate()
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            frames = wf.readframes(wf.getnframes())

        if sampwidth == 2:
            dtype = np.int16
        elif sampwidth == 4:
            dtype = np.int32
        else:
            dtype = np.int16

        audio = np.frombuffer(frames, dtype=dtype).astype(np.float32)

        if n_channels > 1:
            audio = audio.reshape(-1, n_channels).mean(axis=1)

        if sampwidth == 4:
            audio /= np.iinfo(np.int32).max
        else:
            audio /= np.iinfo(np.int16).max

        if sample_rate != self.target_sample_rate:
            audio = self._resample(audio, sample_rate, self.target_sample_rate)

        return audio

    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        if orig_sr == target_sr or audio.size == 0:
            return audio
        duration = audio.shape[0] / orig_sr
        target_length = int(round(duration * target_sr))
        target_indices = np.linspace(0, audio.shape[0] - 1, num=target_length)
        return np.interp(target_indices, np.arange(audio.shape[0]), audio)
