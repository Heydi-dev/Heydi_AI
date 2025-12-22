from typing import AsyncIterable


class ConversationService:
    """
    Placeholder conversation pipeline that simply echoes audio chunks.
    Later this will integrate STT -> LLM -> TTS.
    """

    async def process_audio_stream(
        self, chunks: AsyncIterable[bytes]
    ) -> AsyncIterable[bytes]:
        async for chunk in chunks:
            yield await self.process_audio_chunk(chunk)

    async def process_audio_chunk(self, chunk: bytes) -> bytes:
        return chunk
