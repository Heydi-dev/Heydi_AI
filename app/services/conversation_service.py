import asyncio

from fastapi import WebSocket
from google import genai

from app.services.conversation_recorder import (
    ConversationRecorder,
    NoopConversationRecorder,
)


class ConversationService:
    """
    Conversation pipeline that handles WebSocket audio streams
    with Google's generative AI live session.
    """

    LIVE_MODEL = "gemini-2.5-flash-native-audio-preview-12-2025"

    def __init__(self, conversation_recorder: ConversationRecorder | None = None):
        self.client = genai.Client()
        self._conversation_recorder = conversation_recorder or NoopConversationRecorder()

    async def handle_conversation(
        self, websocket: WebSocket, user_id: int | None = None
    ):
        """Connect to Gemini Live API and assign receive/send handlers."""
        try:
            async with self.client.aio.live.connect(
                model=self.LIVE_MODEL,
                config=self._build_live_config(),
            ) as live_session:
                await self._run_live_session_tasks(
                    self.receive_data(live_session, websocket, user_id),
                    self.send_audio(live_session, websocket),
                )
        except asyncio.CancelledError:
            pass

    async def send_audio(self, live_session, websocket: WebSocket):
        """
        Send audio data from the client to Gemini Live API (uplink direction).
        This runs in a loop until the connection is closed.
        """
        while True:
            try:
                chunk = await websocket.receive_bytes()
                msg = {"data": chunk, "mime_type": "audio/pcm;rate=16000"}
                await live_session.send_realtime_input(audio=msg)
            except Exception as e:
                print(f"Error sending audio: {e}")
                break

    async def receive_data(
        self, live_session, websocket: WebSocket, user_id: int | None = None
    ) -> None:
        """Receive model output and forward supported events to the WebSocket."""
        while True:
            try:
                async for response in live_session.receive():
                    if response.server_content:
                        await self._handle_live_session_content(
                            response.server_content, websocket, user_id
                        )
            except Exception as exc:
                print(f"Error receiving audio: {exc}")
                break

    async def _handle_live_session_content(
        self, server_content, websocket: WebSocket, user_id: int | None
    ) -> None:
        """Handle live session content, then send it to the client."""
        if getattr(server_content, "interrupted", False):
            await self._publish_interrupt(websocket)

        # Handle audio data and send it to the client
        if server_content.model_turn:
            for part in server_content.model_turn.parts:
                if part.inline_data and isinstance(part.inline_data.data, bytes):
                    await websocket.send_bytes(part.inline_data.data)

        # Handle transcription from model output
        if server_content.output_transcription:
            transcription = server_content.output_transcription
            await self._publish_transcription(
                websocket=websocket,
                direction="output",
                text=transcription.text,
                user_id=user_id,
            )

        # Handle transcription from user input
        if server_content.input_transcription:
            transcription = server_content.input_transcription
            await self._publish_transcription(
                websocket=websocket,
                direction="input",
                text=transcription.text,
                user_id=user_id,
            )

        if getattr(server_content, "turn_complete", False):
            await self._publish_turn_complete(websocket)

    async def _publish_transcription(
        self,
        *,
        websocket: WebSocket,
        direction: str,
        text: str | None,
        user_id: int | None,
    ) -> None:
        await websocket.send_json({"type": direction, "transcription": text})
        if user_id is None or not text:
            return

        if direction == "output":
            self._conversation_recorder.record_output(user_id, text)
            return

        self._conversation_recorder.record_input(user_id, text)

    async def _publish_interrupt(self, websocket: WebSocket) -> None:
        await websocket.send_json({"type": "interrupt"})

    async def _publish_turn_complete(self, websocket: WebSocket) -> None:
        await websocket.send_json({"type": "turn_complete"})

    async def _run_live_session_tasks(self, receive_coro, send_coro) -> None:
        tasks = [
            asyncio.create_task(receive_coro),
            asyncio.create_task(send_coro),
        ]
        try:
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            for task in done:
                if task.cancelled():
                    continue
                exc = task.exception()
                if exc:
                    raise exc
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()

    def _build_live_config(self) -> dict:
        return {
            "response_modalities": ["AUDIO"],
            "system_instruction": "You are a helpful and friendly AI assistant.",
            "input_audio_transcription": {},
            "output_audio_transcription": {},
        }
