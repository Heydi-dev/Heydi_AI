import asyncio
from typing import Iterable

from fastapi import WebSocket
from google import genai
from google.genai import types

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

    def _format_turns_for_diary(self, turns: Iterable) -> str:
        formatted_lines = []
        for turn in turns:
            role = getattr(turn, "role", None)
            if role is None and isinstance(turn, dict):
                role = turn.get("role")
            text = getattr(turn, "text", None)
            if text is None and isinstance(turn, dict):
                text = turn.get("text")
            if not text:
                continue
            role_label = "사용자" if role == "user" else "AI"
            formatted_lines.append(f"{role_label}: {text}")
        return "\n".join(formatted_lines)

    def generate_diary(self, turns: Iterable) -> str:
        conversation = self._format_turns_for_diary(turns)
        if not conversation:
            return ""
        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                config=types.GenerateContentConfig(
                    system_instruction=(
                        "당신은 사용자의 대화를 바탕으로 일기를 작성하는 도우미입니다. "
                        "아래 대화 내용을 참고해 사용자가 직접 쓴 것 같은 자연스러운 한국어 일기를 작성하세요. "
                        "일기 본문만 작성하고 제목이나 리스트, 말머리는 쓰지 마세요. "
                        "문체는 일기체로 작성하고 감정 표현을 포함하세요."
                    ),
                ),
                contents=conversation,
            )
        except Exception as e:
            print(f"Error generating diary: {e}")
            return ""
        return response.text or ""

    def summarize_diary(self, diary_text: str) -> str:
        if not diary_text:
            return ""

        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            config=types.GenerateContentConfig(
                system_instruction=(
                    "당신은 일기 내용을 한 줄로 요약하는 도우미입니다. "
                    "아래 일기에서 핵심만 뽑아 한국어 한 문장으로 요약하세요. "
                    "줄바꿈, 따옴표, 말머리 없이 한 줄로만 작성하세요."
                ),
            ),
            contents=diary_text,
        )

        summary = (response.text or "").replace("\r", " ").replace("\n", " ").strip()
        if len(summary) > 120:
            summary = summary[:117].rstrip() + "..."
        return summary

    async def receive_data(
        self, live_session, websocket: WebSocket, user_id: int | None = None
    ) -> None:
        """Receive model output (audio/transcription) and forward to WebSocket."""
        while True:
            try:
                async for response in live_session.receive():
                    if response.server_content:
                        await self._handle_server_content(
                            response.server_content, websocket, user_id
                        )
            except Exception as exc:
                print(f"Error receiving audio: {exc}")
                break

    async def _handle_server_content(
        self, server_content, websocket: WebSocket, user_id: int | None
    ) -> None:
        if getattr(server_content, "interrupted", False):
            await self._publish_interrupt(websocket)

        if server_content.model_turn:
            for part in server_content.model_turn.parts:
                if part.inline_data and isinstance(part.inline_data.data, bytes):
                    await websocket.send_bytes(part.inline_data.data)

        if server_content.output_transcription:
            transcription = server_content.output_transcription
            await self._publish_transcription(
                websocket=websocket,
                direction="output",
                text=transcription.text,
                finished=transcription.finished,
                user_id=user_id,
            )

        if server_content.input_transcription:
            transcription = server_content.input_transcription
            await self._publish_transcription(
                websocket=websocket,
                direction="input",
                text=transcription.text,
                finished=transcription.finished,
                user_id=user_id,
            )

    async def _publish_transcription(
        self,
        *,
        websocket: WebSocket,
        direction: str,
        text: str | None,
        finished: bool | None,
        user_id: int | None,
    ) -> None:
        await websocket.send_json(
            {"type": direction, "transcription": text, "finished": finished}
        )
        if user_id is None or not text:
            return

        if direction == "output":
            self._conversation_recorder.record_output(user_id, text, finished)
            return

        self._conversation_recorder.record_input(user_id, text, finished)
        await self._on_input_transcription(user_id, text, finished)

    async def _publish_interrupt(self, websocket: WebSocket) -> None:
        await websocket.send_json({"type": "interrupt"})

    async def _on_input_transcription(
        self, user_id: int, text: str, finished: bool | None
    ) -> None:
        """Hook for subclasses that need custom behavior on user transcripts."""
        return

    async def send_audio(self, live_session, websocket: WebSocket):
        """Receive audio from the WebSocket and send it to the live session."""
        while True:
            try:
                chunk = await websocket.receive_bytes()
                msg = {"data": chunk, "mime_type": "audio/pcm"}
                await live_session.send_realtime_input(audio=msg)
            except Exception as e:
                print(f"Error sending audio: {e}")
                break

    async def handle_conversation(
        self, websocket: WebSocket, user_id: int | None = None
    ):
        """Handle a WebSocket conversation with audio streaming."""
        try:
            async with self.client.aio.live.connect(
                model=self.LIVE_MODEL,
                config=self._build_live_config(),
            ) as live_session:
                async with asyncio.TaskGroup() as tg:
                    tg.create_task(self.receive_data(live_session, websocket, user_id))
                    tg.create_task(self.send_audio(live_session, websocket))
        except asyncio.CancelledError:
            pass

    def _build_live_config(self) -> dict:
        return {
            "response_modalities": ["AUDIO"],
            "system_instruction": "You are a helpful and friendly AI assistant.",
            "input_audio_transcription": {},
            "output_audio_transcription": {},
        }
