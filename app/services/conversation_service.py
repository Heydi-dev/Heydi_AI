import asyncio
from typing import Iterable

from fastapi import WebSocket
from google import genai
from google.genai import types

class ConversationService:
    """
    Conversation pipeline that handles WebSocket audio streams
    with Google's generative AI live session.
    """

    def __init__(self):
        self.client = genai.Client()

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

    async def receive_audio(self, live_session, websocket: WebSocket):
        """Receive audio from the live session and send it through the WebSocket."""
        while True:
            try:
                turn = live_session.receive()
                async for response in turn:
                    if (response.server_content and response.server_content.model_turn):
                        for part in response.server_content.model_turn.parts:
                            if part.inline_data and isinstance(part.inline_data.data, bytes):
                                await websocket.send_bytes(part.inline_data.data)
            except Exception as e:
                print(f"Error receiving audio: {e}")
                break
    
    async def receive_data(self, live_session, websocket: WebSocket):
        """Asynchronously receive audio and text data from the WebSocket."""
        while True:
            try:
                turn = live_session.receive()
                async for response in turn:
                    if (response.server_content):
                        if (response.server_content.model_turn):
                            for part in response.server_content.model_turn.parts:
                                if part.inline_data and isinstance(part.inline_data.data, bytes):
                                    await websocket.send_bytes(part.inline_data.data)
                        if response.server_content.output_transcription:
                            transcription = response.server_content.output_transcription
                            print("Transcription:", transcription.text, "Finished:", transcription.finished)
                            await websocket.send_json({"type": "output", "transcription": transcription.text, "finished": transcription.finished})
                        if response.server_content.input_transcription:
                            transcription = response.server_content.input_transcription
                            print("User said:", transcription.text, "Finished:", transcription.finished)
                            await websocket.send_json({"type": "input", "transcription": transcription.text, "finished": transcription.finished})
            except Exception as e:
                print(f"Error receiving audio: {e}")
                break
            
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

    async def handle_conversation(self, websocket: WebSocket):
        """Handle a WebSocket conversation with audio streaming."""
        try:
            async with self.client.aio.live.connect(
                model="gemini-2.5-flash-native-audio-preview-12-2025",
                config={
                    "response_modalities": ["AUDIO"],
                    "system_instruction": "You are a helpful and friendly AI assistant.",
                    "input_audio_transcription": {},
                    "output_audio_transcription": {}
                },
            ) as live_session:
                async with asyncio.TaskGroup() as tg:
                    tg.create_task(self.receive_data(live_session, websocket))
                    tg.create_task(self.send_audio(live_session, websocket))
        except asyncio.CancelledError:
            pass
