import asyncio
from typing import AsyncIterable

from fastapi import WebSocket
from google import genai

class ConversationService:
    """
    Conversation pipeline that handles WebSocket audio streams
    with Google's generative AI live session.
    """

    def __init__(self):
        self.client = genai.Client()

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
