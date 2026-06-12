from __future__ import annotations

import asyncio
from typing import Any, Optional

from fastapi import WebSocket
from google.genai import types

from app.services.conversation_service import ConversationService
from app.services.memory_tool_executor import MemoryToolExecutor
from app.services.memory_tool_schema import build_memory_tools
from app.services.user_context_builder import UserContext, UserContextBuilder
from app.services.memory_repository import (
    MemoryRepository,
    PostgresMemoryRepository,
)
from app.services.conversation_recorder import ConversationRecorder
from app.services.tool_call_recorder import NoopToolCallRecorder, ToolCallRecorder


class FutureReminderConversationService(ConversationService):
    """
    Realtime conversation service with memory retrieval + function calling.
    Rollback is possible by switching CONVERSATION_SERVICE_MODE.
    """

    def __init__(
        self,
        memory_repo: Optional[MemoryRepository] = None,
        tool_call_recorder: Optional[ToolCallRecorder] = None,
        conversation_recorder: Optional[ConversationRecorder] = None,
    ) -> None:
        super().__init__(conversation_recorder=conversation_recorder)
        self._memory_repo = memory_repo or PostgresMemoryRepository()
        self._tool_call_recorder = tool_call_recorder or NoopToolCallRecorder()
        self._tool_call_in_progress = False
        self._session_closed = False
        self._can_send_audio = asyncio.Event()
        self._can_send_audio.set()
        # Live session send operations must be serialized to avoid protocol violations.
        self._live_send_lock = asyncio.Lock()

    async def handle_conversation(
        self, websocket: WebSocket, user_id: Optional[int] = None
    ) -> None:
        """
        Connect to Gemini Live API and assign handlers for receiving and sending data.
        Set up tool calls for memory retrieval and storage, and include user context in system instructions.
        """
        if not user_id:
            await super().handle_conversation(websocket, user_id=None)
            return
        self._reset_live_session_state()
        context = await self._build_user_context(user_id)
        config = {
            "response_modalities": ["AUDIO"],
            "system_instruction": self._build_system_instruction(context),
            "input_audio_transcription": {},
            "output_audio_transcription": {},
            "tools": self._build_tools(),
        }

        try:
            async with self.client.aio.live.connect(
                model="gemini-2.5-flash-native-audio-preview-12-2025",
                config=config,
            ) as live_session:
                await self._send_start_prompt(live_session, context)
                await self._run_live_session_tasks(
                    self.receive_data(live_session, websocket, user_id),
                    self.send_audio(live_session, websocket),
                )
        except asyncio.CancelledError:
            pass

    def _reset_live_session_state(self) -> None:
        self._tool_call_in_progress = False
        self._session_closed = False
        self._can_send_audio.set()

    def _mark_session_closed(self) -> None:
        self._session_closed = True
        self._can_send_audio.set()

    async def receive_data(self, live_session, websocket: WebSocket, user_id: int):
        """Receive audio, text, and tool calls from the live session."""
        while True:
            try:
                turn = live_session.receive()
                async for response in turn:
                    if response.tool_call:
                        await self._handle_tool_call(
                            response.tool_call, live_session, user_id
                        )
                    if response.server_content:
                        await self._handle_live_session_content(
                            response.server_content, websocket, user_id
                        )
            except Exception as exc:  # pragma: no cover - realtime safety
                self._mark_session_closed()
                print(f"Error receiving audio: {exc}")
                break

    async def _on_input_transcription(self, user_id: int, text: str) -> None:
        if not self._is_memory_intent(text):
            return
        self._record_debug_event(
            user_id=user_id,
            name="memory_intent_detected",
            args={"text": text},
            output={"note": "user requested memory-like behavior"},
        )

    async def _handle_tool_call(self, tool_call, live_session, user_id: int) -> None:
        function_calls = tool_call.function_calls or []
        if not function_calls:
            return
        responses = []
        await self._pause_audio_for_tool_call()
        try:
            for call in function_calls:
                name = call.name or ""
                call_id = call.id
                args = self._normalize_tool_args(call.args or {})
                self._record_debug_event(
                    user_id=user_id,
                    name="tool_call_received",
                    args={"tool_name": name, "tool_call_id": call_id, "args": args},
                    output={"status": "received"},
                )
                result = await self._execute_tool_call(name, args, user_id)
                json_result = self._to_jsonable(result)
                self._tool_call_recorder.record_call(
                    user_id=user_id,
                    call_id=call_id,
                    name=name,
                    args=args,
                    output=json_result,
                )
                if not call_id:
                    print(f"Skipping tool response because call id is missing: {name}")
                    self._record_debug_event(
                        user_id=user_id,
                        name="tool_response_skipped",
                        args={"tool_name": name},
                        output={"reason": "missing_call_id"},
                    )
                    continue
                response_payload: dict[str, Any]
                if "error" in json_result:
                    response_payload = {"error": json_result["error"]}
                else:
                    response_payload = {"output": json_result}
                responses.append(
                    types.FunctionResponse(
                        name=name,
                        id=call_id,
                        response=response_payload,
                    )
                )
            if responses:
                async with self._live_send_lock:
                    await live_session.send_tool_response(function_responses=responses)
                self._record_debug_event(
                    user_id=user_id,
                    name="tool_response_sent",
                    args={"count": len(responses)},
                    output={"status": "ok"},
                )
        finally:
            self._resume_audio_after_tool_call()

    async def _pause_audio_for_tool_call(self) -> None:
        async with self._live_send_lock:
            self._tool_call_in_progress = True
            self._can_send_audio.clear()

    def _resume_audio_after_tool_call(self) -> None:
        self._tool_call_in_progress = False
        if not self._session_closed:
            self._can_send_audio.set()

    async def _execute_tool_call(
        self, name: str, args: dict[str, Any], user_id: int
    ) -> dict:
        return await MemoryToolExecutor(self._memory_repo).execute(name, args, user_id)

    async def _build_user_context(self, user_id: int) -> UserContext:
        return await UserContextBuilder(self._memory_repo).build(user_id)

    def _build_system_instruction(self, context: UserContext) -> str:
        return UserContextBuilder(self._memory_repo).build_system_instruction(context)

    async def _send_start_prompt(self, live_session, context: UserContext) -> None:
        prompt = (
            "Start the conversation now. If there is a recent special event, mention it first. "
            "Then ask if the user wants to talk about it or something else."
        )
        async with self._live_send_lock:
            await live_session.send_realtime_input(text=prompt)

    def _build_tools(self) -> list[types.Tool]:
        return build_memory_tools()

    def _normalize_tool_args(self, args: Any) -> dict[str, Any]:
        return MemoryToolExecutor.normalize_tool_args(args)

    def _to_jsonable(self, value: Any) -> Any:
        return MemoryToolExecutor.to_jsonable(value)

    def _is_memory_intent(self, text: str) -> bool:
        lowered = text.lower()
        keywords = [
            "\uae30\uc5b5\ud574",
            "\uae30\uc5b5 \ud574",
            "\uae30\uc5b5\ud574\uc918",
            "\uae30\uc5b5\ud574 \uc918",
            "\uc78a\uc9c0\ub9c8",
            "\uc78a\uc9c0 \ub9c8",
            "remember",
            "memorize",
        ]
        return any(keyword in lowered for keyword in keywords)

    def _record_debug_event(
        self,
        *,
        user_id: int,
        name: str,
        args: dict[str, Any],
        output: dict[str, Any],
    ) -> None:
        self._tool_call_recorder.record_call(
            user_id=user_id,
            call_id=None,
            name=name,
            args=self._to_jsonable(args),
            output=self._to_jsonable(output),
        )

    async def send_audio(self, live_session, websocket: WebSocket):
        """
        Send audio data from the client to Gemini Live API (uplink direction).
        This runs in a loop until the connection is closed.
        Wait if a tool call is in progress to avoid protocol violations,
        since Gemini expects tool calls to be atomic with no interleaving input.
        """
        while not self._session_closed:
            try:
                await self._can_send_audio.wait()
                chunk = await websocket.receive_bytes()
                msg = {"data": chunk, "mime_type": "audio/pcm;rate=16000"}
                async with self._live_send_lock:
                    if self._session_closed:
                        break
                    if not self._can_send_audio.is_set():
                        continue
                    await live_session.send_realtime_input(audio=msg)
            except Exception as e:
                self._mark_session_closed()
                print(f"Error sending audio: {e}")
                break

