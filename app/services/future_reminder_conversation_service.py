from __future__ import annotations

import asyncio
from datetime import date, datetime
from typing import Any, Optional

from fastapi import WebSocket
from google.genai import types

from app.core.config import settings
from app.services.conversation_service import ConversationService
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
        self._tool_handlers = {
            "fetch_recent_special_events": self._tool_fetch_recent_special_events,
            "fetch_recent_diary_summaries": self._tool_fetch_recent_diary_summaries,
            "search_memory": self._tool_search_memory,
            "search_diary_summaries": self._tool_search_diary_summaries,
            "search_memory_by_period": self._tool_search_memory_by_period,
            "store_memory": self._tool_store_memory,
        }

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
        handler = self._tool_handlers.get(name.strip())
        if handler is None:
            return {"error": "unknown_tool"}
        try:
            return await handler(user_id, args)
        except Exception as exc:
            return {"error": str(exc)}

    async def _tool_fetch_recent_special_events(
        self, user_id: int, args: dict[str, Any]
    ) -> dict:
        limit = self._clamp_limit(
            args.get("limit"), settings.MEMORY_RECENT_SPECIAL_LIMIT
        )
        events = await self._memory_repo.fetch_recent_special_events(user_id, limit)
        return {"events": self._serialize_rows(events)}

    async def _tool_fetch_recent_diary_summaries(
        self, user_id: int, args: dict[str, Any]
    ) -> dict:
        limit = self._clamp_limit(args.get("limit"), settings.MEMORY_RECENT_DIARY_LIMIT)
        summaries = await self._memory_repo.fetch_recent_diary_summaries(user_id, limit)
        return {"summaries": self._serialize_rows(summaries)}

    async def _tool_search_memory(self, user_id: int, args: dict[str, Any]) -> dict:
        query = self._extract_query(args)
        types_filter = args.get("types")
        limit = self._clamp_limit(args.get("limit"), settings.MEMORY_DEFAULT_SEARCH_LIMIT)
        results = await self._memory_repo.search_memory(
            user_id, query, limit, types_filter
        )
        return {"results": self._serialize_rows(results)}

    async def _tool_search_diary_summaries(
        self, user_id: int, args: dict[str, Any]
    ) -> dict:
        query = self._extract_query(args)
        limit = self._clamp_limit(args.get("limit"), settings.MEMORY_DEFAULT_SEARCH_LIMIT)
        results = await self._memory_repo.search_diary_summaries(user_id, query, limit)
        return {"results": self._serialize_rows(results)}

    async def _tool_search_memory_by_period(
        self, user_id: int, args: dict[str, Any]
    ) -> dict:
        start_date = str(args.get("start_date", "")).strip()
        end_date = str(args.get("end_date", "")).strip()
        types_filter = args.get("types")
        limit = self._clamp_limit(args.get("limit"), settings.MEMORY_DEFAULT_SEARCH_LIMIT)
        results = await self._memory_repo.search_memory_by_period(
            user_id, start_date, end_date, limit, types_filter
        )
        return {"results": self._serialize_rows(results)}

    async def _tool_store_memory(self, user_id: int, args: dict[str, Any]) -> dict:
        normalized_args = self._normalize_store_memory_args(args)
        stored = await self._memory_repo.store_memory(
            user_id,
            normalized_args["memory_type"],
            normalized_args["content"],
            title=normalized_args["title"],
            event_date=normalized_args["event_date"],
            importance=normalized_args["importance"],
            source=normalized_args["source"],
            metadata=normalized_args["metadata"],
        )
        return {"stored": self._to_jsonable(stored)}

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

    def _serialize_rows(self, rows: list[dict]) -> list[dict]:
        serialized = []
        for row in rows:
            serialized.append({k: self._serialize_value(v) for k, v in row.items()})
        return serialized

    def _serialize_value(self, value: Any) -> Any:
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        return value

    def _clamp_limit(self, value: Any, default: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default
        return min(max(parsed, 1), 20)

    def _normalize_store_memory_args(self, args: dict[str, Any]) -> dict[str, Any]:
        """Normalize model tool args so minor key drift does not drop memory writes."""
        memory_type = str(args.get("memory_type", "")).strip()
        content = str(args.get("content", "")).strip()
        title = args.get("title")
        event_date = args.get("event_date")
        importance = args.get("importance", 3)
        source = args.get("source", "AI")
        metadata = args.get("metadata")

        # Compatibility aliases occasionally produced by the model.
        if not content:
            content = str(args.get("memory_value", "")).strip()
        if not title:
            alias_title = args.get("memory_key")
            title = str(alias_title).strip() if alias_title is not None else None
        if not memory_type:
            memory_type = "PREFERENCE"

        return {
            "memory_type": memory_type,
            "content": content,
            "title": title,
            "event_date": event_date,
            "importance": importance,
            "source": source,
            "metadata": metadata,
        }

    def _normalize_tool_args(self, args: Any) -> dict[str, Any]:
        """Convert tool args into a plain dict to avoid SDK container edge cases."""
        if isinstance(args, dict):
            return args
        try:
            return dict(args)
        except Exception:
            return {}

    def _extract_query(self, args: dict[str, Any]) -> str:
        """Support common model drift: query/keyword/keywords."""
        query = args.get("query")
        if isinstance(query, str) and query.strip():
            return query.strip()
        keyword = args.get("keyword")
        if isinstance(keyword, str) and keyword.strip():
            return keyword.strip()
        keywords = args.get("keywords")
        if isinstance(keywords, list):
            tokens = [str(item).strip() for item in keywords if str(item).strip()]
            if tokens:
                return " ".join(tokens)
        return ""

    def _to_jsonable(self, value: Any) -> Any:
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        if isinstance(value, dict):
            return {k: self._to_jsonable(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._to_jsonable(v) for v in value]
        return value

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

