from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Optional

from fastapi import WebSocket
from google.genai import types

from app.core.config import settings
from app.services.conversation_service import ConversationService
from app.services.memory_repository import (
    MEMORY_TYPES,
    MemoryRepository,
    PostgresMemoryRepository,
)
from app.services.conversation_recorder import ConversationRecorder
from app.services.tool_call_recorder import NoopToolCallRecorder, ToolCallRecorder


@dataclass
class UserContext:
    user_id: int
    nickname: Optional[str]
    recent_special_events: list[dict]
    recent_diary_summaries: list[dict]
    recent_preferences: list[dict]
    recent_facts: list[dict]


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
        if not user_id:
            await super().handle_conversation(websocket, user_id=None)
            return
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
                async with asyncio.TaskGroup() as tg:
                    tg.create_task(self.receive_data(live_session, websocket, user_id))
                    tg.create_task(self.send_audio(live_session, websocket))
        except asyncio.CancelledError:
            pass

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
                        await self._handle_server_content(
                            response.server_content, websocket, user_id
                        )
            except Exception as exc:  # pragma: no cover - realtime safety
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
        self._tool_call_in_progress = True
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
            self._tool_call_in_progress = False

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
        profile = await self._memory_repo.fetch_user_profile(user_id)
        nickname = profile.get("nickname") if profile else None
        recent_special_events = await self._memory_repo.fetch_recent_special_events(
            user_id, settings.MEMORY_RECENT_SPECIAL_LIMIT
        )
        recent_diary_summaries = await self._memory_repo.fetch_recent_diary_summaries(
            user_id, settings.MEMORY_RECENT_DIARY_LIMIT
        )
        recent_preferences = await self._memory_repo.fetch_recent_memories(
            user_id, ["PREFERENCE"], settings.MEMORY_RECENT_PREFERENCE_LIMIT
        )
        recent_facts = await self._memory_repo.fetch_recent_memories(
            user_id,
            ["PERSON", "PLACE", "FACT", "GOAL", "RECENT_SUMMARY"],
            settings.MEMORY_RECENT_FACT_LIMIT,
        )
        return UserContext(
            user_id=user_id,
            nickname=nickname,
            recent_special_events=recent_special_events,
            recent_diary_summaries=recent_diary_summaries,
            recent_preferences=recent_preferences,
            recent_facts=recent_facts,
        )

    def _build_system_instruction(self, context: UserContext) -> str:
        lines = [
            "You are a warm, supportive AI assistant. Respond in natural Korean.",
            "You can use memory tools to recall or store user information.",
            "For store_memory, you must use this exact schema: memory_type, content, title, event_date, importance, source, metadata.",
            "Never use memory_key or memory_value. Those keys are invalid.",
            "At the start of the conversation, proactively mention a recent special event if available,",
            "and ask whether the user wants to talk about it or another topic.",
            "If you need more context, call search_memory or search_diary_summaries.",
            "When the user asks plans for a time period (e.g. next week), call search_memory_by_period.",
            "When you learn important personal info (events, preferences, people, places, goals, reminders),",
            "call store_memory.",
            "If you are unsure about a memory, confirm with the user rather than asserting it.",
            "Treat the user like a close friend, warm but not intrusive.",
        ]

        lines.append("\nKnown user context:")
        if context.nickname:
            lines.append(f"- Nickname: {context.nickname}")

        lines.append("- Recent special events:")
        lines.extend(self._format_context_rows(context.recent_special_events))

        lines.append("- Recent diary summaries:")
        lines.extend(self._format_context_rows(context.recent_diary_summaries))

        lines.append("- Preferences:")
        lines.extend(self._format_context_rows(context.recent_preferences))

        lines.append("- Personal facts:")
        lines.extend(self._format_context_rows(context.recent_facts))

        return "\n".join(lines)

    async def _send_start_prompt(self, live_session, context: UserContext) -> None:
        prompt = (
            "Start the conversation now. If there is a recent special event, mention it first. "
            "Then ask if the user wants to talk about it or something else."
        )
        async with self._live_send_lock:
            await live_session.send_realtime_input(text=prompt)

    def _build_tools(self) -> list[types.Tool]:
        tools = [
            types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(
                        name="fetch_recent_special_events",
                        description="Fetch recent special events for the current user.",
                        parameters_json_schema={
                            "type": "object",
                            "properties": {
                                "limit": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "maximum": 10,
                                    "description": "How many recent special events to return.",
                                }
                            },
                            "additionalProperties": False,
                        },
                        response_json_schema={
                            "type": "object",
                            "properties": {
                                "events": {"type": "array", "items": {"type": "object"}}
                            },
                        },
                    ),
                    types.FunctionDeclaration(
                        name="fetch_recent_diary_summaries",
                        description="Fetch recent diary summaries for the current user.",
                        parameters_json_schema={
                            "type": "object",
                            "properties": {
                                "limit": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "maximum": 20,
                                    "description": "How many recent diary summaries to return.",
                                }
                            },
                            "additionalProperties": False,
                        },
                        response_json_schema={
                            "type": "object",
                            "properties": {
                                "summaries": {"type": "array", "items": {"type": "object"}}
                            },
                        },
                    ),
                    types.FunctionDeclaration(
                        name="search_memory",
                        description="Search stored user memories by keyword.",
                        parameters_json_schema={
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "Search keyword to match against stored memories.",
                                },
                                "limit": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "maximum": 20,
                                    "description": "Maximum number of results to return.",
                                },
                                "types": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Optional list of memory types to filter by.",
                                },
                            },
                            "required": ["query"],
                            "additionalProperties": False,
                            "propertyOrdering": ["query", "types", "limit"],
                        },
                        response_json_schema={
                            "type": "object",
                            "properties": {
                                "results": {"type": "array", "items": {"type": "object"}}
                            },
                        },
                    ),
                    types.FunctionDeclaration(
                        name="search_diary_summaries",
                        description="Search diary summaries by keyword.",
                        parameters_json_schema={
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "Search keyword to match diary summaries.",
                                },
                                "limit": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "maximum": 20,
                                    "description": "Maximum number of summaries to return.",
                                },
                            },
                            "required": ["query"],
                            "additionalProperties": False,
                            "propertyOrdering": ["query", "limit"],
                        },
                        response_json_schema={
                            "type": "object",
                            "properties": {
                                "results": {"type": "array", "items": {"type": "object"}}
                            },
                        },
                    ),
                    types.FunctionDeclaration(
                        name="search_memory_by_period",
                        description=(
                            "Search memories with event_date within a date range. "
                            "Use this for schedule questions such as next week plans."
                        ),
                        parameters_json_schema={
                            "type": "object",
                            "properties": {
                                "start_date": {
                                    "type": "string",
                                    "description": "Start date in YYYY-MM-DD.",
                                },
                                "end_date": {
                                    "type": "string",
                                    "description": "End date in YYYY-MM-DD.",
                                },
                                "types": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": (
                                        "Optional memory types. Example: REMINDER, GOAL, SPECIAL_EVENT."
                                    ),
                                },
                                "limit": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "maximum": 20,
                                    "description": "Maximum number of results to return.",
                                },
                            },
                            "required": ["start_date", "end_date"],
                            "additionalProperties": False,
                            "propertyOrdering": [
                                "start_date",
                                "end_date",
                                "types",
                                "limit",
                            ],
                        },
                        response_json_schema={
                            "type": "object",
                            "properties": {
                                "results": {"type": "array", "items": {"type": "object"}}
                            },
                        },
                    ),
                    types.FunctionDeclaration(
                        name="store_memory",
                        description="Store a new memory about the user.",
                        parameters_json_schema={
                            "type": "object",
                            "properties": {
                                "memory_type": {
                                    "type": "string",
                                    "enum": sorted(MEMORY_TYPES),
                                    "description": "Category of memory to store.",
                                },
                                "title": {
                                    "type": "string",
                                    "description": "Short label or title for this memory.",
                                },
                                "content": {
                                    "type": "string",
                                    "description": "The full memory content to store. Required.",
                                },
                                "event_date": {
                                    "type": "string",
                                    "description": "YYYY-MM-DD",
                                },
                                "importance": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "maximum": 5,
                                    "description": "Importance score from 1 (low) to 5 (high).",
                                },
                                "source": {
                                    "type": "string",
                                    "enum": ["USER", "AI", "SYSTEM"],
                                    "description": "Who provided this memory.",
                                },
                                "metadata": {
                                    "type": "object",
                                    "description": "Optional structured metadata for the memory.",
                                },
                            },
                            "required": ["memory_type", "content"],
                            "additionalProperties": False,
                            "propertyOrdering": [
                                "memory_type",
                                "content",
                                "title",
                                "event_date",
                                "importance",
                                "source",
                                "metadata",
                            ],
                        },
                        response_json_schema={
                            "type": "object",
                            "properties": {"stored": {"type": "object"}},
                        },
                    ),
                ]
            )
        ]
        return tools

    def _format_context_rows(self, rows: list[dict]) -> list[str]:
        if not rows:
            return ["  (none)"]
        formatted = []
        for row in rows:
            formatted.append(f"  - {self._format_row(row)}")
        return formatted

    def _format_row(self, row: dict) -> str:
        event_date = self._format_date(row.get("event_date"))
        created_at = self._format_date(row.get("created_at"))
        summary = row.get("summary_one_line")
        content = row.get("content")
        title = row.get("title")
        parts = []
        if event_date:
            parts.append(event_date)
        if created_at and created_at != event_date:
            parts.append(created_at)
        if title:
            parts.append(str(title))
        if summary:
            parts.append(str(summary))
        if content and not summary:
            parts.append(str(content))
        return " | ".join(parts) if parts else str(row)

    def _serialize_rows(self, rows: list[dict]) -> list[dict]:
        serialized = []
        for row in rows:
            serialized.append({k: self._serialize_value(v) for k, v in row.items()})
        return serialized

    def _serialize_value(self, value: Any) -> Any:
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        return value

    def _format_date(self, value: Any) -> Optional[str]:
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        return str(value) if value else None

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
        """Pause uplink while tool calls are being processed."""
        while True:
            try:
                if self._tool_call_in_progress:
                    await asyncio.sleep(0.02)
                    continue
                chunk = await websocket.receive_bytes()
                msg = {"data": chunk, "mime_type": "audio/pcm"}
                async with self._live_send_lock:
                    await live_session.send_realtime_input(audio=msg)
            except Exception as e:
                print(f"Error sending audio: {e}")
                break

