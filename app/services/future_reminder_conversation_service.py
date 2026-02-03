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

    def __init__(self, memory_repo: Optional[MemoryRepository] = None) -> None:
        super().__init__()
        self._memory_repo = memory_repo or PostgresMemoryRepository()

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
                            response.server_content, websocket
                        )
            except Exception as exc:  # pragma: no cover - realtime safety
                print(f"Error receiving audio: {exc}")
                break

    async def _handle_server_content(self, server_content, websocket: WebSocket) -> None:
        if server_content.model_turn:
            for part in server_content.model_turn.parts:
                if part.inline_data and isinstance(part.inline_data.data, bytes):
                    await websocket.send_bytes(part.inline_data.data)
        if server_content.output_transcription:
            transcription = server_content.output_transcription
            await websocket.send_json(
                {
                    "type": "output",
                    "transcription": transcription.text,
                    "finished": transcription.finished,
                }
            )
        if server_content.input_transcription:
            transcription = server_content.input_transcription
            await websocket.send_json(
                {
                    "type": "input",
                    "transcription": transcription.text,
                    "finished": transcription.finished,
                }
            )

    async def _handle_tool_call(self, tool_call, live_session, user_id: int) -> None:
        function_calls = tool_call.function_calls or []
        if not function_calls:
            return
        responses = []
        for call in function_calls:
            name = call.name or ""
            call_id = call.id
            args = call.args or {}
            result = await self._execute_tool_call(name, args, user_id)
            responses.append(
                types.FunctionResponse(
                    name=name,
                    id=call_id,
                    response={"output": result},
                )
            )
        await live_session.send_tool_response(function_responses=responses)

    async def _execute_tool_call(
        self, name: str, args: dict[str, Any], user_id: int
    ) -> dict:
        name = name.strip()
        try:
            if name == "fetch_recent_special_events":
                limit = self._clamp_limit(args.get("limit"), settings.MEMORY_RECENT_SPECIAL_LIMIT)
                events = await self._memory_repo.fetch_recent_special_events(
                    user_id, limit
                )
                return {"events": self._serialize_rows(events)}
            if name == "fetch_recent_diary_summaries":
                limit = self._clamp_limit(args.get("limit"), settings.MEMORY_RECENT_DIARY_LIMIT)
                summaries = await self._memory_repo.fetch_recent_diary_summaries(
                    user_id, limit
                )
                return {"summaries": self._serialize_rows(summaries)}
            if name == "search_memory":
                query = str(args.get("query", "")).strip()
                types_filter = args.get("types")
                limit = self._clamp_limit(args.get("limit"), settings.MEMORY_DEFAULT_SEARCH_LIMIT)
                results = await self._memory_repo.search_memory(
                    user_id, query, limit, types_filter
                )
                return {"results": self._serialize_rows(results)}
            if name == "search_diary_summaries":
                query = str(args.get("query", "")).strip()
                limit = self._clamp_limit(args.get("limit"), settings.MEMORY_DEFAULT_SEARCH_LIMIT)
                results = await self._memory_repo.search_diary_summaries(
                    user_id, query, limit
                )
                return {"results": self._serialize_rows(results)}
            if name == "store_memory":
                memory_type = str(args.get("memory_type", "")).strip()
                content = str(args.get("content", "")).strip()
                title = args.get("title")
                event_date = args.get("event_date")
                importance = args.get("importance", 3)
                source = args.get("source", "AI")
                metadata = args.get("metadata")
                stored = await self._memory_repo.store_memory(
                    user_id,
                    memory_type,
                    content,
                    title=title,
                    event_date=event_date,
                    importance=importance,
                    source=source,
                    metadata=metadata,
                )
                return {"stored": stored}
        except Exception as exc:
            return {"error": str(exc)}
        return {"error": "unknown_tool"}

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
            "At the start of the conversation, proactively mention a recent special event if available,",
            "and ask whether the user wants to talk about it or another topic.",
            "If you need more context, call search_memory or search_diary_summaries.",
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
        await live_session.send_client_content(
            turns=[{"role": "user", "parts": [{"text": prompt}]}],
            turn_complete=True,
        )

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
                                "limit": {"type": "integer", "minimum": 1, "maximum": 10}
                            },
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
                                "limit": {"type": "integer", "minimum": 1, "maximum": 20}
                            },
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
                                "query": {"type": "string"},
                                "limit": {"type": "integer", "minimum": 1, "maximum": 20},
                                "types": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": ["query"],
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
                                "query": {"type": "string"},
                                "limit": {"type": "integer", "minimum": 1, "maximum": 20},
                            },
                            "required": ["query"],
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
                                },
                                "title": {"type": "string"},
                                "content": {"type": "string"},
                                "event_date": {
                                    "type": "string",
                                    "description": "YYYY-MM-DD",
                                },
                                "importance": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "maximum": 5,
                                },
                                "source": {
                                    "type": "string",
                                    "enum": ["USER", "AI", "SYSTEM"],
                                },
                                "metadata": {"type": "object"},
                            },
                            "required": ["memory_type", "content"],
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
