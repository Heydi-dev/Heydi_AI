from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Optional

from app.core.config import settings
from app.services.memory_repository import MemoryRepository


@dataclass
class UserContext:
    """User information and recent memory snippets for a live conversation."""

    user_id: int
    nickname: Optional[str]
    recent_special_events: list[dict]
    recent_diary_summaries: list[dict]
    recent_preferences: list[dict]
    recent_facts: list[dict]


class UserContextBuilder:
    def __init__(self, memory_repo: MemoryRepository) -> None:
        self._memory_repo = memory_repo

    async def build(self, user_id: int) -> UserContext:
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

    def build_system_instruction(self, context: UserContext) -> str:
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

    def _format_context_rows(self, rows: list[dict]) -> list[str]:
        if not rows:
            return ["  (none)"]
        return [f"  - {self._format_row(row)}" for row in rows]

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

    def _format_date(self, value: Any) -> Optional[str]:
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        return str(value) if value else None
