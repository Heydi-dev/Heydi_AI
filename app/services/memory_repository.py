from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterable, Optional, Protocol

from sqlalchemy import text

from app.core.database import AsyncSessionLocal


MEMORY_TYPES = {
    "RECENT_SUMMARY",
    "SPECIAL_EVENT",
    "PREFERENCE",
    "PERSON",
    "PLACE",
    "GOAL",
    "FACT",
    "REMINDER",
    "OTHER",
}

SOURCE_TYPES = {"USER", "AI", "SYSTEM"}


@dataclass
class MemoryItem:
    memory_id: int
    memory_type: str
    title: Optional[str]
    content: str
    event_date: Optional[date]
    importance: int
    source: str
    metadata: Optional[dict]
    created_at: datetime


class MemoryRepository(Protocol):
    async def fetch_user_profile(self, user_id: int) -> dict:
        ...

    async def fetch_recent_special_events(self, user_id: int, limit: int) -> list[dict]:
        ...

    async def fetch_recent_diary_summaries(self, user_id: int, limit: int) -> list[dict]:
        ...

    async def fetch_recent_memories(
        self, user_id: int, types: Iterable[str], limit: int
    ) -> list[dict]:
        ...

    async def search_memory(
        self,
        user_id: int,
        query: str,
        limit: int,
        types: Optional[Iterable[str]] = None,
    ) -> list[dict]:
        ...

    async def search_diary_summaries(
        self,
        user_id: int,
        query: str,
        limit: int,
    ) -> list[dict]:
        ...

    async def store_memory(
        self,
        user_id: int,
        memory_type: str,
        content: str,
        title: Optional[str] = None,
        event_date: Optional[str] = None,
        importance: int = 3,
        source: str = "AI",
        metadata: Optional[dict] = None,
    ) -> dict:
        ...


class PostgresMemoryRepository:
    def __init__(self, session_factory=AsyncSessionLocal) -> None:
        self._session_factory = session_factory

    async def fetch_user_profile(self, user_id: int) -> dict:
        sql = text(
            """
            SELECT u.user_id, u.username, u.nickname,
                   p.reminder_enabled, p.reminder_time
            FROM users u
            LEFT JOIN user_profile p ON u.user_id = p.user_id
            WHERE u.user_id = :user_id
            """
        )
        async with self._session_factory() as session:
            result = await session.execute(sql, {"user_id": user_id})
            row = result.mappings().first()
            return dict(row) if row else {}

    async def fetch_recent_special_events(self, user_id: int, limit: int) -> list[dict]:
        sql = text(
            """
            SELECT memory_id, memory_type, title, content, event_date,
                   importance, source, metadata, created_at
            FROM user_memory
            WHERE user_id = :user_id
              AND deleted_at IS NULL
              AND memory_type = 'SPECIAL_EVENT'
            ORDER BY COALESCE(event_date, created_at) DESC
            LIMIT :limit
            """
        )
        async with self._session_factory() as session:
            result = await session.execute(
                sql, {"user_id": user_id, "limit": limit}
            )
            return [dict(row) for row in result.mappings().all()]

    async def fetch_recent_diary_summaries(self, user_id: int, limit: int) -> list[dict]:
        sql = text(
            """
            SELECT diary_id, created_at, summary_one_line
            FROM diary
            WHERE user_id = :user_id
              AND deleted_at IS NULL
              AND summary_one_line IS NOT NULL
              AND summary_one_line <> ''
            ORDER BY created_at DESC
            LIMIT :limit
            """
        )
        async with self._session_factory() as session:
            result = await session.execute(
                sql, {"user_id": user_id, "limit": limit}
            )
            return [dict(row) for row in result.mappings().all()]

    async def fetch_recent_memories(
        self, user_id: int, types: Iterable[str], limit: int
    ) -> list[dict]:
        types_list = [t for t in (t.upper() for t in types) if t in MEMORY_TYPES]
        if not types_list:
            return []
        types_clause = ", ".join(f"'{t}'" for t in types_list)
        sql = text(
            f"""
            SELECT memory_id, memory_type, title, content, event_date,
                   importance, source, metadata, created_at
            FROM user_memory
            WHERE user_id = :user_id
              AND deleted_at IS NULL
              AND memory_type IN ({types_clause})
            ORDER BY COALESCE(last_mentioned_at, created_at) DESC
            LIMIT :limit
            """
        )
        async with self._session_factory() as session:
            result = await session.execute(
                sql,
                {"user_id": user_id, "limit": limit},
            )
            return [dict(row) for row in result.mappings().all()]

    async def search_memory(
        self,
        user_id: int,
        query: str,
        limit: int,
        types: Optional[Iterable[str]] = None,
    ) -> list[dict]:
        if not query:
            return []
        sql = (
            """
            SELECT memory_id, memory_type, title, content, event_date,
                   importance, source, metadata, created_at
            FROM user_memory
            WHERE user_id = :user_id
              AND deleted_at IS NULL
              AND (title ILIKE :q OR content ILIKE :q)
            """
        )
        params = {"user_id": user_id, "q": f"%{query}%", "limit": limit}
        types_list = None
        if types:
            types_list = [t for t in (t.upper() for t in types) if t in MEMORY_TYPES]
        if types_list:
            types_clause = ", ".join(f"'{t}'" for t in types_list)
            sql += f" AND memory_type IN ({types_clause})"
        sql += " ORDER BY COALESCE(last_mentioned_at, created_at) DESC LIMIT :limit"
        async with self._session_factory() as session:
            result = await session.execute(text(sql), params)
            return [dict(row) for row in result.mappings().all()]

    async def search_diary_summaries(
        self,
        user_id: int,
        query: str,
        limit: int,
    ) -> list[dict]:
        if not query:
            return []
        sql = text(
            """
            SELECT diary_id, created_at, summary_one_line
            FROM diary
            WHERE user_id = :user_id
              AND deleted_at IS NULL
              AND summary_one_line ILIKE :q
            ORDER BY created_at DESC
            LIMIT :limit
            """
        )
        async with self._session_factory() as session:
            result = await session.execute(
                sql, {"user_id": user_id, "q": f"%{query}%", "limit": limit}
            )
            return [dict(row) for row in result.mappings().all()]

    async def store_memory(
        self,
        user_id: int,
        memory_type: str,
        content: str,
        title: Optional[str] = None,
        event_date: Optional[str] = None,
        importance: int = 3,
        source: str = "AI",
        metadata: Optional[dict] = None,
    ) -> dict:
        if not content:
            return {"error": "content_required"}
        normalized_type = self._normalize_memory_type(memory_type)
        normalized_source = self._normalize_source(source)
        parsed_date = self._parse_date(event_date)
        safe_importance = self._clamp_importance(importance)
        metadata_json = json.dumps(metadata) if metadata else None
        sql = text(
            """
            INSERT INTO user_memory (
                user_id, memory_type, title, content, event_date,
                importance, source, metadata
            )
            VALUES (
                :user_id, :memory_type, :title, :content, :event_date,
                :importance, :source, CAST(:metadata AS JSONB)
            )
            RETURNING memory_id, created_at
            """
        )
        async with self._session_factory() as session:
            result = await session.execute(
                sql,
                {
                    "user_id": user_id,
                    "memory_type": normalized_type,
                    "title": title,
                    "content": content,
                    "event_date": parsed_date,
                    "importance": safe_importance,
                    "source": normalized_source,
                    "metadata": metadata_json,
                },
            )
            await session.commit()
            row = result.mappings().first()
            return dict(row) if row else {"ok": True}

    def _normalize_memory_type(self, memory_type: Optional[str]) -> str:
        if not memory_type:
            return "OTHER"
        value = memory_type.strip().upper()
        return value if value in MEMORY_TYPES else "OTHER"

    def _normalize_source(self, source: Optional[str]) -> str:
        if not source:
            return "AI"
        value = source.strip().upper()
        return value if value in SOURCE_TYPES else "AI"

    def _parse_date(self, value: Optional[str]) -> Optional[date]:
        if not value:
            return None
        if isinstance(value, date):
            return value
        for fmt in ("%Y-%m-%d", "%Y.%m.%d", "%Y/%m/%d"):
            try:
                return datetime.strptime(str(value), fmt).date()
            except ValueError:
                continue
        return None

    def _clamp_importance(self, value: Optional[object]) -> int:
        if value is None:
            return 3
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return 3
        return min(max(parsed, 1), 5)
