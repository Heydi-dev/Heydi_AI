from __future__ import annotations

from datetime import date, datetime
from typing import Any

from app.core.config import settings
from app.services.memory_repository import MemoryRepository


class MemoryToolExecutor:
    def __init__(self, memory_repo: MemoryRepository) -> None:
        self._memory_repo = memory_repo
        self._tool_handlers = {
            "fetch_recent_special_events": self._tool_fetch_recent_special_events,
            "fetch_recent_diary_summaries": self._tool_fetch_recent_diary_summaries,
            "search_memory": self._tool_search_memory,
            "search_diary_summaries": self._tool_search_diary_summaries,
            "search_memory_by_period": self._tool_search_memory_by_period,
            "store_memory": self._tool_store_memory,
        }

    async def execute(self, name: str, args: dict[str, Any], user_id: int) -> dict:
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
        return {"stored": self.to_jsonable(stored)}

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
        memory_type = str(args.get("memory_type", "")).strip()
        content = str(args.get("content", "")).strip()
        title = args.get("title")
        event_date = args.get("event_date")
        importance = args.get("importance", 3)
        source = args.get("source", "AI")
        metadata = args.get("metadata")

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

    def _extract_query(self, args: dict[str, Any]) -> str:
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

    @staticmethod
    def normalize_tool_args(args: Any) -> dict[str, Any]:
        if isinstance(args, dict):
            return args
        try:
            return dict(args)
        except Exception:
            return {}

    @classmethod
    def to_jsonable(cls, value: Any) -> Any:
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        if isinstance(value, dict):
            return {k: cls.to_jsonable(v) for k, v in value.items()}
        if isinstance(value, list):
            return [cls.to_jsonable(v) for v in value]
        return value
