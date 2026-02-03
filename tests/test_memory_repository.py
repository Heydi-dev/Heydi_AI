import os

import pytest

os.environ.setdefault(
    "DATABASE_URL", "postgresql+asyncpg://user:pass@localhost:5432/test"
)

from app.services.memory_repository import PostgresMemoryRepository  # noqa: E402


class DummyResult:
    def __init__(self, row=None, rows=None):
        self._row = row
        self._rows = rows or []

    def mappings(self):
        return self

    def first(self):
        return self._row

    def all(self):
        return self._rows


class DummySession:
    def __init__(self):
        self.executed = []
        self.committed = False

    async def execute(self, sql, params):
        self.executed.append((str(sql), params))
        return DummyResult()

    async def commit(self):
        self.committed = True

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


def make_repo():
    session = DummySession()

    def factory():
        return session

    return PostgresMemoryRepository(session_factory=factory), session


@pytest.mark.asyncio
async def test_search_memory_uses_ilike_and_optional_types():
    repo, session = make_repo()

    await repo.search_memory(
        user_id=1,
        query="테스트",
        limit=5,
        types=["preference", "fact"],
    )

    assert session.executed, "Expected a query to be executed."
    sql, params = session.executed[0]
    assert "ILIKE" in sql
    assert "memory_type IN" in sql
    assert params["q"] == "%테스트%"


@pytest.mark.asyncio
async def test_store_memory_inserts_and_commits():
    repo, session = make_repo()

    result = await repo.store_memory(
        user_id=1,
        memory_type="FACT",
        content="새로운 정보",
        importance=10,
    )

    assert session.committed is True
    sql, params = session.executed[0]
    assert "INSERT INTO user_memory" in sql
    assert params["importance"] == 5
    assert result == {"ok": True} or "memory_id" in result
