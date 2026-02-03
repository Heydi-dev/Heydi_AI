import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

os.environ.setdefault(
    "DATABASE_URL", "postgresql+asyncpg://user:pass@localhost:5432/test"
)

from app.services.future_reminder_conversation_service import (  # noqa: E402
    FutureReminderConversationService,
    UserContext,
)


class DummyLiveSession:
    def __init__(self):
        self.sent = []
        self.tool_responses = []

    async def send_client_content(self, **kwargs):
        self.sent.append(kwargs)

    async def send_tool_response(self, **kwargs):
        self.tool_responses.append(kwargs)


@pytest.mark.asyncio
async def test_send_start_prompt_calls_live_session():
    service = FutureReminderConversationService()
    session = DummyLiveSession()
    context = UserContext(
        user_id=1,
        nickname="테스터",
        recent_special_events=[],
        recent_diary_summaries=[],
        recent_preferences=[],
        recent_facts=[],
    )

    await service._send_start_prompt(session, context)

    assert session.sent, "Expected a start prompt to be sent."
    payload = session.sent[0]
    assert payload["turns"][0]["parts"][0]["text"].startswith(
        "Start the conversation now"
    )


def test_build_system_instruction_includes_context():
    service = FutureReminderConversationService()
    context = UserContext(
        user_id=1,
        nickname="테스터",
        recent_special_events=[{"content": "팀 프로젝트 마감"}],
        recent_diary_summaries=[{"summary_one_line": "바쁜 하루"}],
        recent_preferences=[{"content": "커피"}],
        recent_facts=[{"content": "친한 동료가 있음"}],
    )

    instruction = service._build_system_instruction(context)

    assert "Nickname: 테스터" in instruction
    assert "바쁜 하루" in instruction
    assert "팀 프로젝트 마감" in instruction
    assert "커피" in instruction
    assert "친한 동료가 있음" in instruction


def test_build_tools_contains_expected_functions():
    service = FutureReminderConversationService()
    tools = service._build_tools()

    names = []
    for tool in tools:
        names.extend([decl.name for decl in tool.function_declarations or []])

    expected = {
        "fetch_recent_special_events",
        "fetch_recent_diary_summaries",
        "search_memory",
        "search_diary_summaries",
        "store_memory",
    }
    assert expected.issubset(set(names))


@pytest.mark.asyncio
async def test_execute_tool_call_fetches_special_events():
    service = FutureReminderConversationService()
    service._memory_repo = SimpleNamespace(
        fetch_recent_special_events=AsyncMock(
            return_value=[{"content": "회의"}]
        )
    )

    result = await service._execute_tool_call(
        "fetch_recent_special_events", {"limit": 2}, user_id=10
    )

    assert result["events"][0]["content"] == "회의"
    service._memory_repo.fetch_recent_special_events.assert_awaited_with(10, 2)


@pytest.mark.asyncio
async def test_execute_tool_call_store_memory():
    service = FutureReminderConversationService()
    service._memory_repo = SimpleNamespace(
        store_memory=AsyncMock(return_value={"memory_id": 99})
    )

    result = await service._execute_tool_call(
        "store_memory",
        {"memory_type": "FACT", "content": "새로운 정보"},
        user_id=10,
    )

    assert result["stored"]["memory_id"] == 99
    service._memory_repo.store_memory.assert_awaited()


@pytest.mark.asyncio
async def test_handle_tool_call_sends_function_response():
    service = FutureReminderConversationService()
    service._memory_repo = SimpleNamespace(
        fetch_recent_special_events=AsyncMock(return_value=[])
    )

    session = DummyLiveSession()
    tool_call = SimpleNamespace(
        function_calls=[
            SimpleNamespace(
                name="fetch_recent_special_events",
                id="call-1",
                args={"limit": 1},
            )
        ]
    )

    await service._handle_tool_call(tool_call, session, user_id=1)

    assert session.tool_responses, "Expected tool responses to be sent."
