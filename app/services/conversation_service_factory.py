from __future__ import annotations

from app.core.config import settings
from app.services.conversation_recorder import (
    FileConversationRecorder,
    NoopConversationRecorder,
)
from app.services.conversation_service import ConversationService
from app.services.future_reminder_conversation_service import (
    FutureReminderConversationService,
)
from app.services.tool_call_recorder import FileToolCallRecorder, NoopToolCallRecorder


def build_conversation_service() -> ConversationService:
    conversation_recorder = _build_conversation_recorder()
    if settings.CONVERSATION_SERVICE_MODE == "future_reminder":
        return FutureReminderConversationService(
            tool_call_recorder=_build_tool_call_recorder(),
            conversation_recorder=conversation_recorder,
        )
    return ConversationService(conversation_recorder=conversation_recorder)


def _build_tool_call_recorder():
    if settings.TOOL_CALL_LOG_PATH:
        return FileToolCallRecorder(settings.TOOL_CALL_LOG_PATH)
    return NoopToolCallRecorder()


def _build_conversation_recorder():
    if settings.CONVERSATION_LOG_PATH:
        return FileConversationRecorder(settings.CONVERSATION_LOG_PATH)
    return NoopConversationRecorder()
