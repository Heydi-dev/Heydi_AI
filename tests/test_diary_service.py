from types import SimpleNamespace

from app.services.diary_service import DiaryService


def _build_service() -> DiaryService:
    return DiaryService.__new__(DiaryService)


def test_format_turns_for_diary_accepts_objects_and_dicts():
    service = _build_service()
    turns = [
        SimpleNamespace(role="user", text="오늘은 산책했다."),
        {"role": "assistant", "text": "좋았겠어요."},
        {"role": "user", "text": ""},
    ]

    formatted = service._format_turns_for_diary(turns)

    assert formatted == "사용자: 오늘은 산책했다.\nAI: 좋았겠어요."


def test_generate_diary_returns_empty_for_empty_turns():
    service = _build_service()

    assert service.generate_diary([]) == ""


def test_summarize_diary_returns_empty_for_empty_text():
    service = _build_service()

    assert service.summarize_diary("") == ""
