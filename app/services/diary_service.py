from __future__ import annotations

from typing import Iterable

from google import genai
from google.genai import types


class DiaryService:
    """Generate diary entries and diary summaries."""

    def __init__(self) -> None:
        self.client = genai.Client()

    def _format_turns_for_diary(self, turns: Iterable) -> str:
        formatted_lines = []
        for turn in turns:
            role = getattr(turn, "role", None)
            if role is None and isinstance(turn, dict):
                role = turn.get("role")
            text = getattr(turn, "text", None)
            if text is None and isinstance(turn, dict):
                text = turn.get("text")
            if not text:
                continue
            role_label = "사용자" if role == "user" else "AI"
            formatted_lines.append(f"{role_label}: {text}")
        return "\n".join(formatted_lines)

    def generate_diary(self, turns: Iterable) -> str:
        conversation = self._format_turns_for_diary(turns)
        if not conversation:
            return ""
        try:
            response = self.client.models.generate_content(
                model="gemini-3.1-flash-lite",
                config=types.GenerateContentConfig(
                    system_instruction=(
                        "당신은 사용자의 대화를 바탕으로 일기를 작성하는 도우미입니다. "
                        "아래 대화 내용을 참고해 사용자가 직접 쓴 것 같은 자연스러운 한국어 일기를 작성하세요. "
                        "일기 본문만 작성하고 제목이나 리스트, 말머리는 쓰지 마세요. "
                        "문체는 일기체로 작성하고 감정 표현을 포함하세요."
                    ),
                ),
                contents=conversation,
            )
        except Exception as e:
            print(f"Error generating diary: {e}")
            return ""
        return response.text or ""

    def summarize_diary(self, diary_text: str) -> str:
        if not diary_text:
            return ""

        response = self.client.models.generate_content(
            model="gemini-3.1-flash-lite",
            config=types.GenerateContentConfig(
                system_instruction=(
                    "당신은 일기 내용을 한 줄로 요약하는 도우미입니다. "
                    "아래 일기에서 핵심만 뽑아 한국어 한 문장으로 요약하세요. "
                    "줄바꿈, 따옴표, 말머리 없이 한 줄로만 작성하세요."
                ),
            ),
            contents=diary_text,
        )

        summary = (response.text or "").replace("\r", " ").replace("\n", " ").strip()
        if len(summary) > 120:
            summary = summary[:117].rstrip() + "..."
        return summary
