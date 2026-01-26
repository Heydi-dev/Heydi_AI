import json
import re
from typing import Iterable, Optional

from google import genai
from google.genai import types
from pydantic import BaseModel, Field


class MonthlyCommentResponse(BaseModel):
    comment: str = Field(description="Generated comment in Korean.")


class MonthlyCommentService:
    def __init__(self) -> None:
        self._client = genai.Client()

    def generate_activity_comment(self, entries: list[dict]) -> str:
        instruction = (
            "You are a monthly diary assistant. Read the user's monthly diary entries "
            "and write a natural Korean comment about the activities they did often. "
            "Mention concrete activities from the entries. If activities are unclear, "
            "say the records are limited and keep it gentle. Respond in 2-3 sentences. "
            "Return JSON only with {\"comment\": \"...\"}."
        )
        return self._generate_comment(entries, instruction)

    def generate_feedback_comment(self, entries: list[dict]) -> str:
        instruction = (
            "You are a supportive coach. Read the user's monthly diary entries and "
            "write a Korean comment with insight and actionable feedback for next month. "
            "Base the advice on the entries without harsh judgment. If the data is thin, "
            "offer light, general guidance. Respond in 2-3 sentences. "
            "Return JSON only with {\"comment\": \"...\"}."
        )
        return self._generate_comment(entries, instruction)

    def _generate_comment(self, entries: list[dict], instruction: str) -> str:
        formatted = self._format_entries(entries)
        if not formatted:
            return ""

        response = self._client.models.generate_content(
            model="gemini-2.5-flash",
            config=types.GenerateContentConfig(
                system_instruction=instruction,
                temperature=0.4,
                response_mime_type="application/json",
                response_json_schema=MonthlyCommentResponse.model_json_schema(),
            ),
            contents=formatted,
        )
        text = (response.text or "").strip()
        parsed = self._parse_json(text)
        if not parsed or "comment" not in parsed:
            return ""
        return str(parsed.get("comment", "")).strip()

    def _format_entries(self, entries: Iterable[dict]) -> str:
        lines = []
        for entry in entries:
            if not entry:
                continue
            date = entry.get("date") if isinstance(entry, dict) else None
            text = entry.get("text") if isinstance(entry, dict) else None
            if not date or not text:
                continue
            lines.append(f"[{date}] {text}")
        return "\n".join(lines)

    def _parse_json(self, text: str) -> Optional[dict]:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if not match:
                return None
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
