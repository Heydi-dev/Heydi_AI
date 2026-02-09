from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


class ConversationRecorder:
    def record_input(self, user_id: int, text: str, finished: Optional[bool]) -> None:
        raise NotImplementedError

    def record_output(self, user_id: int, text: str, finished: Optional[bool]) -> None:
        raise NotImplementedError


class NoopConversationRecorder(ConversationRecorder):
    def record_input(self, user_id: int, text: str, finished: Optional[bool]) -> None:
        return

    def record_output(self, user_id: int, text: str, finished: Optional[bool]) -> None:
        return


@dataclass
class FileConversationRecorder(ConversationRecorder):
    path: str

    def record_input(self, user_id: int, text: str, finished: Optional[bool]) -> None:
        self._write("input", user_id, text, finished)

    def record_output(self, user_id: int, text: str, finished: Optional[bool]) -> None:
        self._write("output", user_id, text, finished)

    def _write(self, kind: str, user_id: int, text: str, finished: Optional[bool]) -> None:
        payload = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "type": kind,
            "user_id": user_id,
            "text": text,
            "finished": finished,
        }
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
