from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


class ToolCallRecorder:
    def record_call(
        self,
        *,
        user_id: int,
        call_id: Optional[str],
        name: str,
        args: dict[str, Any],
        output: dict[str, Any],
    ) -> None:
        raise NotImplementedError


class NoopToolCallRecorder(ToolCallRecorder):
    def record_call(
        self,
        *,
        user_id: int,
        call_id: Optional[str],
        name: str,
        args: dict[str, Any],
        output: dict[str, Any],
    ) -> None:
        return


@dataclass
class FileToolCallRecorder(ToolCallRecorder):
    path: str

    def record_call(
        self,
        *,
        user_id: int,
        call_id: Optional[str],
        name: str,
        args: dict[str, Any],
        output: dict[str, Any],
    ) -> None:
        payload = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "user_id": user_id,
            "call_id": call_id,
            "name": name,
            "args": args,
            "output": output,
        }
        self._ensure_parent()
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")

    def _ensure_parent(self) -> None:
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
