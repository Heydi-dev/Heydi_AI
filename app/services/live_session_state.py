from __future__ import annotations

import asyncio


class LiveSessionState:
    """Per-connection state for coordinating live session send operations."""

    def __init__(self) -> None:
        self.tool_call_in_progress = False
        self.session_closed = False
        self.can_send_audio = asyncio.Event()
        self.can_send_audio.set()
        self.send_lock = asyncio.Lock()

    def mark_closed(self) -> None:
        self.session_closed = True
        self.can_send_audio.set()

    async def pause_audio_for_tool_call(self) -> None:
        async with self.send_lock:
            self.tool_call_in_progress = True
            self.can_send_audio.clear()

    def resume_audio_after_tool_call(self) -> None:
        self.tool_call_in_progress = False
        if not self.session_closed:
            self.can_send_audio.set()
