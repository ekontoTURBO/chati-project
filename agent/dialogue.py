"""
Dialogue buffer — short-term conversation memory as plain text.

Turns are `(role, text, timestamp, player_id)` per AUDIT.md section 5.
player_id is optional — we set it if we know who's talking (e.g. from
an OSC source or a name learned mid-conversation), else None for
anonymous players. Role is "player" or "chati".

No images or tool-call soup — just text. The prompter renders this
into the system prompt every call.
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Turn:
    role: str                     # "player" or "chati"
    text: str
    timestamp: float              # unix seconds
    player_id: Optional[str] = None  # optional speaker identity


class Dialogue:
    """Rolling window of N conversation turns."""

    def __init__(self, max_turns: int = 8):
        self.turns: deque[Turn] = deque(maxlen=max_turns)

    def add_player(self, text: str, player_id: Optional[str] = None) -> None:
        text = (text or "").strip()
        if text:
            self.turns.append(Turn("player", text, time.time(), player_id))

    def add_chati(self, text: str) -> None:
        text = (text or "").strip()
        if text:
            self.turns.append(Turn("chati", text, time.time(), None))

    def clear(self) -> None:
        self.turns.clear()

    def last_player_text(self) -> str:
        for t in reversed(self.turns):
            if t.role == "player":
                return t.text
        return ""

    def last_player_id(self) -> Optional[str]:
        """Most recent player speaker's id (for reply attribution)."""
        for t in reversed(self.turns):
            if t.role == "player" and t.player_id:
                return t.player_id
        return None

    def render(self) -> str:
        """Render for inclusion in the system prompt."""
        if not self.turns:
            return ""
        now = time.time()
        lines = []
        for t in self.turns:
            age = int(now - t.timestamp)
            if t.role == "chati":
                tag = "You said"
            else:
                who = t.player_id or "They"
                tag = f"{who} said"
            lines.append(f"[{age}s ago] {tag}: {t.text}")
        return "Recent conversation:\n" + "\n".join(lines)

    def __len__(self) -> int:
        return len(self.turns)
