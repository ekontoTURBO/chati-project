"""
Spatial Memory
==============
Tracks player positions and sound events over time so the agent
can reason about spatial context even when things move off-screen.

"The player was to my left 3 seconds ago" — that kind of info.
"""

import time
import logging
from dataclasses import dataclass, field
from collections import deque
from typing import Optional

logger = logging.getLogger("perception.spatial")


@dataclass
class PlayerSighting:
    """A single observation of a player at a point in time."""
    x: float  # screen position 0-1 (0=left, 1=right)
    y: float  # screen position 0-1 (0=top, 1=bottom)
    timestamp: float


@dataclass
class SoundEvent:
    """A speech/sound event with directional info."""
    text: str
    direction: float  # -1=left, 0=center, 1=right (from stereo DOA)
    energy: float  # loudness
    timestamp: float


class SpatialMemory:
    """Short-term memory of spatial events.

    Keeps a rolling window of player sightings and sound events.
    Provides queries like "where did that voice come from" and
    "where was the player last seen."
    """

    def __init__(self, window_seconds: float = 10.0):
        self.window = window_seconds
        self._player_sightings: deque[PlayerSighting] = deque(maxlen=100)
        self._sound_events: deque[SoundEvent] = deque(maxlen=20)

    def add_player_sighting(self, x: float, y: float) -> None:
        """Record a player seen at (x, y) normalized screen coords."""
        self._player_sightings.append(PlayerSighting(x, y, time.time()))
        self._prune()

    def add_sound_event(self, text: str, direction: float, energy: float) -> None:
        """Record a heard sound/speech with directional info."""
        self._sound_events.append(SoundEvent(text, direction, energy, time.time()))
        self._prune()

    def _prune(self) -> None:
        """Remove entries older than the window."""
        cutoff = time.time() - self.window
        while self._player_sightings and self._player_sightings[0].timestamp < cutoff:
            self._player_sightings.popleft()
        while self._sound_events and self._sound_events[0].timestamp < cutoff:
            self._sound_events.popleft()

    def last_player_position(self) -> Optional[PlayerSighting]:
        """Get the most recent player sighting."""
        self._prune()
        if self._player_sightings:
            return self._player_sightings[-1]
        return None

    def last_sound_direction(self) -> Optional[float]:
        """Get direction of the most recent sound event (-1..1)."""
        self._prune()
        if self._sound_events:
            return self._sound_events[-1].direction
        return None

    def recent_sound_direction(self, within_seconds: float = 3.0) -> Optional[float]:
        """Get sound direction if there was a recent sound event."""
        self._prune()
        if not self._sound_events:
            return None
        latest = self._sound_events[-1]
        if time.time() - latest.timestamp <= within_seconds:
            return latest.direction
        return None

    def describe_context(self) -> str:
        """Human-readable summary for LLM prompts."""
        self._prune()
        lines = []

        player = self.last_player_position()
        if player:
            age = time.time() - player.timestamp
            side = "left" if player.x < 0.4 else ("right" if player.x > 0.6 else "center")
            if age < 1.0:
                lines.append(f"Player currently on your {side} (x={player.x:.2f})")
            elif age < 5.0:
                lines.append(f"Player was on your {side} ~{age:.0f}s ago")

        sound_dir = self.recent_sound_direction(within_seconds=5.0)
        if sound_dir is not None:
            if abs(sound_dir) < 0.15:
                dir_desc = "in front of you"
            elif sound_dir < -0.3:
                dir_desc = "from your left"
            elif sound_dir > 0.3:
                dir_desc = "from your right"
            else:
                dir_desc = "nearby"
            lines.append(f"Voice came {dir_desc}")

        return "\n".join(lines) if lines else ""

    def get_approach_direction(self) -> Optional[str]:
        """Determine which way to turn to face the last known player/sound.

        Returns: 'left', 'right', 'forward', or None
        """
        # Prioritize recent sound direction
        sound_dir = self.recent_sound_direction(within_seconds=5.0)
        if sound_dir is not None:
            if sound_dir < -0.3:
                return "left"
            elif sound_dir > 0.3:
                return "right"
            else:
                return "forward"

        # Fall back to last player sighting
        player = self.last_player_position()
        if player:
            if player.x < 0.35:
                return "left"
            elif player.x > 0.65:
                return "right"
            else:
                return "forward"

        return None
