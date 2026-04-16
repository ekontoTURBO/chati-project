"""
Signals — central read-only snapshot for the agent loop.

Each perception subsystem (scene_analyzer, audio_capture, video_capture)
already runs on its own thread with thread-safe getters. This module
aggregates them into ONE object the controller reads from each tick,
instead of polling 4 subsystems separately.

This is the kimjammer / Neuro-sama pattern — perception writes to a
central state, prompter reads from a single snapshot.

Signals does NOT own the subsystems. It gets references and pulls
fresh data on `.snapshot()`. That keeps the data flow one-directional.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Optional

from perception.scene_analyzer import PerceptionState

logger = logging.getLogger("agent.signals")


@dataclass
class SignalsSnapshot:
    """A frozen moment-in-time view the controller reasons about."""
    timestamp: float
    perception: PerceptionState
    latest_speech: str           # may be "" if none this tick
    frame_b64: Optional[str]     # current video frame as base64 JPEG
    tts_playing: bool            # True if TTS is currently outputting
    chunks_consumed: int         # how many speech chunks we drained this tick


class SignalsHub:
    """Aggregates reads from the perception subsystems into one snapshot."""

    def __init__(self, scene_analyzer, audio_capture, video_capture, tts_router):
        self.scene = scene_analyzer
        self.audio = audio_capture
        self.video = video_capture
        self.tts = tts_router

    def snapshot(self) -> SignalsSnapshot:
        """Take a snapshot — called once per main loop tick."""
        chunks = self.audio.get_all_chunks()
        latest_speech = ""
        if chunks:
            # Use the most recent transcription
            latest_speech = (chunks[-1].get("text") or "").strip()

        return SignalsSnapshot(
            timestamp=time.time(),
            perception=self.scene.get_state(),
            latest_speech=latest_speech,
            frame_b64=self.video.get_latest_frame_b64(),
            tts_playing=self.tts.is_playing if hasattr(self.tts, "is_playing") else False,
            chunks_consumed=len(chunks),
        )
