"""
Agent State Machine — 3 states with hysteresis.

Based on AUDIT.md section 6.

States:
- SOLO: no player engagement, roaming/idle
- ENGAGED: near a player or in conversation (collapses old APPROACHING+SOCIALIZING+CONVERSING)
- FOLLOWING: explicit "follow me" command

Transitions require sustained conditions, not single-frame flickers.
"""

import time
import logging
from enum import Enum, auto
from dataclasses import dataclass, field

logger = logging.getLogger("agent.state")


class AgentState(Enum):
    SOLO = auto()
    ENGAGED = auto()
    FOLLOWING = auto()


# Hysteresis thresholds
ENTER_ENGAGED_FRAMES = 3        # ~3s of seeing someone before ENGAGED
EXIT_ENGAGED_SILENCE = 20.0     # 20s no speech AND
EXIT_ENGAGED_NO_PLAYER = 8.0    # 8s no player visible to leave ENGAGED
FOLLOW_TIMEOUT = 30.0           # 30s no player sighting to drop FOLLOWING


@dataclass
class StateContext:
    """Persistent context for decision-making."""
    state: AgentState = AgentState.SOLO
    state_entered: float = 0.0

    # Hysteresis counters
    player_seen_frames: int = 0
    player_gone_seconds: float = 0.0
    last_player_time: float = 0.0
    last_speech_time: float = 0.0

    # Behavior flags
    greeted: bool = False

    # Turn tracking (replaces recent_actions/recent_chatbox spam)
    last_action_time: float = 0.0

    def seconds_in_state(self) -> float:
        return time.time() - self.state_entered


class AgentStateMachine:
    """3-state FSM with hysteresis."""

    def __init__(self):
        self.ctx = StateContext()
        self._transition_to(AgentState.SOLO)

    def _transition_to(self, new_state: AgentState) -> None:
        if self.ctx.state != new_state:
            old = self.ctx.state.name
            self.ctx.state = new_state
            self.ctx.state_entered = time.time()
            logger.info(f"State: {old} -> {new_state.name}")
            if new_state == AgentState.SOLO:
                self.ctx.greeted = False

    def force_follow(self) -> None:
        """Called when user says 'follow me'."""
        self._transition_to(AgentState.FOLLOWING)

    def force_stop_follow(self) -> None:
        """Called when user says 'stop following'."""
        if self.ctx.state == AgentState.FOLLOWING:
            self._transition_to(AgentState.ENGAGED)

    def update(
        self,
        players_visible: int,
        speech_heard: bool,
        tick_seconds: float = 1.0,
    ) -> AgentState:
        """Update state from perception tick.

        Args:
            players_visible: Current player count from YOLO
            speech_heard: Whether a transcription happened this tick
            tick_seconds: Seconds since last update (for time-based counters)

        Returns:
            Current AgentState
        """
        now = time.time()

        if speech_heard:
            self.ctx.last_speech_time = now

        # Update hysteresis counters
        if players_visible > 0:
            self.ctx.player_seen_frames += 1
            self.ctx.player_gone_seconds = 0.0
            self.ctx.last_player_time = now
        else:
            self.ctx.player_seen_frames = 0
            self.ctx.player_gone_seconds += tick_seconds

        # --- Transitions ---
        current = self.ctx.state

        if current == AgentState.FOLLOWING:
            # FOLLOWING only exits via explicit command OR long absence
            if self.ctx.player_gone_seconds > FOLLOW_TIMEOUT:
                self._transition_to(AgentState.SOLO)

        elif current == AgentState.SOLO:
            # Enter ENGAGED if we saw someone for ≥3 frames OR heard speech
            if self.ctx.player_seen_frames >= ENTER_ENGAGED_FRAMES or speech_heard:
                self._transition_to(AgentState.ENGAGED)

        elif current == AgentState.ENGAGED:
            # Leave ENGAGED only when BOTH conditions hold
            silence = now - self.ctx.last_speech_time
            if (silence > EXIT_ENGAGED_SILENCE
                    and self.ctx.player_gone_seconds > EXIT_ENGAGED_NO_PLAYER):
                self._transition_to(AgentState.SOLO)

        return self.ctx.state
