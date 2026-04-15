"""
Agent State Machine
====================
Manages Chati's behavioral states and transitions.

States:
- IDLE: Just started, no data yet
- EXPLORING: No players, roaming the world
- APPROACHING: Player detected, moving toward them
- SOCIALIZING: Near a player, greeting/chatting
- CONVERSING: Active conversation (player spoke)
- FOLLOWING: Following a player who asked

Transitions are driven by PerceptionState data,
not by the LLM — this makes behavior predictable and fast.
"""

import time
import logging
from enum import Enum, auto
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger("agent.state")


class AgentState(Enum):
    IDLE = auto()
    EXPLORING = auto()
    APPROACHING = auto()
    SOCIALIZING = auto()
    CONVERSING = auto()
    FOLLOWING = auto()


@dataclass
class StateContext:
    """Persistent context across state transitions."""
    # Current state
    state: AgentState = AgentState.IDLE
    # When we entered this state
    state_entered: float = 0.0
    # How long we've been in this state
    state_duration: float = 0.0

    # Social tracking
    greeted: bool = False
    last_greeting_time: float = 0.0
    conversation_turns: int = 0
    last_speech_heard: str = ""
    last_speech_time: float = 0.0

    # Exploration tracking
    stuck_counter: int = 0  # increments when scene doesn't change
    last_move_direction: str = ""
    explored_areas: int = 0
    move_history: list = field(default_factory=list)  # last N directions moved

    # Recent actions (for anti-repetition)
    recent_actions: list = field(default_factory=list)
    recent_chatbox: list = field(default_factory=list)

    def add_action(self, action: str) -> None:
        """Track a recent action."""
        self.recent_actions.append(action)
        if len(self.recent_actions) > 10:
            self.recent_actions = self.recent_actions[-10:]
        # Track movement directions separately
        if action.startswith("move(") or action.startswith("turn("):
            self.move_history.append(action)
            if len(self.move_history) > 8:
                self.move_history = self.move_history[-8:]

    def add_chatbox(self, text: str) -> None:
        """Track a recent chatbox message."""
        self.recent_chatbox.append(text)
        if len(self.recent_chatbox) > 5:
            self.recent_chatbox = self.recent_chatbox[-5:]

    def recent_actions_text(self) -> str:
        """Format recent actions for the LLM prompt."""
        if not self.recent_actions:
            return ""
        return (
            "\n\nYour recent actions (DO NOT repeat these):\n"
            + "\n".join(f"- {a}" for a in self.recent_actions[-5:])
        )

    def movement_history_text(self) -> str:
        """Format movement history for navigation awareness."""
        if not self.move_history:
            return ""
        return (
            "\n\nYour recent movement (vary direction, don't repeat):\n"
            + "\n".join(f"- {m}" for m in self.move_history[-5:])
        )

    def recent_chatbox_text(self) -> str:
        """Format recent chatbox messages for anti-repetition."""
        if not self.recent_chatbox:
            return ""
        return (
            "\n\nYou already said these (say something DIFFERENT):\n"
            + "\n".join(f'- "{m}"' for m in self.recent_chatbox[-3:])
        )


class AgentStateMachine:
    """Manages state transitions based on perception data."""

    def __init__(self):
        self.ctx = StateContext()
        self._transition_to(AgentState.IDLE)

    def _transition_to(self, new_state: AgentState) -> None:
        """Transition to a new state."""
        if self.ctx.state != new_state:
            old = self.ctx.state.name
            self.ctx.state = new_state
            self.ctx.state_entered = time.time()
            logger.info(f"State: {old} -> {new_state.name}")

            # Reset state-specific context on transition
            if new_state == AgentState.EXPLORING:
                self.ctx.greeted = False
                self.ctx.conversation_turns = 0
            elif new_state == AgentState.SOCIALIZING:
                self.ctx.stuck_counter = 0

    def update(self, players: int, speech: Optional[str], scene_changed: bool, view_blocked: bool) -> AgentState:
        """Update state based on perception inputs.

        Args:
            players: Number of players visible
            speech: Transcribed speech (None if no speech)
            scene_changed: Whether the scene changed
            view_blocked: Whether the view is obstructed

        Returns:
            Current AgentState after transition
        """
        now = time.time()
        self.ctx.state_duration = now - self.ctx.state_entered

        # Track speech
        if speech:
            self.ctx.last_speech_heard = speech
            self.ctx.last_speech_time = now

        # Track scene changes for stuck detection
        if not scene_changed:
            self.ctx.stuck_counter += 1
        else:
            self.ctx.stuck_counter = 0

        # --- State transitions ---
        current = self.ctx.state

        if speech:
            # Check for follow/come commands
            lower = speech.lower()
            follow_phrases = ["follow me", "come here", "come with me", "over here", "this way"]
            stop_phrases = ["stop following", "stay here", "stop", "wait here", "don't follow"]

            if any(p in lower for p in stop_phrases):
                self._transition_to(AgentState.SOCIALIZING)
            elif any(p in lower for p in follow_phrases):
                self._transition_to(AgentState.FOLLOWING)
            else:
                self._transition_to(AgentState.CONVERSING)

        elif players > 0:
            if current == AgentState.FOLLOWING:
                pass  # Stay following until told to stop
            elif current == AgentState.CONVERSING:
                if now - self.ctx.last_speech_time > 15.0:
                    self._transition_to(AgentState.SOCIALIZING)
            elif current in (AgentState.IDLE, AgentState.EXPLORING):
                self._transition_to(AgentState.APPROACHING)
            elif current == AgentState.APPROACHING:
                if self.ctx.greeted:
                    self._transition_to(AgentState.SOCIALIZING)

        else:
            # No players visible
            if current == AgentState.FOLLOWING:
                # Keep following even if player briefly out of sight
                if now - self.ctx.last_speech_time > 10.0:
                    self._transition_to(AgentState.EXPLORING)
            elif current in (AgentState.SOCIALIZING, AgentState.CONVERSING,
                          AgentState.APPROACHING):
                self._transition_to(AgentState.EXPLORING)
            elif current == AgentState.IDLE:
                self._transition_to(AgentState.EXPLORING)

        return self.ctx.state
