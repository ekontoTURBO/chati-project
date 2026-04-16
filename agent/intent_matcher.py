"""
Tier 1: Instant Intent Matcher
================================
Pattern-based command detection that bypasses the LLM entirely.
Returns tool call dicts for matched commands, None otherwise.

This is the fastest response path — executes in <1ms.
Use for unambiguous commands only. Ambiguous speech falls
through to Tier 2 (Qwen 3.5) or Tier 3 (Gemma 4 vision).

Match philosophy: precision > recall. If unsure, return None
and let a smarter tier handle it.
"""

import re
import logging
from typing import Optional

logger = logging.getLogger("agent.intent")


# --- Command patterns ---
# Each pattern -> list of tool calls to execute in order.
# Use word boundaries (\b) to avoid partial matches.

FOLLOW_PATTERNS = [
    r"\b(follow me|come here|come over here|come with me|over here|this way|follow me please|walk with me)\b",
]

STOP_PATTERNS = [
    r"\b(stop following|stop moving|stop walking|stop there)\b",
    r"\b(stay here|wait here|hold on|freeze|stay there|don't move|dont move)\b",
    # "stop" alone only when short utterance (handled in match function)
]

TURN_PATTERNS = {
    r"\bturn around\b|\bturn back\b|\bface me\b": ("right", "half"),
    r"\bturn left\b|\blook left\b|\blook to your left\b": ("left", "quarter"),
    r"\bturn right\b|\blook right\b|\blook to your right\b": ("right", "quarter"),
    r"\bturn a bit left\b|\bslightly left\b": ("left", "slight"),
    r"\bturn a bit right\b|\bslightly right\b": ("right", "slight"),
}

MOVE_PATTERNS = {
    r"\b(go forward|walk forward|move forward|go ahead)\b": ("forward", 1.0, 1.5),
    r"\b(go back|walk back|move back|step back|back up|go backwards)\b": ("backward", 1.0, 1.5),
    r"\b(go left|walk left|step left)\b": ("left", 1.0, 1.0),
    r"\b(go right|walk right|step right)\b": ("right", 1.0, 1.0),
}

GESTURE_PATTERNS = {
    r"\b(wave at me|wave|say hi)\b": "wave",
    r"\b(dance|start dancing|show me your moves)\b": "dance",
    r"\b(cheer|yay|cheer up)\b": "cheer",
    r"\b(clap|applaud)\b": "clap",
    r"\b(thumbs up|thumbsup|nice)\b": "thumbsup",
    r"\b(point at me|point|point there)\b": "point",
    r"\b(bow|take a bow)\b": "bow",
    r"\b(backflip|do a flip)\b": "backflip",
    r"\b(sad|look sad)\b": "sadness",
}

JUMP_PATTERNS = [r"\b(jump|hop|hop up)\b"]


def _matches_any(text: str, patterns: list[str]) -> bool:
    """Check if text matches any of the regex patterns."""
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def _matches_dict(text: str, patterns: dict) -> Optional[object]:
    """Return value from pattern dict if any pattern matches."""
    for pattern, value in patterns.items():
        if re.search(pattern, text, re.IGNORECASE):
            return value
    return None


def match_intent(speech: str) -> Optional[dict]:
    """Try to match speech to a direct command intent.

    Args:
        speech: Transcribed speech text from the player

    Returns:
        Dict with {intent, tool_calls, reply} if matched, None otherwise.
        tool_calls is a list of dicts like [{"name": "move", "args": {...}}]
        reply is an optional short chatbox message to send.
    """
    if not speech:
        return None

    text = speech.strip()
    lower = text.lower()
    word_count = len(text.split())

    # --- Follow commands ---
    if _matches_any(lower, FOLLOW_PATTERNS):
        logger.info(f"[INTENT] follow: '{speech}'")
        return {
            "intent": "follow",
            "tool_calls": [
                {"name": "send_chatbox", "args": {"text": "okay, coming!"}},
            ],
            "state_change": "FOLLOWING",
        }

    # --- Stop commands ---
    if _matches_any(lower, STOP_PATTERNS) or (lower in ("stop", "stop!", "stop.") and word_count <= 2):
        logger.info(f"[INTENT] stop: '{speech}'")
        return {
            "intent": "stop",
            "tool_calls": [
                {"name": "move", "args": {"direction": "stop"}},
                {"name": "send_chatbox", "args": {"text": "okay, staying put"}},
            ],
            "state_change": "SOCIALIZING",
        }

    # --- Turn commands ---
    turn_result = _matches_dict(lower, TURN_PATTERNS)
    if turn_result:
        direction, amount = turn_result
        logger.info(f"[INTENT] turn {direction} {amount}: '{speech}'")
        return {
            "intent": "turn",
            "tool_calls": [
                {"name": "turn", "args": {"direction": direction, "amount": amount}},
            ],
        }

    # --- Movement commands ---
    move_result = _matches_dict(lower, MOVE_PATTERNS)
    if move_result:
        direction, speed, duration = move_result
        logger.info(f"[INTENT] move {direction}: '{speech}'")
        return {
            "intent": "move",
            "tool_calls": [
                {"name": "move", "args": {
                    "direction": direction,
                    "speed": speed,
                    "duration": duration,
                }},
            ],
        }

    # --- Jump ---
    if _matches_any(lower, JUMP_PATTERNS) and word_count <= 4:
        logger.info(f"[INTENT] jump: '{speech}'")
        return {
            "intent": "jump",
            "tool_calls": [{"name": "jump", "args": {}}],
        }

    # --- Gestures ---
    gesture = _matches_dict(lower, GESTURE_PATTERNS)
    if gesture:
        logger.info(f"[INTENT] gesture {gesture}: '{speech}'")
        return {
            "intent": "gesture",
            "tool_calls": [
                {"name": "gesture", "args": {"type": gesture}},
            ],
        }

    # No match — let a smarter tier handle this
    return None


def needs_vision(speech: str) -> bool:
    """Detect if speech requires visual context to answer.

    Used to route between Tier 2 (text-only Qwen) and
    Tier 3 (vision-capable Gemma 4).
    """
    if not speech:
        return False
    lower = speech.lower()
    vision_keywords = [
        "see", "look", "watch", "show", "describe",
        "what is", "what's", "whats", "avatar", "outfit",
        "wearing", "color", "colour", "looks like",
        "can you see", "do you see", "what do you",
    ]
    return any(kw in lower for kw in vision_keywords)
