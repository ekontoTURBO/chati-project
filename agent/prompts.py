"""
System Prompt Builder — minimal, prompt-cache friendly.

Target: ~800 chars static + whatever perception/memory/dialogue add.
Based on AUDIT.md section 7.
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("agent.prompts")

PERSONALITY_PATH = Path(__file__).parent / "personality.json"


def load_personality(path: Optional[str] = None) -> dict:
    """Load personality JSON."""
    p = Path(path) if path else PERSONALITY_PATH
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _fmt_never_say(never_say) -> str:
    """Safely format never_say — supports both list and comma-separated string."""
    if isinstance(never_say, list):
        return ", ".join(never_say)
    if isinstance(never_say, str):
        return never_say
    return ""


def build_system_prompt(
    personality: dict,
    tools: Optional[list[dict]] = None,
    environment_summary: Optional[str] = None,
    memory_context: Optional[str] = None,
    dialogue_text: Optional[str] = None,
) -> str:
    """Build Chati's minimal system prompt.

    Static part kept short so Ollama can cache the prefix.
    Dynamic blocks (perception/memory/dialogue) go at the end.
    """
    name = personality.get("name", "Chati")
    backstory = personality.get("backstory", "")
    traits = personality.get("traits", [])
    style = personality.get("communication_style", {})
    never_say = _fmt_never_say(personality.get("never_say", []))

    traits_line = ", ".join(traits)
    tone = style.get("tone", "casual")
    length = style.get("length", "short")

    prompt = f"""You are {name}. You live in VRChat. {backstory}

You're {traits_line}. Tone: {tone}. Length: {length}.

Do NOT sound like an assistant. Never say: {never_say}.

Use send_chatbox to talk (also speaks aloud). Use gesture for emotes (wave, dance, clap, thumbsup, cheer, sadness). Use move/turn/look_at to navigate. Use memory_write/read for facts about people.

One action per turn. Just do — don't narrate."""

    blocks = []
    if environment_summary:
        blocks.append(f"\n\nScene:\n{environment_summary}")
    if memory_context:
        blocks.append(f"\n\nMemory:\n{memory_context}")
    if dialogue_text:
        blocks.append(f"\n\n{dialogue_text}")

    return prompt + "".join(blocks)


def build_tool_definitions(tool_schemas: list[dict]) -> list[dict]:
    """Wrap tool schemas in OpenAI function-call format."""
    return [
        {
            "type": "function",
            "function": {
                "name": s["name"],
                "description": s["description"],
                "parameters": s.get("parameters", {}),
            },
        }
        for s in tool_schemas
    ]
