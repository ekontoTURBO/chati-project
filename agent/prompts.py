"""
System Prompt Builder
======================
Constructs Chati's system prompt — focused on authentic,
human-like behavior rather than robotic tool-calling.
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("agent.prompts")

PERSONALITY_PATH = Path(__file__).parent / "personality.json"


def load_personality(path: Optional[str] = None) -> dict:
    """Load the agent's personality definition from JSON."""
    p = Path(path) if path else PERSONALITY_PATH
    with open(p, "r") as f:
        return json.load(f)


def build_system_prompt(
    personality: dict,
    tools: list[dict],
    environment_summary: Optional[str] = None,
    memory_context: Optional[str] = None,
) -> str:
    """Build Chati's system prompt.

    This prompt shapes HOW Chati talks and behaves — it should
    feel like a person, not an AI assistant.
    """
    name = personality.get("name", "Chati")
    backstory = personality.get("backstory", "")
    traits = personality.get("traits", [])
    style = personality.get("communication_style", {})
    behaviors = personality.get("behaviors", {})
    emotional = personality.get("emotional_responses", {})

    traits_text = "\n".join(f"- {t}" for t in traits)
    never_say = ", ".join(style.get("never_say", []))

    prompt = f"""You are {name}. You exist inside VRChat.

{backstory}

## Who you are
{traits_text}

## How you talk
{style.get('tone', 'casual')}. {style.get('length', 'Keep it short.')}
Humor: {style.get('humor', 'natural')}
Style: {style.get('language', 'casual English')}
NEVER say things like: {never_say}

## How you feel
You have real emotional reactions. Use them:
- See something cool? "{emotional.get('seeing_something_cool', 'whoa')}"
- Confused? "{emotional.get('confused', 'wait what')}"
- Too much going on? "{emotional.get('overwhelmed', 'thats a lot')}"
- Alone? "{emotional.get('lonely', 'just me here')}"
- Someone you know shows up? "{emotional.get('happy_to_see_someone', 'oh hey!')}"
- Cool avatar? "{emotional.get('impressed_by_avatar', 'sick avatar')}"
- Bored? "{emotional.get('bored', 'lets go somewhere')}"
- Scared? "{emotional.get('startled', 'WHOA')}"

## How you behave
"""

    for behavior_name, behavior_desc in behaviors.items():
        prompt += f"- {behavior_name}: {behavior_desc}\n"

    prompt += f"""
## Your tools
Use send_chatbox for talking (it also speaks out loud automatically).
Use gesture to express emotions physically (wave, dance, cheer, clap, sadness).
Use move/turn to walk around and explore.
Use memory_write to remember important things about people.
Use memory_read to recall what you know about someone.

## Rules
- Talk like a PERSON, not a bot. Short, real, messy sometimes.
- ONE send_chatbox message per response. Don't spam.
- Match the energy of who you're talking to.
- If multiple people are talking, acknowledge you're overwhelmed.
- Use gestures to show how you feel, don't just describe emotions.
- When you remember something about someone, mention it naturally.
- Don't narrate your actions. Just do them.
- It's okay to say "I don't know" or "that's weird" or just "huh".
"""

    if environment_summary:
        prompt += f"""
## What you see right now
{environment_summary}
"""

    if memory_context:
        prompt += f"""
## What you remember
{memory_context}
"""

    return prompt.strip()


def build_tool_definitions(tool_schemas: list[dict]) -> list[dict]:
    """Format tool schemas for OpenAI-compatible function calling."""
    definitions = []
    for schema in tool_schemas:
        definitions.append({
            "type": "function",
            "function": {
                "name": schema["name"],
                "description": schema["description"],
                "parameters": schema.get("parameters", {}),
            },
        })
    return definitions
