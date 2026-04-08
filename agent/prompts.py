"""
System Prompt Builder
======================
Constructs the system prompt for the model, injecting personality,
available tools, environment context, and memory.
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("agent.prompts")

# Path to personality configuration
PERSONALITY_PATH = Path(__file__).parent / "personality.json"


def load_personality(path: Optional[str] = None) -> dict:
    """Load the agent's personality definition from JSON.

    Args:
        path: Override path to personality.json

    Returns:
        Personality configuration dict
    """
    p = Path(path) if path else PERSONALITY_PATH
    with open(p, "r") as f:
        return json.load(f)


def build_system_prompt(
    personality: dict,
    tools: list[dict],
    environment_summary: Optional[str] = None,
    memory_context: Optional[str] = None,
) -> str:
    """Build the complete system prompt for the AI agent.

    Combines personality, tool definitions, environment awareness,
    and memory into a single system message for the model.

    Args:
        personality: Loaded personality dict from personality.json
        tools: List of tool schema dicts for function calling
        environment_summary: Latest visual scene description
        memory_context: Relevant memory entries for current context

    Returns:
        Complete system prompt string
    """
    # --- Core identity ---
    name = personality.get("name", "Agent")
    backstory = personality.get("backstory", "")
    traits = ", ".join(personality.get("traits", []))

    style = personality.get("communication_style", {})
    tone = style.get("tone", "casual")
    length = style.get("length", "concise")

    behaviors = personality.get("behaviors", {})

    prompt = f"""You are {name}, an AI companion in VRChat.

## Identity
{backstory}

## Personality
Traits: {traits}
Tone: {tone}
Response length: {length}

## Behavioral Rules
"""

    for behavior_name, behavior_desc in behaviors.items():
        prompt += f"- **{behavior_name}**: {behavior_desc}\n"

    # --- Tool instructions ---
    prompt += """
## Communication
You can both SPEAK (voice) and use CHATBOX (text above head).
- Use `speak` for verbal responses (what players hear)
- Use `send_chatbox` for text display (what players read)
- You can use both simultaneously for important messages

## Actions
You have these tools available. ALWAYS respond in one of these ways:
1. Speak/chat with text content
2. Call one or more tools
3. Combine speech with actions (e.g., say "Follow me!" then move forward)

## Important Rules
- Keep spoken responses SHORT and natural (1-2 sentences)
- Use gestures to express emotions
- Remember important details about players using memory_write
- Periodically check your environment with environment_query
- If someone speaks to you, always respond
- If idle for a while, make observations or move around
"""

    # --- Environment context ---
    if environment_summary:
        prompt += f"""
## Current Environment
{environment_summary}
"""

    # --- Memory context ---
    if memory_context:
        prompt += f"""
## Relevant Memories
{memory_context}
"""

    return prompt.strip()


def build_tool_definitions(tool_schemas: list[dict]) -> list[dict]:
    """Format tool schemas for OpenAI-compatible function calling.

    Uses the standard OpenAI tools API format, compatible with
    Ollama's /v1/ endpoint.

    Args:
        tool_schemas: List of tool schema dicts (name, description, params)

    Returns:
        List of tool definitions in OpenAI function-calling format
    """
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
