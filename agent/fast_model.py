"""
Tier 2: Fast Text-Only Model (Qwen 3.5)
=========================================
Handles quick conversational responses and tool calls that
don't require vision. Qwen 3.5 2B is ~3-5x faster than Gemma 4
because it skips vision processing entirely.

Fallback chain: qwen3.5:2b -> qwen2.5:1.5b -> gemma4:e2b
"""

import logging
from typing import Optional

from openai import OpenAI

logger = logging.getLogger("agent.fast_model")


# Preferred fast model tags in Ollama (first available wins)
PREFERRED_MODELS = [
    "qwen3.5:2b",
    "qwen3.5:1.7b",
    "qwen3.5:4b",
    "qwen2.5:1.5b",
    "qwen2.5:3b",
]


class FastModel:
    """Text-only fast responder using Qwen via Ollama."""

    def __init__(self, base_url: str = "http://localhost:11434/v1"):
        self.base_url = base_url
        self.client = OpenAI(base_url=base_url, api_key="ollama")
        self.model_name: Optional[str] = None

    def probe(self) -> Optional[str]:
        """Find the best available fast model. Returns model tag or None."""
        try:
            models = self.client.models.list()
            available = {m.id for m in models.data}
            for preferred in PREFERRED_MODELS:
                if preferred in available:
                    self.model_name = preferred
                    logger.info(f"Fast model selected: {preferred}")
                    return preferred
            logger.warning(
                f"No fast model found. Available: {available}. "
                f"Pull one with: ollama pull {PREFERRED_MODELS[0]}"
            )
            return None
        except Exception as e:
            logger.error(f"Could not probe fast model: {e}")
            return None

    def is_available(self) -> bool:
        return self.model_name is not None

    def respond(
        self,
        system_prompt: str,
        user_message: str,
        tools: Optional[list[dict]] = None,
        max_tokens: int = 256,
    ) -> dict:
        """Get a fast response from Qwen.

        Args:
            system_prompt: The system message (personality, rules)
            user_message: What the player said or context for Chati
            tools: Optional tool definitions for function calling
            max_tokens: Max response length

        Returns:
            Dict with 'content' (text response) and 'tool_calls' (list)
        """
        if not self.model_name:
            return {"content": "", "tool_calls": []}

        try:
            kwargs = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "max_tokens": max_tokens,
                "temperature": 0.7,
            }
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"

            response = self.client.chat.completions.create(**kwargs)
            msg = response.choices[0].message

            return {
                "content": (msg.content or "").strip(),
                "tool_calls": msg.tool_calls or [],
                "usage": response.usage,
            }
        except Exception as e:
            logger.error(f"Fast model error: {e}")
            return {"content": "", "tool_calls": []}
