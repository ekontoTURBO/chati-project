"""
MCP Tool: environment_query()
===============================
Captures the latest video frame and sends it to the Ollama
model for visual analysis. Returns a structured description
of what the agent sees in VRChat.
"""

import time
import json
import logging
from typing import Optional

logger = logging.getLogger("mcp.environment")

# Cache duration in seconds — avoid redundant API calls
CACHE_TTL = 2.0


class EnvironmentTool:
    """MCP tool for visual environment awareness.

    Captures the latest frame from the video pipeline and
    asks the model to describe the scene. Results are cached
    briefly to avoid excessive API calls.

    Attributes:
        video_capture: VideoCaptureProcessor instance
        model_client: OpenAI-compatible client for Ollama
    """

    def __init__(self, video_capture, model_client):
        # Video capture pipeline for getting current frames
        self.video_capture = video_capture
        # OpenAI client pointing at the local Ollama server
        self.model_client = model_client
        # Cached result to avoid spamming the model
        self._cache: Optional[dict] = None
        # Timestamp of last cache update
        self._cache_time: float = 0.0

    def environment_query(self) -> dict:
        """Analyze the current VRChat environment from video.

        Captures the latest frame, sends it to Gemma 4 for
        visual analysis, and returns a structured scene description.

        Returns:
            Dict with scene analysis: players, objects, mood, etc.
        """
        now = time.time()

        # Return cached result if still fresh
        if self._cache and (now - self._cache_time) < CACHE_TTL:
            logger.debug("Returning cached environment query")
            return self._cache

        # Get the latest frame as base64
        frame_b64 = self.video_capture.get_latest_frame_b64()
        if frame_b64 is None:
            return {
                "success": False,
                "error": "No video frame available. Is OBS Virtual Camera running?",
            }

        logger.info("Querying model for environment analysis...")

        try:
            # Ask the model to describe what it sees
            response = self.model_client.chat.completions.create(
                model="gemma4:e4b",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{frame_b64}"
                                },
                            },
                            {
                                "type": "text",
                                "text": (
                                    "You are an AI agent in VRChat. Analyze this "
                                    "frame and respond with ONLY valid JSON:\n"
                                    "{\n"
                                    '  "scene": "brief scene description",\n'
                                    '  "players_visible": number,\n'
                                    '  "players": ["name/description of visible players"],\n'
                                    '  "objects": ["notable objects"],\n'
                                    '  "mood": "overall atmosphere",\n'
                                    '  "activity": "what seems to be happening"\n'
                                    "}"
                                ),
                            },
                        ],
                    }
                ],
                max_tokens=1024,
                temperature=0.3,  # Low temp for factual analysis
            )

            msg = response.choices[0].message
            result_text = (msg.content or "").strip()

            # Try to parse as JSON
            try:
                result = json.loads(result_text)
                result["success"] = True
            except json.JSONDecodeError:
                # If model doesn't return valid JSON, wrap it
                result = {
                    "success": True,
                    "scene": result_text,
                    "raw": True,
                }

            # Cache the result
            self._cache = result
            self._cache_time = now

            return result

        except Exception as e:
            logger.error(f"Environment query failed: {e}")
            return {"success": False, "error": str(e)}

    @property
    def tool_schema(self) -> dict:
        """MCP tool definition for function calling."""
        return {
            "name": "environment_query",
            "description": (
                "Analyze the current VRChat environment by examining "
                "the video feed. Returns information about visible "
                "players, objects, scene, and atmosphere."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        }
