"""
Agent Controller — Main Loop
==============================
The central brain of the VRChat AI agent. Orchestrates:
1. Perception (audio + video capture)
2. Reasoning (Gemma 4 via vLLM)
3. Action (tool execution via OSC)
4. Memory (long-term recall)

Run this file to start the agent:
    python -m agent.controller
"""

import asyncio
import json
import time
import logging
import random
import sys
import signal
from pathlib import Path
from typing import Optional

from openai import OpenAI

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from vrchat_bridge.osc_client import VRChatOSCClient
from vrchat_bridge.tts_output import TTSAudioRouter
from perception.audio_capture import AudioCaptureProcessor
from perception.video_capture import VideoCaptureProcessor
from mcp_tools.speak import SpeakTool
from mcp_tools.gesture import GestureTool
from mcp_tools.move import MoveTool
from mcp_tools.look import LookTool
from mcp_tools.memory import MemoryTool
from mcp_tools.environment import EnvironmentTool
from mcp_tools.chatbox import ChatboxTool
from mcp_tools.world import WorldTool
from agent.prompts import load_personality, build_system_prompt, build_tool_definitions

logger = logging.getLogger("agent.controller")

# Maximum conversation history entries to keep
# Constraint: Must balance context richness vs. VRAM usage
MAX_HISTORY = 20

# Interval between agent "heartbeat" cycles (seconds)
# Constraint: Too fast = high GPU load; too slow = unresponsive
HEARTBEAT_INTERVAL = 3.0

# Interval for environment awareness checks (seconds)
ENVIRONMENT_CHECK_INTERVAL = 20.0


class AgentController:
    """Main agent controller that ties all components together.

    The controller runs an async loop that:
    1. Collects audio chunks + latest video frame
    2. Builds a multimodal prompt with system persona + context
    3. Sends to vLLM for inference
    4. Parses response for text and tool calls
    5. Executes tool calls and handles follow-ups
    6. Maintains conversation history and long-term memory

    Attributes:
        model_url: URL of the vLLM OpenAI-compatible API
        model_name: Model ID used for API calls
    """

    def __init__(
        self,
        model_url: str = "http://localhost:11434/v1",
        model_name: str = "gemma4:e2b",
        osc_send_port: int = 9000,
        osc_recv_port: int = 9001,
        camera_index: int = 0,
    ):
        # URL of the vLLM server's OpenAI-compatible API
        self.model_url = model_url
        # Model identifier for API calls
        self.model_name = model_name

        # --- Initialize components ---

        # OpenAI client pointed at local vLLM server
        self.client = OpenAI(
            base_url=model_url,
            api_key="ollama",  # Ollama doesn't require a real API key
        )

        # VRChat OSC bridge
        self.osc = VRChatOSCClient(
            send_port=osc_send_port,
            recv_port=osc_recv_port,
        )

        # TTS audio output → VB-Audio Cable
        self.tts_router = TTSAudioRouter()

        # Perception pipelines
        self.audio_capture = AudioCaptureProcessor()
        self.video_capture = VideoCaptureProcessor(camera_index=camera_index)

        # MCP Tools — each tool wraps a specific capability
        self.speak_tool = SpeakTool(self.tts_router)
        self.gesture_tool = GestureTool(self.osc)
        self.move_tool = MoveTool(self.osc)
        self.look_tool = LookTool(self.osc)
        self.memory_tool = MemoryTool()
        self.environment_tool = EnvironmentTool(self.video_capture, self.client)
        self.chatbox_tool = ChatboxTool(self.osc)
        self.world_tool = WorldTool()

        # Tool dispatch map — maps tool names to handler functions
        self._tool_handlers = {
            "speak": self.speak_tool.speak,
            "gesture": self.gesture_tool.gesture,
            "move": self.move_tool.move,
            "jump": self.move_tool.jump,
            "look_at": self.look_tool.look_at,
            "memory_write": self.memory_tool.memory_write,
            "memory_read": self.memory_tool.memory_read,
            "environment_query": self.environment_tool.environment_query,
            "send_chatbox": self.chatbox_tool.send_chatbox,
            "join_world": self.world_tool.join_world,
        }

        # Conversation history — sliding window of messages
        self._history: list[dict] = []
        # Personality config loaded from personality.json
        self._personality = load_personality()
        # All tool schemas for function calling
        self._tool_schemas = self._collect_tool_schemas()
        # Tool definitions in OpenAI format
        self._tool_definitions = build_tool_definitions(self._tool_schemas)

        # Last time we did an environment check
        self._last_env_check: float = 0.0
        # Latest environment summary (human-readable)
        self._environment_summary: Optional[str] = None
        # Flag to control the main loop
        self._running = False
        # Idle behavior tracking
        self._idle_ticks: int = 0
        self._last_idle_action: float = 0.0
        # How often idle actions happen (seconds)
        self._idle_action_interval: float = 10.0
        # Social mode tracking
        self._greeted: bool = False
        self._last_social_action: float = 0.0
        # Recent actions log — tells the model what it already did
        self._recent_actions: list[str] = []
        self._max_recent_actions: int = 10

    def _collect_tool_schemas(self) -> list[dict]:
        """Collect all tool schemas from MCP tools.

        Returns:
            List of tool schema dicts for function calling
        """
        schemas = [
            self.speak_tool.tool_schema,
            self.gesture_tool.tool_schema,
            self.move_tool.tool_schema,
            self.look_tool.tool_schema,
            self.memory_tool.tool_schema_write,
            self.memory_tool.tool_schema_read,
            self.environment_tool.tool_schema,
            self.chatbox_tool.tool_schema,
            self.world_tool.tool_schema,
        ]
        # Add jump as a separate tool
        schemas.append({
            "name": "jump",
            "description": "Make the avatar jump.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        })
        return schemas

    def start(self) -> None:
        """Start all agent components and enter the main loop."""
        logger.info("=" * 60)
        logger.info(" VRChat AI Agent — Starting Up")
        logger.info("=" * 60)

        # Connect to VRChat
        logger.info("Connecting to VRChat OSC...")
        self.osc.connect()

        # Start TTS audio routing
        logger.info("Starting TTS audio router...")
        self.tts_router.start()

        # Start perception pipelines
        logger.info("Starting audio capture...")
        self.audio_capture.start()
        logger.info("Starting video capture...")
        self.video_capture.start()

        # Verify vLLM server is reachable
        logger.info(f"Checking vLLM server at {self.model_url}...")
        try:
            models = self.client.models.list()
            logger.info(f"vLLM server OK. Available models: {[m.id for m in models.data]}")
        except Exception as e:
            logger.error(f"Cannot reach vLLM server: {e}")
            logger.error("Make sure the vLLM Docker container is running: docker start chati-vllm")
            self.shutdown()
            return

        # Handle graceful shutdown
        signal.signal(signal.SIGINT, lambda s, f: self.shutdown())

        self._running = True
        logger.info("")
        logger.info(f"  Agent '{self._personality['name']}' is ready!")
        logger.info("  Listening for audio and watching video...")
        logger.info("  Press Ctrl+C to stop.")
        logger.info("")

        # Enter the main loop
        try:
            asyncio.run(self._main_loop())
        except KeyboardInterrupt:
            self.shutdown()

    async def _main_loop(self) -> None:
        """Async main loop — perception → reasoning → action cycle."""
        while self._running:
            try:
                # --- Perception: Collect inputs ---
                audio_chunks = self.audio_capture.get_all_chunks()
                latest_frame = self.video_capture.get_latest_frame_b64()

                # --- Check if we need an environment update ---
                now = time.time()
                if now - self._last_env_check > ENVIRONMENT_CHECK_INTERVAL:
                    await self._update_environment()
                    self._last_env_check = now

                # --- Reasoning: Process audio if detected ---
                if audio_chunks:
                    # Audio detected — someone is likely talking
                    logger.info(f"* Audio detected: {len(audio_chunks)} chunk(s)")
                    await self._process_interaction(audio_chunks, latest_frame)
                else:
                    # No audio — maybe do idle behavior
                    await self._idle_tick()

            except Exception as e:
                logger.error(f"Main loop error: {e}", exc_info=True)

            # Heartbeat interval
            await asyncio.sleep(HEARTBEAT_INTERVAL)

    async def _process_interaction(
        self,
        audio_chunks: list[dict],
        frame_b64: Optional[str],
    ) -> None:
        """Process a speech interaction — Whisper transcriptions + vision.

        Args:
            audio_chunks: List of dicts with 'text' from Whisper STT
            frame_b64: Latest video frame as base64 JPEG
        """
        # Combine all transcriptions into one message
        heard_text = " ".join(c["text"] for c in audio_chunks).strip()
        if not heard_text:
            return

        logger.info(f"Processing speech: \"{heard_text}\"")

        content_parts = []

        # Add video frame if available
        if frame_b64:
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"},
            })

        # Add the transcribed speech as text
        content_parts.append({
            "type": "text",
            "text": (
                f"A player in VRChat said to you: \"{heard_text}\"\n\n"
                "Look at the image to see who's talking and where you are. "
                "Respond naturally — use send_chatbox to reply and gesture "
                "to react. Keep your response short and conversational."
            ),
        })

        user_message = {"role": "user", "content": content_parts}
        await self._send_to_model(user_message)

    async def _send_to_model(self, user_message: dict) -> None:
        """Send a message to the model server and handle the response.

        Manages the tool-call loop: if the model returns tool calls,
        execute them and send results back for follow-up.

        Args:
            user_message: The user message dict to send
        """
        # Build memory context from recent entries
        memory_ctx = self._get_memory_context()

        # Build system prompt
        system_prompt = build_system_prompt(
            personality=self._personality,
            tools=self._tool_schemas,
            environment_summary=self._environment_summary,
            memory_context=memory_ctx,
        )

        # Construct messages array
        messages = [
            {"role": "system", "content": system_prompt},
            *self._history[-MAX_HISTORY:],
            user_message,
        ]

        # --- Verbose: show what we're sending ---
        logger.info("=" * 60)
        logger.info(" MODEL REQUEST")
        logger.info("=" * 60)
        logger.info(f"  System prompt: {len(system_prompt)} chars")
        logger.info(f"  History: {len(self._history)} messages")
        logger.info(f"  Tools: {[t['function']['name'] for t in self._tool_definitions]}")
        if memory_ctx:
            logger.info(f"  Memory: {memory_ctx[:200]}")
        if self._environment_summary:
            logger.info(f"  Environment: {self._environment_summary[:200]}")
        # Show user message content types
        if isinstance(user_message.get("content"), list):
            types = [p.get("type", "?") for p in user_message["content"]]
            logger.info(f"  Input: {types}")
        logger.info("-" * 60)

        # Call the model with tool support
        t_start = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=self._tool_definitions,
                tool_choice="auto",
                temperature=1.0,
                top_p=0.95,
                max_tokens=1024,
            )
        except Exception as e:
            logger.error(f"Model API call failed: {e}")
            return
        t_elapsed = time.time() - t_start

        choice = response.choices[0]
        assistant_message = choice.message

        # --- Verbose: show what came back ---
        logger.info("=" * 60)
        logger.info(f" MODEL RESPONSE ({t_elapsed:.1f}s)")
        logger.info("=" * 60)
        if assistant_message.content:
            logger.info(f"  Text: {assistant_message.content.strip()}")
        if assistant_message.tool_calls:
            for tc in assistant_message.tool_calls:
                logger.info(f"  Tool call: {tc.function.name}({tc.function.arguments})")
        if not assistant_message.content and not assistant_message.tool_calls:
            logger.info("  (empty response)")
        usage = getattr(response, "usage", None)
        if usage:
            logger.info(
                f"  Tokens: {usage.prompt_tokens} in / "
                f"{usage.completion_tokens} out / {usage.total_tokens} total"
            )
        logger.info("-" * 60)

        # Add to history
        self._history.append(user_message)
        self._history.append(assistant_message.model_dump())

        # --- Handle text response ---
        if assistant_message.content:
            text = assistant_message.content.strip()
            if text:
                logger.info(f"  >> SPEAK: {text}")
                # Mute audio capture while speaking (prevent feedback loop)
                self.audio_capture.muted = True
                self.speak_tool.speak(text)
                self.chatbox_tool.send_chatbox(text)
                self.audio_capture.muted = False

        # --- Handle tool calls ---
        if assistant_message.tool_calls:
            for tool_call in assistant_message.tool_calls:
                await self._execute_tool_call(tool_call)

        # Trim history to sliding window
        if len(self._history) > MAX_HISTORY * 2:
            self._history = self._history[-MAX_HISTORY:]

    async def _execute_tool_call(self, tool_call) -> None:
        """Execute a single tool call from the model's response.

        Args:
            tool_call: OpenAI-format tool call object
        """
        func_name = tool_call.function.name
        try:
            func_args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            logger.error(f"Invalid tool call args: {tool_call.function.arguments}")
            return

        logger.info(f"  >> TOOL: {func_name}({json.dumps(func_args)})")

        handler = self._tool_handlers.get(func_name)
        if handler is None:
            logger.warning(f"  !! Unknown tool: {func_name}")
            return

        try:
            if func_args:
                result = handler(**func_args)
            else:
                result = handler()

            result_str = json.dumps(result, ensure_ascii=False)
            if len(result_str) > 300:
                result_str = result_str[:300] + "..."
            logger.info(f"  << RESULT: {result_str}")

            # Track what we did for anti-repetition
            action_summary = f"{func_name}({json.dumps(func_args, ensure_ascii=False)})"
            self._recent_actions.append(action_summary)
            if len(self._recent_actions) > self._max_recent_actions:
                self._recent_actions = self._recent_actions[-self._max_recent_actions:]

            # Add tool result to history for follow-up
            self._history.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result),
            })

        except Exception as e:
            logger.error(f"Tool execution failed: {func_name} — {e}")

    def _players_visible(self) -> int:
        """Check how many players are visible from the latest environment data."""
        if not self._environment_summary:
            return 0
        try:
            data = json.loads(self._environment_summary)
            # Handle nested JSON from Gemma's markdown responses
            scene = data.get("scene", "")
            if isinstance(scene, str) and scene.startswith("```"):
                scene = scene.strip("`").strip()
                if scene.startswith("json"):
                    scene = scene[4:].strip()
                data = json.loads(scene)
            return int(data.get("players_visible", 0))
        except (json.JSONDecodeError, ValueError, TypeError):
            return 0

    async def _update_environment(self) -> None:
        """Run a periodic environment awareness check."""
        logger.info("~ Environment check...")
        result = self.environment_tool.environment_query()
        if result.get("success"):
            self._environment_summary = json.dumps(result, indent=2)
            players = self._players_visible()
            logger.info(f"~ Environment ({players} players): {self._environment_summary[:250]}")
        else:
            logger.warning(f"~ Environment check failed: {result.get('error', '?')}")

    async def _idle_tick(self) -> None:
        """Handle idle behavior based on whether players are nearby.

        Two modes:
        - SOCIAL: Players visible → stop moving, face them, wave, talk
        - EXPLORE: No players → navigate the world, look around, comment
        """
        self._idle_ticks += 1
        now = time.time()

        if now - self._last_idle_action < self._idle_action_interval:
            return
        self._last_idle_action = now

        frame_b64 = self.video_capture.get_latest_frame_b64()
        if not frame_b64:
            return

        players = self._players_visible()

        if players > 0:
            await self._social_tick(frame_b64, players)
        else:
            await self._explore_tick(frame_b64)

    async def _social_tick(self, frame_b64: str, player_count: int) -> None:
        """Player detected — stop everything and interact.

        First encounter: wave and greet.
        After greeting: observe, react to changes, wait for response.
        """
        # Stop any ongoing movement
        self.move_tool.move(direction="stop")

        # Build recent actions context
        actions_ctx = ""
        if self._recent_actions:
            actions_ctx = (
                "\n\nYour recent actions (DO NOT repeat these):\n"
                + "\n".join(f"- {a}" for a in self._recent_actions[-5:])
            )

        if not self._greeted:
            # First time seeing a player — greet them
            self._greeted = True
            self._last_social_action = time.time()
            logger.info(f"~ SOCIAL MODE: greeting {player_count} player(s)!")

            greet_message = {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"},
                    },
                    {
                        "type": "text",
                        "text": (
                            f"You just noticed {player_count} player(s) in VRChat! "
                            "Greet them:\n"
                            "1. Use gesture to wave\n"
                            "2. Use send_chatbox with a SHORT friendly greeting "
                            "(react to their avatar or the scene — be specific)\n\n"
                            "IMPORTANT: Look for any TEXT floating above players' "
                            "heads — that's their chatbox message to you. If you "
                            "see text, READ it and RESPOND to what they said.\n\n"
                            "Keep it to ONE chatbox message. Be natural and brief."
                            + actions_ctx
                        ),
                    },
                ],
            }
            await self._send_to_model(greet_message)
        else:
            # Already greeted — slow down, observe, don't repeat yourself
            now = time.time()
            if now - self._last_social_action < 20.0:
                return
            self._last_social_action = now

            logger.info(f"~ SOCIAL MODE: observing {player_count} player(s)...")

            observe_message = {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"},
                    },
                    {
                        "type": "text",
                        "text": (
                            "You're hanging out near a player in VRChat. "
                            "Look at the image carefully and pick ONE thing to do:\n"
                            "- If you see TEXT above a player's head, READ it and "
                            "REPLY with send_chatbox (this is their message to you!)\n"
                            "- Comment on something NEW you see (send_chatbox)\n"
                            "- React to what the player is doing\n"
                            "- Do a gesture (dance, cheer, thumbsup, clap)\n"
                            "- Look around (look_at)\n\n"
                            "PRIORITY: always read and respond to chatbox text first.\n"
                            "Say something DIFFERENT from before. One action only."
                            + actions_ctx
                        ),
                    },
                ],
            }
            await self._send_to_model(observe_message)

    async def _explore_tick(self, frame_b64: str) -> None:
        """No players around — explore the world."""
        if self._greeted:
            self._greeted = False
            logger.info("~ Players gone, resetting social state")

        actions_ctx = ""
        if self._recent_actions:
            actions_ctx = (
                "\n\nYour recent actions (DO NOT repeat these):\n"
                + "\n".join(f"- {a}" for a in self._recent_actions[-5:])
            )

        logger.info("~ EXPLORE MODE: no players, exploring...")

        explore_message = {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"},
                },
                {
                    "type": "text",
                    "text": (
                        "You're exploring VRChat alone. Look at the scene "
                        "and pick ONE action. Vary your behavior:\n\n"
                        "- WALL/OBSTACLE: move backward or left/right to turn\n"
                        "- OPEN SPACE: move forward_left or forward_right\n"
                        "- INTERESTING OBJECT: send_chatbox to comment\n"
                        "- DOORWAY/HALLWAY: move toward it\n"
                        "- NOTHING NEW: look_at left/right to scan, or gesture\n"
                        "- TEXT ABOVE SOMEONE'S HEAD: a player is talking! "
                        "Read their message and reply with send_chatbox\n\n"
                        "ONE or TWO tools max."
                        + actions_ctx
                    ),
                },
            ],
        }
        await self._send_to_model(explore_message)

    def _get_memory_context(self) -> Optional[str]:
        """Retrieve relevant memory entries for context.

        Returns:
            Formatted memory string, or None if no memories
        """
        result = self.memory_tool.memory_list(limit=5)
        if not result.get("success") or result.get("count", 0) == 0:
            return None

        lines = ["Recent memories:"]
        for entry in result["entries"]:
            lines.append(f"  - {entry['key']}: {entry['value']}")
        return "\n".join(lines)

    def shutdown(self) -> None:
        """Gracefully shut down all components."""
        logger.info("Shutting down agent...")
        self._running = False

        self.audio_capture.stop()
        self.video_capture.stop()
        self.tts_router.stop()
        self.osc.disconnect()

        logger.info("Agent stopped. Goodbye!")


def main():
    """Entry point for the agent controller."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Parse optional config overrides
    import argparse
    parser = argparse.ArgumentParser(description="VRChat AI Agent Controller")
    parser.add_argument(
        "--model-url",
        default="http://localhost:11434/v1",
        help="Ollama server URL (default: http://localhost:11434/v1)",
    )
    parser.add_argument(
        "--model-name",
        default="gemma4:e2b",
        help="Model name for API calls",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index for OBS Virtual Camera (default: 0)",
    )
    parser.add_argument(
        "--osc-port",
        type=int,
        default=9000,
        help="VRChat OSC send port (default: 9000)",
    )
    args = parser.parse_args()

    controller = AgentController(
        model_url=args.model_url,
        model_name=args.model_name,
        camera_index=args.camera,
        osc_send_port=args.osc_port,
    )
    controller.start()


if __name__ == "__main__":
    main()
