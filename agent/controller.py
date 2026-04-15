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

# Disable cuDNN to prevent cudnnGetLibConfig crash
# (PyTorch 2.5.1 cuDNN 9.1 vs CUDA 13.2 driver mismatch)
import torch
torch.backends.cudnn.enabled = False
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
from agent.state_machine import AgentStateMachine, AgentState
from perception.scene_analyzer import SceneAnalyzer

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

        # TTS audio output → VB-Audio Cable (with push-to-talk via OSC)
        self.tts_router = TTSAudioRouter(osc_client=self.osc)

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
        self.chatbox_tool = ChatboxTool(self.osc, speak_tool=self.speak_tool)
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
            "turn": self.look_tool.turn,
        }

        # Conversation history — sliding window of messages
        self._history: list[dict] = []
        # Personality config loaded from personality.json
        self._personality = load_personality()
        # All tool schemas for function calling
        self._tool_schemas = self._collect_tool_schemas()
        # Tool definitions in OpenAI format
        self._tool_definitions = build_tool_definitions(self._tool_schemas)

        # --- New: Scene analyzer + State machine ---
        self.scene_analyzer = SceneAnalyzer(self.video_capture)
        self.state_machine = AgentStateMachine()

        # Flag to control the main loop
        self._running = False
        # Last time we called the LLM for a decision
        self._last_llm_action: float = time.time()
        # How often to call the LLM (seconds) — per state
        self._action_intervals = {
            AgentState.IDLE: 5.0,
            AgentState.EXPLORING: 8.0,
            AgentState.APPROACHING: 3.0,
            AgentState.SOCIALIZING: 12.0,
            AgentState.CONVERSING: 2.0,  # Fast response in conversation
            AgentState.FOLLOWING: 5.0,
        }

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
            self.look_tool.turn_schema,
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

        # Start real-time scene analyzer (YOLO + OCR)
        logger.info("Starting scene analyzer (YOLO + OCR)...")
        self.scene_analyzer.start()

        # Verify Ollama server is reachable
        logger.info(f"Checking Ollama server at {self.model_url}...")
        try:
            models = self.client.models.list()
            logger.info(f"Ollama server OK. Available models: {[m.id for m in models.data]}")
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
        """Async main loop — perception → state → reasoning → action.

        The scene analyzer runs continuously in a background thread.
        This loop reads the perception state, updates the state machine,
        and calls the LLM when appropriate for the current state.
        """
        while self._running:
            try:
                # --- 1. Read perception data ---
                perception = self.scene_analyzer.get_state()
                audio_chunks = self.audio_capture.get_all_chunks()
                speech = None
                if audio_chunks:
                    speech = " ".join(c["text"] for c in audio_chunks).strip()
                    if speech:
                        logger.info(f'* Heard: "{speech}"')

                # --- 2. Update state machine ---
                state = self.state_machine.update(
                    players=perception.players_visible,
                    speech=speech,
                    scene_changed=perception.scene_changed,
                    view_blocked=perception.view_blocked,
                )

                # --- 3. Log perception summary ---
                summary = perception.summary()
                if perception.scene_changed or speech:
                    logger.info(f"[{state.name}] {summary}")

                # --- 4. Act based on state ---
                now = time.time()
                interval = self._action_intervals.get(state, 8.0)

                # Always act immediately for speech
                if speech:
                    await self._handle_conversation(speech, perception)
                elif now - self._last_llm_action >= interval:
                    self._last_llm_action = now
                    await self._handle_state(state, perception)

            except Exception as e:
                logger.error(f"Main loop error: {e}", exc_info=True)

            await asyncio.sleep(1.0)  # 1s tick — perception runs separately

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

        # Build environment summary from real-time perception
        perception = self.scene_analyzer.get_state()
        env_summary = perception.for_prompt() if perception.timestamp > 0 else None

        # Build system prompt
        system_prompt = build_system_prompt(
            personality=self._personality,
            tools=self._tool_schemas,
            environment_summary=env_summary,
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
        if env_summary:
            logger.info(f"  Perception: {env_summary[:200]}")
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
            # Flag movement to suppress motion detection during moves/turns
            is_movement = func_name in ("move", "turn", "look_at")
            if is_movement:
                self.scene_analyzer.agent_is_moving = True

            if func_args:
                result = handler(**func_args)
            else:
                result = handler()

            if is_movement:
                self.scene_analyzer.agent_is_moving = False

            result_str = json.dumps(result, ensure_ascii=False)
            if len(result_str) > 300:
                result_str = result_str[:300] + "..."
            logger.info(f"  << RESULT: {result_str}")

            # Track what we did for anti-repetition
            action_summary = f"{func_name}({json.dumps(func_args, ensure_ascii=False)})"
            self.state_machine.ctx.add_action(action_summary)
            # Track chatbox messages separately for conversation anti-repetition
            if func_name == "send_chatbox" and func_args.get("text"):
                self.state_machine.ctx.add_chatbox(func_args["text"])

            # Add tool result to history for follow-up
            self._history.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result),
            })

        except Exception as e:
            logger.error(f"Tool execution failed: {func_name} — {e}")

    # ================================================================
    # State-based action handlers
    # ================================================================

    async def _handle_state(self, state: AgentState, perception) -> None:
        """Dispatch to the appropriate handler based on agent state."""
        ctx = self.state_machine.ctx
        frame_b64 = self.video_capture.get_latest_frame_b64()
        if not frame_b64:
            return

        if state == AgentState.EXPLORING:
            await self._act_explore(frame_b64, perception, ctx)
        elif state == AgentState.APPROACHING:
            await self._act_approach(frame_b64, perception, ctx)
        elif state == AgentState.SOCIALIZING:
            await self._act_socialize(frame_b64, perception, ctx)
        elif state == AgentState.FOLLOWING:
            self._act_follow(perception)
        elif state == AgentState.CONVERSING:
            pass  # Handled by _handle_conversation
        elif state == AgentState.IDLE:
            await self._act_explore(frame_b64, perception, ctx)

    def _act_follow(self, perception) -> None:
        """Follow a player based on their detected position.

        Uses the player's screen position to determine movement:
        - Player left of center → turn left
        - Player right of center → turn right
        - Player small (far away) → move forward
        - Player large (close) → stop, we're near them
        """
        if perception.players_visible == 0:
            logger.info("[FOLLOWING] Lost sight of player, stopping")
            self.move_tool.move(direction="stop")
            return

        px, py = perception.player_positions[0]
        self.scene_analyzer.agent_is_moving = True

        if px < 0.35:
            logger.info(f"[FOLLOWING] Player at left ({px:.2f}), turning left")
            self.look_tool.turn(direction="left", amount="slight")
        elif px > 0.65:
            logger.info(f"[FOLLOWING] Player at right ({px:.2f}), turning right")
            self.look_tool.turn(direction="right", amount="slight")
        elif py < 0.5:
            logger.info(f"[FOLLOWING] Player ahead but far ({py:.2f}), moving forward")
            self.move_tool.move(direction="forward", speed=0.8, duration=1.0)
        else:
            logger.info(f"[FOLLOWING] Player close and centered, keeping pace")
            self.move_tool.move(direction="forward", speed=0.4, duration=0.5)

        self.scene_analyzer.agent_is_moving = False

    async def _handle_conversation(self, speech: str, perception) -> None:
        """Handle a speech interaction — highest priority."""
        frame_b64 = self.video_capture.get_latest_frame_b64()
        ctx = self.state_machine.ctx
        ctx.conversation_turns += 1
        self._last_llm_action = time.time()

        # Check if entering/exiting follow mode
        state = self.state_machine.ctx.state
        if state == AgentState.FOLLOWING:
            logger.info("[FOLLOWING] Acknowledged! Following player.")
            self.chatbox_tool.send_chatbox("Sure, I'll follow you!")
            return
        lower = speech.lower()
        if any(p in lower for p in ["stop following", "stay here", "wait here"]):
            logger.info("[FOLLOWING] Stop command received.")
            self.chatbox_tool.send_chatbox("Okay, I'll stay here!")
            return

        # Stop moving to listen
        self.move_tool.move(direction="stop")

        logger.info(f'[CONVERSING] Responding to: "{speech}"')

        content_parts = []
        if frame_b64:
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"},
            })

        # Include perception data for context
        perception_text = perception.for_prompt() if perception else ""

        content_parts.append({
            "type": "text",
            "text": (
                f'Someone said: "{speech}"\n\n'
                f"{perception_text}\n\n"
                "React like a real person. One send_chatbox message. "
                "Gesture if it fits the moment."
                + ctx.recent_actions_text()
                + ctx.recent_chatbox_text()
            ),
        })

        await self._send_to_model({"role": "user", "content": content_parts})

    async def _act_approach(self, frame_b64: str, perception, ctx) -> None:
        """Approaching a player — move toward them and greet."""
        if not ctx.greeted:
            ctx.greeted = True
            self.move_tool.move(direction="stop")

            logger.info(f"[APPROACHING] Greeting {perception.players_visible} player(s)")

            # Include chatbox text if OCR detected any
            chatbox_info = ""
            if perception.visible_text:
                chatbox_info = (
                    f'\nA player said (chatbox): "{" ".join(perception.visible_text)}"\n'
                    "Respond to what they said!\n"
                )

            await self._send_to_model({
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}},
                    {"type": "text", "text": (
                        f"You just noticed someone nearby.\n"
                        f"{perception.for_prompt()}\n"
                        f"{chatbox_info}"
                        "Say hi — react to what you actually see. "
                        "Comment on their avatar, what they're doing, the vibe. "
                        "Wave with a gesture. One send_chatbox message."
                        + ctx.recent_chatbox_text()
                    )},
                ],
            })
        else:
            # Already greeted, transition will move us to SOCIALIZING
            pass

    async def _act_socialize(self, frame_b64: str, perception, ctx) -> None:
        """Hanging out near a player — observe, react, chat."""
        logger.info(f"[SOCIALIZING] Observing {perception.players_visible} player(s)")

        chatbox_info = ""
        if perception.visible_text:
            chatbox_info = (
                f'\nPlayer chatbox text: "{" ".join(perception.visible_text)}"\n'
                "RESPOND to their message with send_chatbox!\n"
            )

        await self._send_to_model({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}},
                {"type": "text", "text": (
                    "You're hanging out near someone. Look at the scene.\n"
                    f"{chatbox_info}"
                    f"{perception.for_prompt()}\n\n"
                    "What would you naturally do? Maybe comment on "
                    "something, react to what they're doing, do a gesture, "
                    "or just vibe. Don't force it. One action max."
                    + ctx.recent_actions_text()
                    + ctx.recent_chatbox_text()
                )},
            ],
        })

    async def _act_explore(self, frame_b64: str, perception, ctx) -> None:
        """Exploring alone — navigate based on perception data."""
        logger.info(f"[EXPLORING] {perception.summary()}")

        # Use perception data to give specific navigation hints
        nav_hint = ""
        if perception.view_blocked:
            nav_hint = "Your view is BLOCKED by a wall. Move backward or turn left/right.\n"
        elif ctx.stuck_counter > 3:
            nav_hint = "You seem STUCK (scene hasn't changed). Try a different direction.\n"

        chatbox_info = ""
        if perception.visible_text:
            chatbox_info = (
                f'\nText on screen: "{" ".join(perception.visible_text)}"\n'
                "Someone is talking! Reply with send_chatbox.\n"
            )

        await self._send_to_model({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}},
                {"type": "text", "text": (
                    "You're on your own, just wandering around.\n"
                    f"{nav_hint}{chatbox_info}"
                    f"{perception.for_prompt()}\n\n"
                    "What do you feel like doing? Walk somewhere, "
                    "turn around to see what's behind you, comment "
                    "on something interesting, do a little dance, "
                    "whatever feels right. Use turn() to look in new directions."
                    + ctx.recent_actions_text()
                    + ctx.movement_history_text()
                )},
            ],
        })

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

        self.scene_analyzer.stop()
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
