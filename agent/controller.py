"""
Agent Controller — Main Loop (simplified post-audit)
======================================================
Single-path reasoning flow:
1. Perception (scene analyzer + whisper + video)
2. State machine (3 states: SOLO, ENGAGED, FOLLOWING) with hysteresis
3. LLM call (Gemma 4 via Ollama) with full context
4. Execute tool calls returned by model

See AUDIT.md for the rationale of this design.
"""

import asyncio
import json
import time
import logging
import sys
import signal
from pathlib import Path
from typing import Optional

from openai import OpenAI

# Project root for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from vrchat_bridge.osc_client import VRChatOSCClient
from vrchat_bridge.tts_output import TTSAudioRouter
from perception.audio_capture import AudioCaptureProcessor
from perception.video_capture import VideoCaptureProcessor
from perception.scene_analyzer import SceneAnalyzer
from mcp_tools.speak import SpeakTool
from mcp_tools.gesture import GestureTool
from mcp_tools.move import MoveTool
from mcp_tools.look import LookTool
from mcp_tools.memory import MemoryTool
from mcp_tools.chatbox import ChatboxTool
from mcp_tools.world import WorldTool
from agent.prompts import load_personality, build_system_prompt, build_tool_definitions
from agent.state_machine import AgentStateMachine, AgentState
from agent.dialogue import Dialogue
from agent.signals import SignalsHub, SignalsSnapshot

logger = logging.getLogger("agent.controller")


# Heartbeat interval — main loop tick in seconds
HEARTBEAT_INTERVAL = 1.0

# Minimum seconds between LLM calls when idle (no speech trigger)
# Keeps GPU free when nothing's happening.
IDLE_TICK_INTERVAL = 15.0

# Seconds of silence in ENGAGED before the agent says something on its own
ENGAGED_LULL_INTERVAL = 12.0

# Max dialogue turns kept in the short-term buffer
DIALOGUE_TURNS = 8


class AgentController:
    """Single-path agent — no tiers, one LLM call per trigger."""

    def __init__(
        self,
        model_url: str = "http://localhost:11434/v1",
        model_name: str = "gemma4:e4b",
        osc_send_port: int = 9000,
        osc_recv_port: int = 9001,
        camera_index: int = 0,
    ):
        self.model_url = model_url
        self.model_name = model_name

        # --- Model client (OpenAI-compatible Ollama API) ---
        self.client = OpenAI(base_url=model_url, api_key="ollama")

        # --- VRChat OSC ---
        self.osc = VRChatOSCClient(
            send_port=osc_send_port,
            recv_port=osc_recv_port,
        )

        # --- Output: TTS with auto push-to-talk ---
        self.tts_router = TTSAudioRouter(osc_client=self.osc)

        # --- Perception pipelines ---
        self.audio_capture = AudioCaptureProcessor()
        self.video_capture = VideoCaptureProcessor(camera_index=camera_index)
        self.scene_analyzer = SceneAnalyzer(self.video_capture)

        # --- Tools ---
        self.speak_tool = SpeakTool(self.tts_router)
        self.gesture_tool = GestureTool(self.osc)
        self.move_tool = MoveTool(self.osc)
        self.look_tool = LookTool(self.osc)
        self.memory_tool = MemoryTool()
        self.chatbox_tool = ChatboxTool(self.osc)  # NO speak_tool coupling
        self.world_tool = WorldTool()

        self._tool_handlers = {
            "speak": self.speak_tool.speak,
            "gesture": self.gesture_tool.gesture,
            "move": self.move_tool.move,
            "jump": self.move_tool.jump,
            "look_at": self.look_tool.look_at,
            "turn": self.look_tool.turn,
            "memory_write": self.memory_tool.memory_write,
            "memory_read": self.memory_tool.memory_read,
            "send_chatbox": self.chatbox_tool.send_chatbox,
            "join_world": self.world_tool.join_world,
        }

        # --- State & memory ---
        self.state_machine = AgentStateMachine()
        self.dialogue = Dialogue(max_turns=DIALOGUE_TURNS)
        self._personality = load_personality()
        self._tool_schemas = self._collect_tool_schemas()
        self._tool_definitions = build_tool_definitions(self._tool_schemas)

        # --- Signals hub — central read for perception (Phase 3, AUDIT.md 10) ---
        self.signals = SignalsHub(
            scene_analyzer=self.scene_analyzer,
            audio_capture=self.audio_capture,
            video_capture=self.video_capture,
            tts_router=self.tts_router,
        )

        # Loop bookkeeping
        self._running = False
        self._last_act_time: float = time.time()
        self._last_perception_log: float = 0.0

    def _collect_tool_schemas(self) -> list[dict]:
        return [
            self.speak_tool.tool_schema,
            self.gesture_tool.tool_schema,
            self.move_tool.tool_schema,
            self.look_tool.tool_schema,
            self.look_tool.turn_schema,
            self.memory_tool.tool_schema_write,
            self.memory_tool.tool_schema_read,
            self.chatbox_tool.tool_schema,
            self.world_tool.tool_schema,
            {
                "name": "jump",
                "description": "Make the avatar jump.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        ]

    # ================================================================
    # Startup / shutdown
    # ================================================================

    def start(self) -> None:
        logger.info("=" * 60)
        logger.info(" Chati — Starting Up")
        logger.info("=" * 60)

        logger.info("Connecting to VRChat OSC...")
        self.osc.connect()

        logger.info("Starting TTS audio router...")
        self.tts_router.start()

        logger.info("Starting audio capture...")
        self.audio_capture.start()

        logger.info("Starting video capture...")
        self.video_capture.start()

        logger.info("Starting scene analyzer (YOLO + OCR)...")
        self.scene_analyzer.start()

        logger.info(f"Checking Ollama server at {self.model_url}...")
        try:
            models = self.client.models.list()
            logger.info(f"Ollama OK. Models: {[m.id for m in models.data]}")
        except Exception as e:
            logger.error(f"Cannot reach Ollama: {e}")
            logger.error("Start it with: ollama serve (in WSL)")
            self.shutdown()
            return

        signal.signal(signal.SIGINT, lambda s, f: self.shutdown())

        self._running = True
        logger.info("")
        logger.info(f"  Agent '{self._personality['name']}' is ready!")
        logger.info("  Press Ctrl+C to stop.")
        logger.info("")

        try:
            asyncio.run(self._main_loop())
        except KeyboardInterrupt:
            self.shutdown()

    def shutdown(self) -> None:
        logger.info("Shutting down agent...")
        self._running = False
        try:
            self.scene_analyzer.stop()
            self.audio_capture.stop()
            self.video_capture.stop()
            self.tts_router.stop()
            self.osc.disconnect()
        except Exception as e:
            logger.error(f"Shutdown error: {e}")
        logger.info("Agent stopped. Goodbye!")

    # ================================================================
    # Main loop
    # ================================================================

    async def _main_loop(self) -> None:
        """One tick = read perception -> update state -> maybe act."""
        while self._running:
            try:
                await self._tick()
            except Exception as e:
                logger.error(f"Main loop error: {e}", exc_info=True)
            await asyncio.sleep(HEARTBEAT_INTERVAL)

    async def _tick(self) -> None:
        # 1. Single read of the world through Signals
        snap = self.signals.snapshot()
        speech = snap.latest_speech

        # 2. Barge-in: if a new speech chunk arrived while TTS was active,
        #    cancel the current playback so Chati listens instead of rambling.
        if speech and snap.tts_playing:
            logger.info(f'[BARGE-IN] Player spoke during TTS: "{speech[:50]}"')
            self.tts_router.cancel()

        # 3. Append player speech to dialogue
        if speech:
            self.dialogue.add_player(speech)

        # 4. Update state machine with hysteresis
        state = self.state_machine.update(
            players_visible=snap.perception.players_visible,
            speech_heard=bool(speech),
            tick_seconds=HEARTBEAT_INTERVAL,
        )

        # 5. Explicit follow/stop commands handled cheaply (no LLM)
        if speech:
            lower = speech.lower()
            if any(p in lower for p in ["follow me", "come with me", "come here"]):
                self.state_machine.force_follow()
            elif any(p in lower for p in ["stop following", "stop follow", "stay here"]):
                self.state_machine.force_stop_follow()

        # 6. Periodic perception log
        now = snap.timestamp
        if now - self._last_perception_log > 10.0:
            logger.info(f"[{state.name}] {snap.perception.summary()}")
            self._last_perception_log = now

        # 7. Decide if we should act
        reason = self._should_act(state, speech, now)
        if not reason:
            return

        # 8. Single-path LLM call using the same snapshot
        self._last_act_time = now
        await self._act(reason, state, speech, snap)

    def _should_act(self, state: AgentState, speech: str, now: float) -> Optional[str]:
        """Return a reason string if we should call the LLM, else None."""
        if speech:
            return "they_spoke"

        time_since = now - self._last_act_time

        if state == AgentState.FOLLOWING:
            # Following moves on its own loop — less LLM chatter
            if time_since > 4.0:
                return "following_tick"
            return None

        if state == AgentState.ENGAGED:
            # Lull: long silence while still near someone
            silence = now - self.state_machine.ctx.last_speech_time
            if silence > ENGAGED_LULL_INTERVAL and time_since > ENGAGED_LULL_INTERVAL:
                return "lull"
            # First moment of ENGAGED (greeting)
            if not self.state_machine.ctx.greeted and time_since > 1.0:
                self.state_machine.ctx.greeted = True
                return "just_arrived"
            return None

        # SOLO — occasional idle tick
        if time_since > IDLE_TICK_INTERVAL:
            return "idle"
        return None

    # ================================================================
    # Single action path
    # ================================================================

    async def _act(self, reason: str, state: AgentState, speech: str, snap: SignalsSnapshot) -> None:
        """Single LLM call — one prompt, one response, one set of tool calls.

        Takes a SignalsSnapshot taken at the start of this tick, so the LLM
        reasons over a consistent view — not a racy mix of stale perception
        and fresh audio (Phase 3, AUDIT.md item 10).
        """
        frame_b64 = snap.frame_b64
        perception_text = snap.perception.for_prompt() if snap.perception else ""
        memory_ctx = self._get_memory_context()
        dialogue_text = self.dialogue.render()

        system_prompt = build_system_prompt(
            personality=self._personality,
            tools=self._tool_schemas,
            environment_summary=perception_text,
            memory_context=memory_ctx,
            dialogue_text=dialogue_text,
        )

        # Per-reason trigger text injected as the user message
        trigger = self._reason_to_trigger(reason, state, speech)

        # Build a FLAT message list — no image history across turns
        content_parts = []
        if frame_b64:
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"},
            })
        content_parts.append({"type": "text", "text": trigger})

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content_parts},
        ]

        logger.info(f"[{state.name}] act({reason}): system={len(system_prompt)}ch dialogue={len(self.dialogue)}t")

        # Mute audio capture BEFORE the model call (audit 3.5)
        self.audio_capture.muted = True

        t_start = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=self._tool_definitions,
                tool_choice="auto",
                temperature=1.0,
                top_p=0.95,
                max_tokens=256,
            )
        except Exception as e:
            logger.error(f"Model call failed: {e}")
            self.audio_capture.muted = False
            return

        elapsed = time.time() - t_start
        msg = response.choices[0].message

        # Track what Chati said for dialogue continuity
        spoken_text = self._handle_response(msg)

        # Record Chati's turn in dialogue buffer
        if spoken_text:
            self.dialogue.add_chati(spoken_text)

        usage = getattr(response, "usage", None)
        if usage:
            logger.info(
                f"  <<- {elapsed:.1f}s | {usage.prompt_tokens} in / "
                f"{usage.completion_tokens} out"
            )

        # Unmute AFTER the model call (TTS inside handlers manages its own mute via PTT)
        self.audio_capture.muted = False

    def _reason_to_trigger(self, reason: str, state: AgentState, speech: str) -> str:
        """Convert a 'reason' code into a short user-facing trigger line."""
        if reason == "they_spoke":
            return f'They said: "{speech}"\n\nReply naturally — one chatbox message, gesture if it fits.'
        if reason == "just_arrived":
            return "You just noticed someone. Say hi — react to their avatar or the vibe. One chatbox message."
        if reason == "lull":
            return "It's been quiet for a while. Say something to break the silence, or do a small gesture."
        if reason == "following_tick":
            return "You're following them. Keep close — move forward if they're far, turn if they're to the side."
        if reason == "idle":
            return "Nobody around. Wander or look at something interesting. Maybe turn to explore a new direction."
        return "React to what's happening."

    def _handle_response(self, msg) -> str:
        """Execute tool calls; return any spoken/chatbox text for dialogue log.

        Plain text content is ignored (per the new flow, Chati should use
        send_chatbox tool to talk — raw content responses are treated as
        dialogue candidates for the buffer only).
        """
        spoken_pieces = []

        # Plain content fallback — if model returned text without a tool, still say it
        if msg.content and msg.content.strip():
            text = msg.content.strip()
            logger.info(f"  [TEXT] {text}")
            spoken_pieces.append(text)
            # Actually speak + chatbox it
            self.chatbox_tool.send_chatbox(text[:144])
            self.speak_tool.speak(text)

        # Execute tool calls
        if msg.tool_calls:
            for tc in msg.tool_calls:
                spoken = self._execute_tool_call(tc)
                if spoken:
                    spoken_pieces.append(spoken)

        return " ".join(spoken_pieces).strip()

    def _execute_tool_call(self, tool_call) -> str:
        """Execute one tool call. Returns text (for dialogue log) if it was a chatbox/speak."""
        name = tool_call.function.name
        try:
            args = json.loads(tool_call.function.arguments or "{}")
        except json.JSONDecodeError:
            logger.error(f"Bad tool args: {tool_call.function.arguments}")
            return ""

        handler = self._tool_handlers.get(name)
        if not handler:
            logger.warning(f"Unknown tool: {name}")
            return ""

        logger.info(f"  [TOOL] {name}({json.dumps(args, ensure_ascii=False)})")

        try:
            result = handler(**args) if args else handler()
        except Exception as e:
            logger.error(f"Tool failed {name}: {e}")
            return ""

        # If the tool sent a chatbox message, ALSO speak it (owner = controller).
        # We do this here instead of coupling inside chatbox.py (audit 1.7).
        spoken_text = ""
        if name == "send_chatbox":
            spoken_text = args.get("text", "")
            if spoken_text:
                self.speak_tool.speak(spoken_text)
        elif name == "speak":
            spoken_text = args.get("text", "")

        return spoken_text

    # ================================================================
    # Helpers
    # ================================================================

    def _get_memory_context(self) -> Optional[str]:
        """Top 5 recent memory entries as a compact string."""
        result = self.memory_tool.memory_list(limit=5)
        if not result.get("success") or result.get("count", 0) == 0:
            return None
        lines = [f"- {e['key']}: {e['value']}" for e in result["entries"]]
        return "\n".join(lines)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    import argparse
    parser = argparse.ArgumentParser(description="Chati VRChat AI Agent")
    parser.add_argument(
        "--model-url",
        default="http://localhost:11434/v1",
        help="Ollama server URL",
    )
    parser.add_argument(
        "--model-name",
        default="gemma4:e4b",
        help="Model tag in Ollama",
    )
    parser.add_argument(
        "--camera", type=int, default=0, help="Camera index",
    )
    parser.add_argument(
        "--osc-port", type=int, default=9000, help="VRChat OSC send port",
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
