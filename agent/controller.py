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
import re
import time
import logging
import sys
import signal
from pathlib import Path
from typing import Optional


# Emoji ranges — strip before sending to TTS (Piper reads them literally)
_EMOJI_RE = re.compile(
    "[\U0001F600-\U0001F64F"   # emoticons
    "\U0001F300-\U0001F5FF"    # symbols & pictographs
    "\U0001F680-\U0001F6FF"    # transport
    "\U0001F1E0-\U0001F1FF"    # flags
    "\U00002700-\U000027BF"    # dingbats
    "\U0001F900-\U0001F9FF"    # supplemental
    "\U00002600-\U000026FF"    # misc symbols
    "\U0001FA70-\U0001FAFF"    # ext symbols A
    "\U0001F780-\U0001F7FF"    # geometric ext
    "]+",
    flags=re.UNICODE,
)


def _strip_emojis(text: str) -> str:
    """Remove emojis for TTS without modifying the chatbox text.

    Also collapses runs of whitespace produced by emoji removal so
    Piper doesn't get tripped up on double spaces.
    """
    no_emoji = _EMOJI_RE.sub("", text)
    return re.sub(r"\s+", " ", no_emoji).strip()


# Spatial cues in speech that should trigger a body turn
_TURN_CUES = {
    "behind you": ("right", "half"),
    "behind me": ("right", "half"),
    "turn around": ("right", "half"),
    "over here": None,              # direction-less, just face speaker
    "look at me": None,
    "look here": None,
}


# Gemma 4 E4B often emits tool calls as text content rather than using the
# OpenAI tool_calls format. Example: `send_chatbox{text:<|"|>hello<|"|>}`.
# This regex extracts those pseudo-calls so we can execute them properly
# instead of reading the syntax aloud. Supports both `<|"|>...<|"|>` and
# plain `"..."` quoting.
# Strict: value is between matching quote-delimiters
_PSEUDO_QUOTED_RE = re.compile(
    r"\*?(\w+)\s*\{\s*(\w+)\s*:\s*"
    r"(?:<\|\"\|>|\"|')"
    r"(.+?)"
    r"(?:<\|\"\|>|\"|')"
    r"\s*\}?\*?",
    re.DOTALL,
)

# Forgiving: unquoted value up to } or newline (for simple args like
# `gesture{type: wave}`)
_PSEUDO_UNQUOTED_RE = re.compile(
    r"\*?(\w+)\s*\{\s*(\w+)\s*:\s*"
    r"([^\"'<{}\n]+?)"
    r"\s*\}\*?",
    re.DOTALL,
)


def _extract_pseudo_calls(text: str) -> tuple[list[dict], str]:
    """Extract Gemma's pseudo-tool-calls from plain text content.

    Tries the strict quoted pattern first (covers the `<|"|>...<|"|>` case).
    Any leftover text is then scanned for unquoted tool syntax.

    Returns (calls, leftover_text).
    """
    calls = []
    remaining = text

    # Pass 1: quoted values
    new_calls_1 = []
    for m in _PSEUDO_QUOTED_RE.finditer(remaining):
        val = m.group(3).strip()
        if val:
            new_calls_1.append({
                "name": m.group(1).strip(),
                "arg_key": m.group(2).strip(),
                "arg_value": val,
            })
    if new_calls_1:
        calls.extend(new_calls_1)
        remaining = _PSEUDO_QUOTED_RE.sub("", remaining)

    # Pass 2: unquoted values (on what's left)
    for m in _PSEUDO_UNQUOTED_RE.finditer(remaining):
        val = m.group(3).strip()
        if val and not val.startswith("<"):
            calls.append({
                "name": m.group(1).strip(),
                "arg_key": m.group(2).strip(),
                "arg_value": val,
            })
    remaining = _PSEUDO_UNQUOTED_RE.sub("", remaining)

    # Clean up stray pipe/quote artifacts
    remaining = re.sub(r"<\|[^|]*\|>", "", remaining)
    remaining = re.sub(r"\*+", "", remaining).strip()
    return calls, remaining

from openai import OpenAI, AsyncOpenAI

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

        # --- Model clients ---
        # Sync client only used for the initial .models.list() probe.
        self.client = OpenAI(base_url=model_url, api_key="ollama")
        # Async client for generation — true cancellation support so we can
        # interrupt an in-flight LLM call when a new player utterance arrives.
        self._async_client = AsyncOpenAI(base_url=model_url, api_key="ollama")

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

        # In-flight generation tracking — for interruptible responses.
        # When a new player utterance arrives while we're generating:
        #   1. cancel the current task (kills the HTTP request)
        #   2. bump the id (so any late-completing response is discarded)
        #   3. cancel TTS if playing
        # Then a fresh _act runs with both utterances in dialogue.
        self._gen_id: int = 0
        self._gen_task: Optional[asyncio.Task] = None

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
        # Cancel any in-flight generation
        if self._gen_task and not self._gen_task.done():
            self._gen_task.cancel()
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

        # 2. New speech = INTERRUPT everything pending
        if speech:
            # (a) Stop any TTS currently playing
            if snap.tts_playing:
                logger.info(f'[BARGE-IN] Player spoke during TTS: "{speech[:50]}"')
                self.tts_router.cancel()
            # (b) Cancel any in-flight LLM generation + mark its response stale
            self._gen_id += 1
            if self._gen_task and not self._gen_task.done():
                logger.info("[INTERRUPT] cancelling in-flight generation")
                self._gen_task.cancel()
            # (c) Append to dialogue IMMEDIATELY so the next generation sees it
            self.dialogue.add_player(speech)

        # 3. Update state machine with hysteresis
        state = self.state_machine.update(
            players_visible=snap.perception.players_visible,
            speech_heard=bool(speech),
            tick_seconds=HEARTBEAT_INTERVAL,
        )

        # 4. Explicit follow/stop commands handled cheaply (no LLM)
        if speech:
            lower = speech.lower()
            if any(p in lower for p in ["follow me", "come with me", "come here"]):
                self.state_machine.force_follow()
            elif any(p in lower for p in ["stop following", "stop follow", "stay here"]):
                self.state_machine.force_stop_follow()

            # Spatial cues → immediate physical reaction before LLM thinks
            for cue, turn_args in _TURN_CUES.items():
                if cue in lower and turn_args:
                    direction, amount = turn_args
                    logger.info(f"[CUE] '{cue}' -> turn {direction} {amount}")
                    try:
                        self.look_tool.turn(direction=direction, amount=amount)
                    except Exception as e:
                        logger.warning(f"Cue turn failed: {e}")
                    break

        # 5. Periodic perception log
        now = snap.timestamp
        if now - self._last_perception_log > 10.0:
            logger.info(f"[{state.name}] {snap.perception.summary()}")
            self._last_perception_log = now

        # 6. Don't double-schedule: if a generation is already running
        #    and this tick has no speech, skip.
        if self._gen_task and not self._gen_task.done() and not speech:
            return

        # 7. Decide if we should act
        reason = self._should_act(state, speech, now)
        if not reason:
            return

        # 8. Schedule _act as a cancellable task — does NOT block the loop.
        self._last_act_time = now
        self._gen_task = asyncio.create_task(
            self._act(reason, state, speech, snap, self._gen_id)
        )

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

    async def _act(
        self,
        reason: str,
        state: AgentState,
        speech: str,
        snap: SignalsSnapshot,
        gen_id: int,
    ) -> None:
        """Interruptible LLM call — cancelled via asyncio if new speech arrives.

        Runs as a Task spawned by _tick. If cancelled mid-flight, the HTTP
        request to Ollama is aborted. Even if it completes, we double-check
        gen_id matches the current one — otherwise the response is stale
        (a newer utterance replaced it) and gets discarded.
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

        trigger = self._reason_to_trigger(reason, state, speech)

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

        logger.info(
            f"[{state.name}] act({reason}) gen#{gen_id}: "
            f"system={len(system_prompt)}ch dialogue={len(self.dialogue)}t"
        )

        # Mute audio capture BEFORE the model call (audit 3.5)
        self.audio_capture.muted = True

        t_start = time.time()
        try:
            response = await self._async_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=self._tool_definitions,
                tool_choice="auto",
                temperature=1.0,
                top_p=0.95,
                max_tokens=512,
            )
        except asyncio.CancelledError:
            logger.info(f"  [CANCELLED] gen#{gen_id} aborted by interruption")
            self.audio_capture.muted = False
            raise
        except Exception as e:
            logger.error(f"Model call failed: {e}")
            self.audio_capture.muted = False
            return

        # Unmute now — response is in hand
        self.audio_capture.muted = False

        # Staleness check: did a newer generation bump the id while we were waiting?
        if gen_id != self._gen_id:
            logger.info(f"  [STALE] gen#{gen_id} discarded (current is #{self._gen_id})")
            return

        elapsed = time.time() - t_start
        msg = response.choices[0].message

        # Track what Chati said for dialogue continuity
        spoken_text = self._handle_response(msg)

        if spoken_text:
            self.dialogue.add_chati(spoken_text)

        usage = getattr(response, "usage", None)
        if usage:
            logger.info(
                f"  <<- {elapsed:.1f}s | {usage.prompt_tokens} in / "
                f"{usage.completion_tokens} out"
            )

    def _reason_to_trigger(self, reason: str, state: AgentState, speech: str) -> str:
        """Convert a 'reason' code into a short user-facing trigger line.

        Note: we only get reliable signal for *sight* (YOLO/OCR) and for
        *direct speech aimed at Chati* (Whisper STT). We have no idea
        what ambient conversations are happening around her. So triggers
        must avoid phrases like "it's quiet" that Chati would parrot.
        """
        if reason == "they_spoke":
            return (
                f'Someone said to you: "{speech}"\n\n'
                "Reply to THEM. Say what you actually think — brief if it's "
                "a quick back-and-forth, longer if you have a real thought. "
                "Use a gesture if it fits. Don't change the subject."
            )
        if reason == "just_arrived":
            return (
                "You just noticed someone. React to what they actually look "
                "like or what's happening on screen. Be specific — no generic "
                "openers. Say as much or as little as feels natural."
            )
        if reason == "lull":
            return (
                "No one has said anything to you in a while. React to "
                "something you SEE, or share a passing thought — the kind "
                "an older entity stuck in here might have. Don't comment "
                "on noise level or ask 'is anyone there'."
            )
        if reason == "following_tick":
            return (
                "You're following them. Move forward if they're far, "
                "turn toward them if they're to the side."
            )
        if reason == "idle":
            return (
                "Nothing's grabbing your attention right now. Wander, "
                "look somewhere new, or gesture. Don't narrate."
            )
        return "React to what's happening."

    def _handle_response(self, msg) -> str:
        """Execute tool calls; return spoken text for dialogue log.

        Handles THREE cases:
          1. Proper OpenAI tool_calls array (preferred)
          2. Pseudo-tool-calls embedded in content (Gemma quirk) — parsed
             and executed as real calls instead of spoken aloud
          3. Plain conversational text with no tool syntax — sent as chatbox
        """
        spoken_pieces: list[str] = []
        content = (msg.content or "").strip()

        # --- 1. Proper tool calls from OpenAI API ---
        if msg.tool_calls:
            for tc in msg.tool_calls:
                spoken = self._execute_tool_call(tc)
                if spoken:
                    spoken_pieces.append(spoken)

        # --- 2. Pseudo-tool-calls hiding inside content ---
        if content:
            pseudo_calls, leftover = _extract_pseudo_calls(content)
            if pseudo_calls:
                for pc in pseudo_calls:
                    spoken = self._run_pseudo_call(pc)
                    if spoken:
                        spoken_pieces.append(spoken)
                content = leftover  # what's left after pulling out tool syntax

            # --- 3. Plain conversational text — speak as chatbox ---
            if content:
                # Sanity: skip if it's just punctuation or a stray quote
                if len(content.strip("\"'` {}<>|")) > 1:
                    logger.info(f"  [TEXT] {content[:100]}")
                    # chatbox truncates to 144 chars; speak gets the full text
                    self.chatbox_tool.send_chatbox(content)
                    tts_text = _strip_emojis(content)
                    if tts_text:
                        self.speak_tool.speak(tts_text)
                    spoken_pieces.append(content)

        return " ".join(spoken_pieces).strip()

    def _run_pseudo_call(self, pc: dict) -> str:
        """Execute a pseudo-tool-call extracted from text content.

        pc has keys: name, arg_key, arg_value.
        Returns the spoken text if this was a chatbox/speak call, else "".
        """
        name = pc["name"]
        arg_key = pc["arg_key"]
        arg_value = pc["arg_value"]

        # Skip empty args (malformed / truncated output)
        if not arg_value:
            logger.debug(f"  [SKIP pseudo] {name}(empty)")
            return ""

        handler = self._tool_handlers.get(name)
        if not handler:
            logger.warning(f"  [pseudo] unknown tool: {name}")
            return ""

        args = {arg_key: arg_value}
        logger.info(f"  [PSEUDO] {name}({arg_key}={arg_value!r})")

        try:
            handler(**args)
        except Exception as e:
            logger.error(f"  [pseudo] {name} failed: {e}")
            return ""

        spoken_text = ""
        if name == "send_chatbox":
            spoken_text = arg_value
            tts_text = _strip_emojis(arg_value)
            if tts_text:
                self.speak_tool.speak(tts_text)
        elif name == "speak":
            spoken_text = arg_value

        return spoken_text

    def _execute_tool_call(self, tool_call) -> str:
        """Execute one tool call. Returns text (for dialogue log) if it was a chatbox/speak."""
        name = tool_call.function.name
        try:
            args = json.loads(tool_call.function.arguments or "{}")
        except json.JSONDecodeError:
            logger.error(f"Bad tool args: {tool_call.function.arguments}")
            return ""

        # Skip no-op tool calls (model sometimes emits empty chatbox)
        if name == "send_chatbox":
            text = (args.get("text") or "").strip()
            if not text:
                logger.debug("  [SKIP] empty send_chatbox")
                return ""
            args["text"] = text

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

        # Controller owns "say this out loud" (audit 1.7 — no hidden coupling)
        spoken_text = ""
        if name == "send_chatbox":
            raw_text = args.get("text", "")
            spoken_text = raw_text
            # Strip emojis for TTS but keep the raw text for the dialogue log
            tts_text = _strip_emojis(raw_text)
            if tts_text:
                self.speak_tool.speak(tts_text)
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
