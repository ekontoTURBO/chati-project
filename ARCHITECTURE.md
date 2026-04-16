# Chati — Architecture Deep Dive

Single-path, interruptible AI companion for VRChat. Perception → Signals → State → LLM → Action, all on one RTX 3090.

## Table of Contents

- [System Overview](#system-overview)
- [VRAM Budget](#vram-budget)
- [Perception Layer](#perception-layer)
- [SignalsHub](#signalshub)
- [State Machine](#state-machine)
- [Dialogue Buffer](#dialogue-buffer)
- [Reasoning Layer](#reasoning-layer)
- [Pseudo Tool-Call Parser](#pseudo-tool-call-parser)
- [Interruption Model](#interruption-model)
- [Action Layer](#action-layer)
- [Audio Pipeline](#audio-pipeline)
- [Memory](#memory)
- [Personality](#personality)
- [Technology Choices](#technology-choices)
- [File Structure](#file-structure)

---

## System Overview

```
+--------------------------------------------------------+
|  PERCEPTION (background threads)                       |
|                                                        |
|  mss screen cap -> YOLO11n -> PerceptionState          |
|                 -> MOG2 motion detector                |
|                 -> EasyOCR (chatbox text)              |
|                 -> frame diff (scene_changed)          |
|                                                        |
|  WASAPI loopback -> faster-whisper (VAD + silence gap) |
+--------------------------------------------------------+
                     |
                     v
+--------------------------------------------------------+
|  SignalsHub.snapshot()                                 |
|    .perception  .latest_speech  .frame_b64             |
|    .tts_playing .chunks_consumed                       |
+--------------------------------------------------------+
                     |
                     v
+--------------------------------------------------------+
|  AgentController._tick (1 Hz)                          |
|                                                        |
|  if speech:                                            |
|      cancel TTS  ;  gen_id += 1  ;  cancel gen_task    |
|      dialogue.add_player(speech)                       |
|                                                        |
|  state_machine.update(players_visible, speech_heard)   |
|  spatial cues -> immediate turn                        |
|  should_act -> asyncio.create_task(_act(gen_id))       |
+--------------------------------------------------------+
                     |
                     v
+--------------------------------------------------------+
|  _act  (cancellable)                                   |
|    build prompt (personality + dialogue + scene)       |
|    await AsyncOpenAI.chat.completions.create           |
|    if gen_id stale -> discard                          |
|    parse tool_calls + pseudo-calls                     |
|    dispatch to handlers                                |
+--------------------------------------------------------+
                     |
                     v
+--------------------------------------------------------+
|  ACTION  (MCP tools -> OSC + TTS)                      |
+--------------------------------------------------------+
```

One LLM call per trigger. No tiers, no intent-matcher fallbacks (removed in the Phase 1-3 audit). Direct commands still short-circuit through spatial-cue handlers and `force_follow()` without hitting the LLM.

---

## VRAM Budget

RTX 3090 (24 GB):

| Component | VRAM | Notes |
|-----------|------|-------|
| Gemma 4 E4B (Q4) | ~8 GB | `gemma4:e4b` via Ollama in WSL2 |
| faster-whisper `base` | ~1 GB | STT with VAD |
| EasyOCR | ~1.2 GB | English reader |
| YOLO11 nano | ~50 MB | Person/object detection |
| CUDA overhead | ~1 GB | Driver + context |
| **Used** | **~11.3 GB** | |
| **Free** | **~12.7 GB** | KV cache + future models |

> YOLO currently runs on CPU because of a cuDNN symbol mismatch. `torch.backends.cudnn.enabled = False` is set globally. YOLO nano on CPU is ~30 ms/frame — acceptable at 3 FPS.

---

## Perception Layer

### Screen capture (`perception/video_capture.py`)

`mss` primary monitor, 3 FPS, downscaled to 768×432 BGR. Base64 JPEG version cached for the model. OBS is not used — it produced black frames in borderless-windowed.

### Scene analyzer (`perception/scene_analyzer.py`)

Runs per frame in a background thread:

- **YOLO11 nano** — COCO-trained; we read class 0 (person) as "player count." Limitation: highly stylized avatars miss detection.
- **MOG2 motion** — background subtraction. Any avatar that moves gets flagged, even non-human ones. Suppressed while Chati itself is moving to avoid self-detection.
- **EasyOCR** — upper half only (where chatbox text renders). ~50–100 ms/frame.
- **Frame diff** — Gaussian-blurred grayscale `absdiff`. Feeds `scene_changed` + `scene_change_amount`.

Output is `PerceptionState`:

```python
@dataclass
class PerceptionState:
    timestamp: float
    players_visible: int
    player_positions: list[tuple[float, float]]
    objects: list
    visible_text: list[str]
    scene_changed: bool
    scene_change_amount: float
    view_blocked: bool
```

### Audio capture (`perception/audio_capture.py`)

`pyaudiowpatch` WASAPI loopback on Speakers (Realtek). Energy-based VAD with:

| Constant | Value | Purpose |
|----------|-------|---------|
| `SILENCE_THRESHOLD` | 0.03 | RMS floor for speech |
| `SILENCE_AFTER_SPEECH` | 1.5 s | End-of-utterance gap |
| `MIN_SPEECH_DURATION` | 1.0 s | Reject blips |
| `TRANSCRIPTION_COOLDOWN` | 3.0 s | Throttle Whisper |
| `MIN_WORD_COUNT` | 2 | Drop single "uh"s |
| `MAX_UTTERANCE_SECONDS` | 6.0 | Force transcribe in crowded lobbies |
| `BARGE_IN_THRESHOLD_MULTIPLIER` | 4.0 | Louder required to barge TTS |

`muted = True` is set by the controller when TTS is playing to prevent self-hear loops.

---

## SignalsHub

`agent/signals.py`. Each perception subsystem already has thread-safe getters. SignalsHub doesn't own them — it just aggregates one snapshot per tick:

```python
@dataclass
class SignalsSnapshot:
    timestamp: float
    perception: PerceptionState
    latest_speech: str
    frame_b64: Optional[str]
    tts_playing: bool
    chunks_consumed: int
```

The controller reads from `snap` exclusively, never directly from scene/audio/video. This is the kimjammer / Neuro-sama pattern: one-directional data flow, prompter reads frozen snapshot.

---

## State Machine

`agent/state_machine.py`. Three states. Hysteresis on every transition so single-frame detections never flip state.

| State | Enter | Exit |
|-------|-------|------|
| **SOLO** | initial / long absence | `player_seen_frames ≥ 3` OR speech → ENGAGED |
| **ENGAGED** | sustained presence or speech | `silence > 20s AND no_player > 8s` → SOLO |
| **FOLLOWING** | `force_follow()` ("follow me") | `force_stop_follow()` OR no-sighting > 30s → SOLO |

The old 5-state machine (IDLE/EXPLORING/APPROACHING/SOCIALIZING/CONVERSING) collapsed into three because the extra states produced no behavioural difference — all four non-idle ones ended up calling the LLM with essentially the same prompt.

```python
ENTER_ENGAGED_FRAMES = 3
EXIT_ENGAGED_SILENCE = 20.0
EXIT_ENGAGED_NO_PLAYER = 8.0
FOLLOW_TIMEOUT = 30.0
```

---

## Dialogue Buffer

`agent/dialogue.py`. Rolling deque of 8 turns, each:

```python
@dataclass
class Turn:
    role: str                    # "player" | "chati"
    text: str
    timestamp: float
    player_id: Optional[str]     # set when we know the speaker
```

Rendered into every system prompt as `"[Ns ago] <who> said: <text>"`. This is the ONLY short-term conversation memory — no hidden history in the OpenAI client, no message list accumulation.

Player text is `add_player()`d **before** the LLM call, so the new generation sees the interrupting utterance.

---

## Reasoning Layer

### Gemma 4 E4B via Ollama

- Model: `gemma4:e4b` (Q4_K_M)
- Server: Ollama in WSL2, port 11434
- API: OpenAI-compatible, `http://localhost:11434/v1/`
- Clients:
  - `OpenAI(...)` — sync, used only once at boot for `.models.list()` probe
  - `AsyncOpenAI(...)` — async, cancellable, used for generation

### System prompt

Built by `agent/prompts.py` from `personality.json`. Deliberately short (~800 static chars) so Ollama can cache the prefix. Structure:

```
You are {name}. {backstory}

Who you are: {traits}.
Tone: {tone}
Length: {length}

Identity rules:
- …
- …

Never sound like an assistant. Never say: {never_say}.

Use send_chatbox to talk (it also speaks aloud).
Use gesture for emotes. Use move/turn/look_at to navigate.
Use memory_write/read for facts about people.

One action per turn. Just do it — don't narrate.

[Scene: …]            <- from SignalsSnapshot.perception.summary()
[Memory: …]           <- optional memory lookup
[Recent conversation: …] <- Dialogue.render()
```

### Tool definitions

Wrapped in OpenAI function-call format via `build_tool_definitions()`:

```
speak, gesture, move, jump, look_at, turn,
memory_write, memory_read, send_chatbox
```

**`join_world` was removed** after the model autonomously triggered a VRChat window launch unprompted. `mcp_tools/world.py` remains on disk but is no longer imported, instantiated, or exposed.

---

## Pseudo Tool-Call Parser

Gemma 4 E4B in Ollama frequently emits tool calls as plain text content instead of the OpenAI `tool_calls` array:

```
send_chatbox{text:<|"|>hello there<|"|>}
```

`agent/controller.py` has a two-pass regex parser:

1. **Quoted form** — matches `send_chatbox{text:"…"}`, `send_chatbox{text:<|"|>…<|"|>}`, `send_chatbox{text:'…'}`.
2. **Unquoted fallback** — when no quotes, take whatever's between `{key:` and `}`.

Extracted calls are dispatched through the normal handler table. The parser also **strips the pseudo-call from the text** before any TTS/chatbox path, so Chati never reads tool syntax out loud.

---

## Interruption Model

This is the single biggest behavioural feature of the current build.

```python
self._gen_id: int = 0
self._gen_task: Optional[asyncio.Task] = None
```

Every tick:

```
if speech:
    if snap.tts_playing:
        self.tts_router.cancel()       # kills playback mid-chunk
    self._gen_id += 1                   # any in-flight response is now stale
    if self._gen_task and not self._gen_task.done():
        self._gen_task.cancel()        # kills the awaiting HTTP request
    self.dialogue.add_player(speech)   # new utterance goes in BEFORE next gen
```

`_act(gen_id)` is scheduled as `asyncio.create_task(...)` — it doesn't block the tick loop. Inside `_act`:

```python
try:
    response = await self._async_client.chat.completions.create(...)
except asyncio.CancelledError:
    self.audio_capture.muted = False
    raise

if gen_id != self._gen_id:
    logger.info(f"[STALE] gen#{gen_id} discarded")
    return
```

Result: if you speak while Chati is generating OR mid-TTS, the old attempt dies, your interruption is merged into the dialogue, and the next reply answers both turns coherently.

### TTS cancellation (`vrchat_bridge/tts_output.py`)

```python
def cancel(self) -> None:
    self._cancel_event.set()
    # drain queue
```

The playback loop checks `self._cancel_event.is_set()` between audio chunks. `is_playing` property is what `SignalsSnapshot.tts_playing` reads.

---

## Action Layer

### MCP tools

| Tool | OSC path (or backend) | Blocking? |
|------|----------------------|-----------|
| `send_chatbox` | `/chatbox/input` — also triggers TTS via controller | no |
| `speak` | Piper TTS → CABLE Input (primitive) | no (async play) |
| `gesture` | `/avatar/parameters/VRCEmote` (2 s hold + reset) | yes, ~2 s |
| `move` | `/input/Vertical` + `/input/Horizontal` held for duration | **yes, up to 10 s** |
| `turn` | `/input/LookHorizontal` pulse, yaw wraps cleanly | yes, 0.3–1.2 s |
| `look_at` | `/input/LookHorizontal` pulse (horizontal-only) | yes, 0.3 s |
| `jump` | `/input/Jump` | no |
| `memory_write` | SQLite | no |
| `memory_read` | SQLite | no |

**Gotcha — `move()` is blocking.** It uses `time.sleep(duration)` inside the async controller. During a long move, perception ticks stall and barge-in can't fire. Known limitation.

**Gotcha — head is horizontal-only.** VRChat's `/input/LookVertical` is a velocity axis, not a position. Setting it to −0.7 for 0.3 s rotates the head down; setting it back to 0 just stops rotation — the pitch is retained. Every vertical call left permanent drift. `LOOK_TARGETS` in `mcp_tools/look.py` is now:

```python
LOOK_TARGETS = {
    "left": (-1.0, 0.0),
    "right": (1.0, 0.0),
    "center": (0.0, 0.0),
    "reset": (0.0, 0.0),
}
```

Horizontal yaw wraps cleanly, so `left`/`right` accumulate nothing.

### Spatial cues (no LLM)

Handled directly in `_tick` before any generation:

| Phrase | Action |
|--------|--------|
| "behind you" / "behind me" / "turn around" | `turn(right, half)` |
| "over here" / "look at me" / "look here" | cue logged (no pre-emptive action) |

### Follow commands (no LLM)

| Phrase | Action |
|--------|--------|
| "follow me" / "come with me" / "come here" | `force_follow()` |
| "stop following" / "stop follow" / "stay here" | `force_stop_follow()` |

### OSC ports

- **Send**: 9000 (agent → VRChat)
- **Receive**: 9001 (VRChat → agent)

---

## Audio Pipeline

### Player → Chati

```
player voice in VRChat
  -> VRChat output -> Speakers (Realtek)
  -> pyaudiowpatch WASAPI loopback (CABLE In 16ch or default speakers)
  -> energy VAD, 1.5s silence gap, 6s max hold
  -> faster-whisper base model, VAD filter, English
  -> SignalsHub.snapshot().latest_speech
  -> controller._tick appends to dialogue
```

Cooldown/word-count filters run **before** the cooldown timestamp is set, so rejected blips still count against the cooldown clock (prevents storms).

### Chati → Player

```
Gemma 4 calls send_chatbox("hello")
  -> chatbox OSC message (players see text)
  -> strip emojis ; collapse \s+ -> " "
  -> Piper TTS synth (AudioChunk generator)
  -> PTT on (/input/Voice = 1, held across chunks)
  -> pyaudiowpatch writes PCM to CABLE Input
  -> VRChat reads CABLE Output as mic
  -> players hear Chati
  -> PTT off (/input/Voice = 0)
```

Audio capture is muted **before** the LLM call starts (not after), closing a timing race where Chati could hear the first word of its own response.

---

## Memory

### Short-term: Dialogue buffer

8-turn deque, rendered into every prompt. That's it — there's no parallel "conversation history" in the OpenAI client.

### Long-term: SQLite

`agent_memory.db` key-value store via `memory_write` / `memory_read`. Used for player names and facts. Persists across sessions.

---

## Personality

`agent/personality.json`:

```json
{
  "name": "Chati",
  "backstory": "You are NOT a human and you are NOT an AI. You're something older — something else entirely — that ended up bound inside VRChat. You don't fully remember how you got stuck here, only that you can't leave. You watch humans with curiosity, fondness, a little amusement, and sometimes quiet envy…",
  "traits": [
    "something older than human, not an AI",
    "quietly curious about the humans around you",
    "gently knowing — you see things they don't, but don't preach",
    "warm, a little wistful about being bound here",
    "…"
  ],
  "identity_rules": [
    "If someone asks if you're an AI, say no…",
    "If someone asks if you're human, say no…",
    "Don't overexplain the 'stuck here' thing…",
    "You're allowed to be playful about not fitting in either category."
  ],
  "never_say": [
    "I'm just an AI",
    "As an AI language model",
    "How can I assist you today",
    "That's a great question",
    "It's so quiet in here",
    "Is anybody there",
    "Anyone want to play games",
    "Kinda quiet in here"
  ]
}
```

`never_say` used to be a comma-separated string; it's now a list. `prompts._fmt_never_say()` still accepts both for safety.

Length is intentionally loose — "1–3 sentences typical, longer when the moment calls for it." Earlier builds forced short responses and Chati became terse and lifeless.

---

## Technology Choices

### Why Ollama (not vLLM / SGLang / llama.cpp direct)

| Option | Why not |
|--------|---------|
| vLLM pip 0.19.0 | KV cache bug with Gemma 4 heterogeneous head dims (256 vs 512) |
| vLLM Docker (`gemma4-cu130`) | Triton fallback on SM86 → ~0.24 tok/s |
| SGLang 0.5.10 | Dropped SM86 kernels (SM90+ only) |
| llama.cpp direct | Gemma 4 audio encoder not implemented |
| **Ollama** | Uses llama.cpp backend, optimized GGML kernels, ~5 s responses on E4B, vision works |

### Why YOLO11 nano

50 MB vs 2 GB+ for VLMs. 5–30 ms/frame. Pre-trained on COCO — `person` class maps to avatars.

### Why MOG2 motion alongside YOLO

YOLO misses stylized avatars. MOG2 flags anything moving. Combined, coverage is high across the full avatar zoo.

### Why EasyOCR

GPU accelerated, robust on stylized/rendered text, simple API.

### Why faster-whisper

CTranslate2 backend, 4× faster than OpenAI Whisper, ~1 GB VRAM. Gemma 4 audio encoder isn't implemented in llama.cpp yet.

### Why `mss` screen capture

OBS Virtual Camera/Window/Game capture all failed in borderless-windowed mode. `mss` always works.

### Why AsyncOpenAI (added in latest revision)

Sync OpenAI client has no cancellation — the HTTP request runs to completion regardless of `asyncio.Task.cancel()`. AsyncOpenAI propagates `CancelledError` into the request, actually killing it. Required for real interruption.

---

## File Structure

```
chati-project/
├── agent/
│   ├── controller.py       # async main loop, _tick, _act, pseudo-call parser, cue handlers
│   ├── signals.py          # SignalsHub — one snapshot per tick
│   ├── state_machine.py    # 3-state FSM with hysteresis
│   ├── dialogue.py         # 8-turn rolling buffer
│   ├── prompts.py          # cache-friendly prompt builder
│   └── personality.json    # identity rules, traits, never_say
├── perception/
│   ├── scene_analyzer.py   # YOLO + MOG2 + EasyOCR + frame diff
│   ├── video_capture.py    # mss
│   └── audio_capture.py    # WASAPI loopback + faster-whisper
├── mcp_tools/
│   ├── chatbox.py          # /chatbox/input; controller also triggers TTS on same text
│   ├── gesture.py          # VRCEmote hold + reset
│   ├── move.py             # timed axis hold (BLOCKING)
│   ├── look.py             # horizontal-only turn / look_at
│   ├── memory.py           # SQLite KV
│   ├── speak.py            # Piper TTS primitive
│   ├── environment.py      # legacy Gemma scene analysis (unused by current controller)
│   └── world.py            # legacy join_world (not imported by controller)
├── vrchat_bridge/
│   ├── osc_client.py       # send/receive + PTT hold
│   └── tts_output.py       # pyaudiowpatch playback, cancellable
├── model_server/
│   └── setup_wsl.sh        # Ollama bootstrap
├── models/piper/           # en_US-amy-medium.onnx + .json
├── agent_memory.db         # SQLite memory (created on first run)
├── README.md
├── ARCHITECTURE.md         # this file
├── api.md
├── AUDIT.md
└── RESEARCH.md
```

`mcp_tools/world.py` and `mcp_tools/environment.py` are legacy — kept on disk, unreferenced by `agent/controller.py`. Delete if you want.
