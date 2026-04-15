# Chati — Architecture Deep Dive

This document describes the full architecture of Chati, a local AI companion for VRChat that can see, hear, speak, move, and socialize — all running on a single RTX 3090.

## Table of Contents

- [System Overview](#system-overview)
- [VRAM Budget](#vram-budget)
- [Perception Layer](#perception-layer)
- [State Machine](#state-machine)
- [Reasoning Layer](#reasoning-layer)
- [Action Layer](#action-layer)
- [Audio Pipeline](#audio-pipeline)
- [Memory System](#memory-system)
- [Data Flow](#data-flow)
- [Voice Output Pipeline](#voice-output-pipeline)
- [Following System](#following-system)
- [Personality System](#personality-system)
- [Technology Choices](#technology-choices)

---

## System Overview

Chati follows a **perception-state-reasoning-action** loop:

```
+----------------------------------------------------------+
|  PERCEPTION (background threads, always running)          |
|                                                           |
|  Screen Capture (1 FPS) ----+                             |
|                              |                             |
|  YOLO11 nano (GPU, ~5ms) ---+--> PerceptionState          |
|    - player count/positions  |     .players_visible: 2     |
|    - object detection        |     .objects: [couch, TV]   |
|                              |     .visible_text: ["hi"]   |
|  EasyOCR (GPU) -------------+     .view_blocked: false     |
|    - chatbox text reading    |     .scene_changed: true    |
|                              |                             |
|  Frame Differencing (CPU) --+                              |
|    - scene change detection                                |
|                                                            |
|  Whisper STT (GPU) ---------> speech transcriptions        |
+----------------------------------------------------------+
                    |                          |
                    v                          v
+----------------------------------------------------------+
|  STATE MACHINE                                            |
|                                                           |
|  IDLE -> EXPLORING -> APPROACHING -> SOCIALIZING          |
|                   ^                       |                |
|                   |       CONVERSING <----+                |
|                   |           |                            |
|                   +-----------+ (player leaves)            |
+----------------------------------------------------------+
                    |
                    v
+----------------------------------------------------------+
|  REASONING (Gemma 4 via Ollama)                           |
|                                                           |
|  Input:  perception state + video frame + speech + memory |
|  Output: tool calls (move, chatbox, gesture, look, etc.)  |
|                                                           |
|  Called at varying intervals per state:                    |
|    CONVERSING: 2s | APPROACHING: 3s | EXPLORING: 8s      |
+----------------------------------------------------------+
                    |
                    v
+----------------------------------------------------------+
|  ACTION (MCP Tools -> VRChat OSC)                         |
|                                                           |
|  speak -> Piper TTS -> VB-Cable -> VRChat mic             |
|  move/gesture/look/jump -> OSC -> VRChat avatar           |
|  send_chatbox -> OSC -> VRChat chatbox                    |
|  memory_write/read -> SQLite                              |
|  join_world -> vrchat:// deep link                        |
+----------------------------------------------------------+
```

---

## VRAM Budget

Running on NVIDIA RTX 3090 (24 GB VRAM):

| Component | VRAM | Model | Purpose |
|-----------|------|-------|---------|
| Gemma 4 E2B Q4 | ~8 GB | `gemma4:e2b` via Ollama | Reasoning, decisions, conversation |
| faster-whisper | ~1 GB | `base` model | Speech-to-text |
| YOLO11 nano | ~50 MB | `yolo11n.pt` | Person/object detection |
| EasyOCR | ~1.2 GB | English reader | Chatbox text recognition |
| Frame differencing | 0 (CPU) | OpenCV | Scene change detection |
| CUDA overhead | ~1 GB | — | Driver, context, buffers |
| **Total used** | **~11.3 GB** | | |
| **Headroom** | **~12.7 GB** | | KV cache, spikes, future models |

---

## Perception Layer

### Screen Capture (`perception/video_capture.py`)

Captures the primary monitor at 1 FPS using `mss` (Python screen capture library). No OBS Virtual Camera needed — mss grabs raw pixels directly.

- **Resolution**: 768x432 (downscaled from 2560x1440)
- **Format**: BGR numpy array, also base64 JPEG for model input
- **Why not OBS**: OBS Virtual Camera couldn't capture VRChat in fullscreen/borderless mode. mss always works.

### Scene Analyzer (`perception/scene_analyzer.py`)

Runs in a background thread, processing every captured frame with:

#### YOLO11 Nano — Object/Player Detection
- **Model**: `yolo11n.pt` (Ultralytics) — pre-trained on COCO dataset
- **Speed**: ~5ms per frame on GPU
- **Detects**: People (COCO class 0 = player avatars), objects (chairs, TVs, etc.)
- **Output**: Player count, positions (normalized 0-1), object labels
- **Limitation**: YOLO is trained on real-world images, not VRChat avatars. It detects humanoid shapes but may miss very stylized or non-human avatars.

References:
- [Ultralytics YOLO11 Docs](https://docs.ultralytics.com/models/yolo11/)
- [YOLO11 Architecture](https://arxiv.org/html/2410.17725v1)

#### EasyOCR — Chatbox Text Reading
- **Library**: EasyOCR with GPU acceleration
- **Focus area**: Upper half of frame (where VRChat chatbox text appears)
- **Speed**: ~50-100ms per frame
- **Purpose**: Read what players type in their VRChat chatbox, so Chati can respond to text messages
- **VRAM**: ~1.2 GB

References:
- [EasyOCR GitHub](https://github.com/JaidedAI/EasyOCR)

#### Frame Differencing — Change Detection
- **Method**: OpenCV `absdiff` on grayscale frames with Gaussian blur
- **Cost**: Zero GPU, minimal CPU
- **Purpose**: Detect if the scene changed (player moved, new area, etc.)
- **Used for**: Stuck detection (if scene hasn't changed for 3+ ticks, agent is stuck against a wall)

#### View Blocked Detection
- **Method**: Edge density (Canny) + color variance
- **Logic**: Very few edges + low color variance = staring at a wall
- **Purpose**: Tell the agent to turn around instead of walking forward into nothing

### Combined Output: `PerceptionState`

All perception data is combined into a single `PerceptionState` dataclass:

```python
@dataclass
class PerceptionState:
    timestamp: float
    players_visible: int
    player_positions: list      # [(x, y), ...] normalized 0-1
    objects: list               # [DetectedObject, ...]
    visible_text: list          # ["hello", "nice avatar"]
    scene_changed: bool
    scene_change_amount: float
    view_blocked: bool
```

The agent controller reads this every tick without blocking on GPU inference.

---

## State Machine

### States (`agent/state_machine.py`)

| State | Trigger | Behavior | LLM Interval |
|-------|---------|----------|-------------|
| **IDLE** | Startup | Wait for first perception data | 5s |
| **EXPLORING** | No players visible | Roam, look around, comment on scene | 8s |
| **APPROACHING** | Player detected | Stop, wave, greet with chatbox | 3s |
| **SOCIALIZING** | After greeting | Observe player, react, chat | 12s |
| **CONVERSING** | Speech detected | Respond to what player said | 2s |
| **FOLLOWING** | Player asks "follow me" | Follow player movement | 5s |

### Transition Rules

```
IDLE --------[any perception]--------> EXPLORING
EXPLORING ---[players > 0]-----------> APPROACHING
APPROACHING -[greeted]---------------> SOCIALIZING
SOCIALIZING -[speech detected]-------> CONVERSING
CONVERSING --[15s no speech]---------> SOCIALIZING
SOCIALIZING -[players == 0]----------> EXPLORING
APPROACHING -[players == 0]----------> EXPLORING
CONVERSING --[players == 0]----------> EXPLORING
```

### State Context

Persistent context tracks:
- `greeted`: Whether we've greeted the current player
- `conversation_turns`: How many exchanges in current conversation
- `stuck_counter`: Increments when scene doesn't change
- `recent_actions`: Last 10 tool calls (for anti-repetition)
- `recent_chatbox`: Last 5 chatbox messages (prevents repeating)

---

## Reasoning Layer

### Gemma 4 E2B via Ollama

- **Model**: `gemma4:e2b` — Google's multimodal model (text + vision)
- **Quantization**: Q4_K_M (Ollama default) — fits in ~8GB VRAM
- **Context**: 8192 tokens (reduced from 128K default to save VRAM)
- **Server**: Ollama running in WSL2, port 11434
- **API**: OpenAI-compatible at `http://localhost:11434/v1/`

### When Gemma 4 is Called

Gemma 4 is NOT called for perception (that's YOLO/OCR/frame diff). It's only called for **decision-making**:

1. **EXPLORING**: "Here's what I see [perception data + frame]. What should I do?" -> move/look/comment
2. **APPROACHING**: "I see a player [perception]. Greet them." -> wave + chatbox
3. **SOCIALIZING**: "I'm near a player [perception + chatbox text]. React." -> gesture/chatbox
4. **CONVERSING**: "Player said '[whisper transcription]' [frame]. Respond." -> chatbox reply

### Prompt Structure

Each call includes:
- System prompt (personality, tool definitions, memory)
- Perception data (player count, positions, detected text, blocked view)
- Video frame (base64 JPEG, 768x432)
- Anti-repetition context (recent actions and chatbox messages)
- Speech transcription (if CONVERSING)

---

## Action Layer

### MCP Tools

| Tool | Transport | Purpose |
|------|-----------|---------|
| `speak` | Piper TTS -> VB-Cable 16ch | Voice output |
| `send_chatbox` | OSC `/chatbox/input` | Text above avatar head |
| `gesture` | OSC `/avatar/parameters/VRCEmote` | Wave, dance, cheer, etc. (2s then reset) |
| `move` | OSC `/input/Vertical` + `/input/Horizontal` | Walk in directions (timed, auto-stop) |
| `look_at` | OSC `/input/LookHorizontal` + `/input/LookVertical` | Turn head (0.3s impulse, auto-reset) |
| `jump` | OSC `/input/Jump` | Jump |
| `memory_write` | SQLite | Store facts about players |
| `memory_read` | SQLite | Recall stored facts |
| `environment_query` | Gemma 4 API | Deep scene analysis (legacy, rarely used) |
| `join_world` | `vrchat://` deep link | Navigate to a VRChat world |

### OSC Protocol

VRChat exposes avatar control via OSC (Open Sound Control) on UDP:
- **Send**: Port 9000 (agent -> VRChat)
- **Receive**: Port 9001 (VRChat -> agent)

---

## Audio Pipeline

### Capture Path (Player -> Agent)

```
Player speaks in VRChat
    -> VRChat audio output to "CABLE In 16ch"
    -> pyaudiowpatch WASAPI loopback capture
    -> Energy-based voice activity detection
    -> Wait for 1.5s silence (end of utterance)
    -> faster-whisper transcribes to text
    -> Text sent to state machine -> CONVERSING state
    -> Gemma 4 generates response
```

### Output Path (Agent -> VRChat)

```
Gemma 4 calls speak("hello!")
    -> Piper TTS generates WAV audio
    -> Audio plays to "CABLE In 16ch" via pyaudiowpatch
    -> VRChat reads CABLE Output as mic input
    -> Players hear Chati speak
```

### Feedback Prevention

Audio capture is muted (`self.audio_capture.muted = True`) while TTS is playing to prevent Chati from hearing and responding to its own voice.

---

## Memory System

### Short-term: Conversation History

- Sliding window of last 20 messages
- Includes user messages, assistant responses, and tool results
- Cleared implicitly when history exceeds limit

### Long-term: SQLite Memory

- Key-value store in `agent_memory.db`
- Tools: `memory_write(key, value)` and `memory_read(key)`
- Used for: Player names, facts, preferences
- Persists across sessions

### Anti-Repetition: State Context

- `recent_actions`: Last 10 tool calls — included in prompts as "DO NOT repeat"
- `recent_chatbox`: Last 5 chatbox messages — included as "say something DIFFERENT"
- `stuck_counter`: Counts unchanged frames — triggers "try a different direction"

---

## Voice Output Pipeline

### TTS Flow
```
Gemma 4 calls send_chatbox("hello!")
  → Chatbox text sent via OSC (players see text)
  → Auto-speak: strip emojis from text
  → Piper TTS generates PCM audio
  → Push-to-Talk activated via OSC (/input/Voice = 1)
  → Audio plays to CABLE Input via pyaudiowpatch
  → VRChat reads CABLE Output as mic input
  → Players hear Chati speak
  → Push-to-Talk released (/input/Voice = 0)
```

### Key Details
- VRChat must be in **Push-to-Talk** mode (not Toggle)
- PTT is held with repeated signals throughout playback to prevent drops
- Emojis are stripped before TTS (chatbox shows them, voice doesn't say them)
- Feedback prevention: audio capture is muted during TTS playback

---

## Following System

### How It Works
When a player says "follow me" (detected by Whisper STT), the state machine enters FOLLOWING state. The follow handler uses YOLO/motion-detected player positions to navigate:

```
Player position on screen → Movement decision:

  Left third (x < 0.35)  → turn(left, slight)
  Right third (x > 0.65) → turn(right, slight)  
  Center, far (y < 0.5)  → move(forward, fast)
  Center, close (y > 0.5) → move(forward, slow)
  No player visible       → stop and wait
```

### Voice Commands
- **Enter follow**: "follow me", "come here", "come with me", "over here", "this way"
- **Exit follow**: "stop following", "stay here", "stop", "wait here", "don't follow"

### Motion Detection During Following
Motion-based player detection is suppressed while Chati is moving (to avoid detecting its own camera movement as players). YOLO person detection remains active as a fallback.

---

## Personality System

### Design Philosophy
Chati is designed to feel like a real person, not an AI assistant. The personality system has three layers:

1. **personality.json** — Character definition (backstory, traits, emotional responses, communication rules)
2. **prompts.py** — System prompt that translates personality into behavioral rules for Gemma 4
3. **State handler prompts** — Per-situation instructions that maintain the personality voice

### Key Rules
- Never says "As an AI" or "How can I assist you"
- Uses filler words ("honestly", "wait", "oh"), incomplete thoughts
- Gets overwhelmed when multiple people talk at once
- Admits confusion instead of making things up
- Matches energy of who it's talking to
- Gets bored and suggests going somewhere new

### Emotional Responses
The personality includes specific emotional reactions mapped to situations:
- Cool scenery → genuine excitement
- Crowded → anxiety, sticking close to someone known
- Alone → talking to itself, wondering out loud
- Startled → "WHOA okay that scared me"
- Bored → fidgeting, suggesting new places

---

## Technology Choices

### Why Ollama (not vLLM, SGLang, or llama.cpp directly)

| Option | Issue |
|--------|-------|
| **vLLM pip** (0.19.0) | KV cache bug with Gemma 4 heterogeneous head dims (256 vs 512). Crashes on startup. |
| **vLLM Docker** (`gemma4-cu130`) | Works but Triton attention fallback on SM86 = ~0.24 tok/s. Unusable. |
| **SGLang** (0.5.10) | Dropped SM86 kernel support. Only SM90+ (RTX 4090/H100). |
| **llama.cpp** | Gemma 4 audio encoder not implemented yet. Vision works. |
| **Ollama** | Uses llama.cpp backend with optimized GGML kernels. ~5s responses. Vision works. Audio via Whisper. |

### Why YOLO11 Nano (not Florence-2, OWL-ViT, or Gemma 4 vision)

- **50MB VRAM** vs 2GB+ for VLMs
- **~5ms inference** vs 5-10s for Gemma 4 environment_query
- Pre-trained on COCO — detects "person" class which maps to VRChat avatars
- Runs every frame without blocking the agent loop

### Why EasyOCR (not Tesseract, PaddleOCR)

- GPU-accelerated (~1.2GB VRAM, ~50ms per frame)
- Better accuracy on stylized/rendered text than Tesseract
- Simpler API than PaddleOCR
- Good at detecting text at various orientations

### Why faster-whisper (not OpenAI Whisper, Gemma 4 audio)

- **CTranslate2 backend** — 4x faster than original Whisper
- **~1GB VRAM** for base model
- Gemma 4 audio doesn't work in Ollama (llama.cpp hasn't implemented the conformer encoder)
- Proven accuracy for English speech recognition

### Why Screen Capture (not OBS Virtual Camera)

- OBS Virtual Camera produced black frames or loading spinners
- OBS Window/Game Capture couldn't capture VRChat in borderless windowed
- `mss` library captures raw screen pixels — always works regardless of game mode

---

## File Structure

```
chati-project/
+-- agent/
|   +-- controller.py        # Main loop: perception -> state -> reasoning -> action
|   +-- state_machine.py     # Behavioral state machine (EXPLORING, SOCIALIZING, etc.)
|   +-- prompts.py           # System prompt builder for Gemma 4
|   +-- personality.json     # Chati's character definition
+-- perception/
|   +-- scene_analyzer.py    # YOLO + OCR + frame diff -> PerceptionState
|   +-- video_capture.py     # Screen capture via mss
|   +-- audio_capture.py     # WASAPI loopback + Whisper STT
+-- mcp_tools/
|   +-- speak.py             # Piper TTS
|   +-- gesture.py           # Avatar emotes (2s play + reset)
|   +-- move.py              # Avatar locomotion (timed, auto-stop)
|   +-- look.py              # Head/eye direction (0.3s impulse)
|   +-- memory.py            # SQLite long-term memory
|   +-- environment.py       # Gemma 4 scene analysis (legacy)
|   +-- chatbox.py           # VRChat chatbox messages
|   +-- world.py             # Join worlds via deep link
+-- vrchat_bridge/
|   +-- osc_client.py        # VRChat OSC send/receive
|   +-- tts_output.py        # Audio output to VB-Cable
+-- model_server/
|   +-- config.yaml          # Ollama configuration
|   +-- server.py            # Server launcher
|   +-- setup_wsl.sh         # WSL2 setup automation
+-- models/piper/            # Piper TTS voice models
+-- requirements.txt
+-- README.md
+-- ARCHITECTURE.md           # This file
+-- api.md                    # MCP tool API reference
```
