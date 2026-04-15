# Chati — VRChat AI Agent

A local, real-time AI companion for VRChat with **real-time computer vision** (YOLO11 + OCR), **speech recognition** (Whisper), and **multimodal reasoning** (Gemma 4 E2B) — all running locally on a single RTX 3090.

Chati can see players in real-time, read chatbox messages, hear speech, speak back, move, gesture, remember facts, and explore VRChat worlds autonomously.

> For the full technical deep-dive, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Architecture

```
+================================================================+
|  PERCEPTION LAYER (background threads, GPU)                     |
|                                                                 |
|  Screen Capture -----> YOLO11 nano -----> player count,         |
|  (mss, 1 FPS)         (~5ms/frame)       positions, objects     |
|                   +--> EasyOCR ---------> chatbox text           |
|                   +--> Frame Diff ------> scene changes          |
|                                                                 |
|  Audio Capture -----> Whisper STT ------> speech transcription   |
|  (WASAPI loopback)    (faster-whisper)                          |
+================================================================+
              |                              |
              v                              v
+================================================================+
|  STATE MACHINE           |  REASONING (Gemma 4 via Ollama)      |
|                          |                                      |
|  IDLE -> EXPLORING       |  Only called for decisions:          |
|       -> APPROACHING     |  "What should I do next?"            |
|       -> SOCIALIZING     |  "Player said X, respond."           |
|       -> CONVERSING      |  NOT used for perception.            |
+================================================================+
              |
              v
+================================================================+
|  ACTION LAYER (MCP Tools -> VRChat OSC)                         |
|                                                                 |
|  speak (Piper TTS)  |  move/look/jump  |  chatbox  |  gesture  |
|  memory read/write   |  join_world      |  environment_query    |
+================================================================+
```

> See [ARCHITECTURE.md](ARCHITECTURE.md) for the full technical deep-dive including VRAM budget, state transitions, and technology choice rationale.

## How It Works

1. **Real-Time Vision** (YOLO11 + OCR): Continuously scans the screen at 1 FPS. YOLO detects players and objects in ~5ms. EasyOCR reads chatbox text above players' heads. Frame differencing detects scene changes. No LLM needed for perception.
2. **Hearing** (Whisper STT): Captures VRChat audio via WASAPI loopback, detects speech start/end, transcribes with faster-whisper on GPU.
3. **State Machine**: Drives behavior based on perception — EXPLORING when alone, APPROACHING when spotting a player, SOCIALIZING after greeting, CONVERSING when speech is detected. No random decisions.
4. **Reasoning** (Gemma 4): Only called for high-level decisions: "What should I do?" or "Player said X, respond." Gets perception data + video frame + speech, returns tool calls.
5. **Acting** (MCP Tools): Executes tool calls via VRChat OSC — move, gesture, chatbox, TTS speech, memory, world joins.

## Requirements

### Hardware
- **GPU**: NVIDIA RTX 3090 (24 GB VRAM) or better
- **OS**: Windows 11

### Software Prerequisites

1. **WSL2 (Ubuntu)**
   ```powershell
   # Run in PowerShell as Administrator
   wsl --install
   # Restart PC, then open Ubuntu from Start Menu
   ```

2. **VB-Audio Cable** (virtual microphone for TTS)
   - Download: https://vb-audio.com/Cable/
   - Install `VBCABLE_Setup_x64.exe` as **Administrator**
   - **Restart your PC** after installation
   - You also need the **VB-Audio Cable 16ch** variant for audio routing

3. **VRChat** with OSC enabled
   - In VRChat: **Action Menu -> Options -> OSC -> Enabled**

4. **Piper TTS Voice Model**
   - Download: https://github.com/rhasspy/piper/releases
   - Recommended: `en_US-lessac-medium.onnx` + `.json`
   - Place both files in: `models/piper/`

## Audio Routing

Chati uses two virtual cables to separate TTS output from VRChat audio capture:

| Route | Cable | How |
|-------|-------|-----|
| **Agent speaks -> VRChat mic** | CABLE In 16ch | TTS writes here, VRChat reads as mic |
| **VRChat audio -> Agent hears** | CABLE In 16ch (loopback) | Agent captures via WASAPI loopback |

**VRChat audio settings:**
- Output: **CABLE In 16ch** (so Chati can hear players)
- Input: **CABLE Output** (so VRChat reads TTS)

Note: TTS feedback loop is prevented by muting capture while Chati speaks.

## Installation

### Step 1: Install Python dependencies (Windows)
```powershell
cd chati-project
pip install -r requirements.txt
```

### Step 2: Set up Ollama in WSL2
```bash
# Open Ubuntu (WSL2) terminal
cd /mnt/c/Users/YOUR_USERNAME/Desktop/chati-project/model_server
bash setup_wsl.sh
```
This installs Ollama and pulls the Gemma 4 E2B model (~7 GB).

### Step 3: Download Piper voice model
```powershell
mkdir models\piper
# Download from https://github.com/rhasspy/piper/releases
# Place en_US-lessac-medium.onnx and en_US-lessac-medium.onnx.json in models/piper/
```

## Running

### 1. Start Ollama (WSL2 terminal)
```bash
ollama serve
```

### 2. Enable VRChat OSC
In VRChat: Action Menu -> Options -> OSC -> Enabled

### 3. Set VRChat audio routing
- Output device: **CABLE In 16ch**
- Input device: **CABLE Output**

### 4. Start the agent (Windows PowerShell)
```powershell
cd chati-project
python -m agent.controller
```

### Optional arguments
```
--model-url    Ollama server URL (default: http://localhost:11434/v1)
--model-name   Model tag (default: gemma4:e2b)
--camera       Unused (screen capture is automatic)
--osc-port     VRChat OSC port (default: 9000)
```

## Agent Behavior

### Explore Mode (no players nearby)
The agent looks at the scene and decides what to do:
- Move toward interesting areas (doors, objects)
- Turn around when facing walls
- Look left/right to scan the environment
- Comment on things via chatbox
- Do gestures (dance, wave, cheer)

### Social Mode (player detected)
When a player is visible, the agent:
1. Immediately stops all movement
2. Waves and sends a greeting via chatbox
3. Reads text above players' heads (chatbox messages)
4. Responds to what players say (via Whisper STT or visual chatbox)
5. Varies behavior — comments on the scene, does gestures, asks questions
6. Tracks recent actions to avoid repeating itself

### Speech Recognition
When a player speaks near Chati (audio routed via virtual cable):
1. Energy-based voice activity detection starts accumulating audio
2. After 1.5s of silence, the utterance is complete
3. faster-whisper transcribes the speech to text on GPU
4. The transcription + current video frame are sent to Gemma 4
5. Gemma 4 responds with chatbox messages, gestures, and movement

## Project Structure
```
chati-project/
+-- agent/                  # AI brain
|   +-- controller.py       # Main loop (perception -> state -> reasoning -> action)
|   +-- state_machine.py    # Behavioral states (EXPLORING, SOCIALIZING, etc.)
|   +-- personality.json    # Chati's character definition
|   +-- prompts.py          # System prompt builder for Gemma 4
+-- perception/             # Sensory input (real-time, background threads)
|   +-- scene_analyzer.py   # YOLO11 + EasyOCR + frame diff -> PerceptionState
|   +-- video_capture.py    # Screen capture via mss
|   +-- audio_capture.py    # WASAPI loopback + Whisper STT
+-- mcp_tools/              # AI agent capabilities (tool calls)
|   +-- speak.py            # Piper TTS voice output
|   +-- gesture.py          # Avatar emotes (wave, dance, etc.)
|   +-- move.py             # Avatar locomotion
|   +-- look.py             # Head/eye direction
|   +-- memory.py           # Long-term memory (SQLite)
|   +-- environment.py      # Gemma 4 deep scene analysis (legacy)
|   +-- chatbox.py          # VRChat chatbox messages
|   +-- world.py            # Join VRChat worlds via deep link
+-- vrchat_bridge/          # VRChat communication
|   +-- osc_client.py       # OSC send/receive (UDP)
|   +-- tts_output.py       # Audio output to VB-Cable
+-- model_server/           # Ollama server config (runs in WSL2)
|   +-- config.yaml         # Server parameters
|   +-- server.py           # Server launcher
|   +-- setup_wsl.sh        # WSL2 setup automation
+-- models/piper/           # Piper TTS voice models (download separately)
+-- ARCHITECTURE.md         # Full technical architecture deep-dive
+-- api.md                  # MCP tool API reference
+-- requirements.txt        # Python dependencies
+-- README.md               # This file
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Ollama OOM error | Reduce context: set `num_ctx` in Ollama modelfile |
| No audio capture | Set VRChat output to "CABLE In 16ch" in Sound settings |
| TTS not heard in VRChat | Set VRChat input to "CABLE Output" |
| VRChat OSC not working | Enable OSC in VRChat Action Menu -> Options |
| Ollama no GPU | Update Windows NVIDIA drivers to v535+ |
| WSL2 not starting | Run `wsl --install` in PowerShell as Admin |
| Screen capture black | Make sure VRChat is in **Borderless Windowed** mode |
| Whisper slow | Try `whisper_model="tiny"` in audio_capture.py for faster STT |
| Agent repeats itself | Recent actions tracker prevents loops — check logs |

## License
MIT
