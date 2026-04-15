# Chati — VRChat AI Agent

A local, real-time AI companion for VRChat with **real-time computer vision** (YOLO11 + motion detection + OCR), **speech recognition** (Whisper), **voice output** (Piper TTS), and **multimodal reasoning** (Gemma 4 E2B) — all running locally on a single RTX 3090.

Chati can see players (any avatar type), read chatbox messages, hear and understand speech, speak back with voice, move, gesture, follow players, remember facts, and explore VRChat worlds autonomously. It has a human-like personality — gets excited about cool things, overwhelmed in crowds, and bored when alone.

> For the full technical deep-dive, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Architecture

```
+================================================================+
|  PERCEPTION LAYER (background threads, 3 FPS)                   |
|                                                                 |
|  Screen Capture -----> YOLO11 nano -----> object detection      |
|  (mss, 3 FPS)    +--> Motion Detection -> player detection      |
|                   |    (MOG2 bg sub)      (any avatar type)     |
|                   +--> EasyOCR ---------> chatbox text           |
|                   +--> Frame Diff ------> scene changes          |
|                                                                 |
|  Audio Capture -----> Whisper STT ------> speech transcription   |
|  (Realtek loopback)   (faster-whisper)                          |
+================================================================+
              |                              |
              v                              v
+================================================================+
|  STATE MACHINE           |  REASONING (Gemma 4 via Ollama)      |
|                          |                                      |
|  IDLE -> EXPLORING       |  Human-like personality              |
|       -> APPROACHING     |  Emotional reactions                 |
|       -> SOCIALIZING     |  Context-aware decisions             |
|       -> CONVERSING      |  Voice commands (follow me, stop)    |
|       -> FOLLOWING       |                                      |
+================================================================+
              |
              v
+================================================================+
|  ACTION LAYER (MCP Tools -> VRChat OSC)                         |
|                                                                 |
|  speak (Piper TTS + PTT) | move/turn/look | chatbox | gesture  |
|  memory read/write       | join_world     | follow player       |
+================================================================+
```

## How It Works

1. **Real-Time Vision** (3 FPS): YOLO11 detects objects, motion detection finds player avatars of ANY type (anime, furry, robot — anything that moves), EasyOCR reads chatbox text, frame differencing detects scene changes.
2. **Hearing** (Whisper STT): Captures audio from your speakers via WASAPI loopback, detects speech boundaries, transcribes with faster-whisper. Recognizes voice commands like "follow me" and "stop."
3. **Voice Output** (Piper TTS): Generates speech, plays to VB-Audio Cable with automatic push-to-talk via OSC. Players hear Chati speak.
4. **State Machine**: Drives behavior — EXPLORING when alone, APPROACHING when spotting someone, SOCIALIZING after greeting, CONVERSING when speech is detected, FOLLOWING when asked.
5. **Personality**: Human-like reactions — gets excited about cool things, overwhelmed in crowds, bored when alone, admits confusion. Never sounds like a customer service bot.

## Requirements

### Hardware
- **GPU**: NVIDIA RTX 3090 (24 GB VRAM) or better
- **OS**: Windows 11

### Software Prerequisites

1. **WSL2 (Ubuntu)**
   ```powershell
   wsl --install
   ```

2. **VB-Audio Cable** (virtual microphone for TTS → VRChat)
   - Download: https://vb-audio.com/Cable/
   - Install as Administrator, restart PC

3. **VRChat** with OSC enabled
   - Action Menu → Options → OSC → Enabled
   - Voice mode: **Push-to-Talk**

4. **Piper TTS Voice Model**
   - Download `en_US-lessac-medium.onnx` + `.json` from https://github.com/rhasspy/piper/releases
   - Place in `models/piper/`

## Audio Routing

| Route | How |
|-------|-----|
| **Chati speaks → players hear** | Piper TTS → CABLE Input → CABLE Output → VRChat mic (with auto PTT) |
| **Players speak → Chati hears** | VRChat → Speakers (Realtek) → WASAPI loopback → Whisper STT |
| **You hear everything** | VRChat output → Speakers (Realtek) → your headphones |

**VRChat audio settings:**
- Output: **Speakers (Realtek)** (you hear VRChat normally)
- Input: **CABLE Output (VB-Audio Virtual Cable)** (VRChat reads TTS)
- Voice: **Push-to-Talk** (agent controls PTT via OSC)

## Installation

### Step 1: Python dependencies (Windows)
```powershell
cd chati-project
pip install -r requirements.txt
```

### Step 2: Ollama in WSL2
```bash
cd /mnt/c/Users/YOUR_USERNAME/Desktop/chati-project/model_server
bash setup_wsl.sh
```

### Step 3: Piper voice model
```powershell
mkdir models\piper
# Download en_US-lessac-medium.onnx + .json from Piper releases
# Place both in models/piper/
```

## Running

### 1. Start Ollama (WSL2 terminal)
```bash
ollama serve
```

### 2. Start the agent (Windows PowerShell)
```powershell
cd chati-project
python -m agent.controller
```

### Voice Commands
- **"follow me"** / **"come here"** / **"this way"** — Chati follows you
- **"stop"** / **"stay here"** / **"stop following"** — Chati stops following

## Agent States

| State | Trigger | Behavior |
|-------|---------|----------|
| EXPLORING | No players nearby | Wanders, turns, comments on scenery |
| APPROACHING | Player detected | Stops, waves, greets with chatbox + voice |
| SOCIALIZING | After greeting | Hangs out, reacts to what's happening |
| CONVERSING | Speech detected | Responds to what player said |
| FOLLOWING | "Follow me" command | Tracks player position, follows movement |

## Project Structure
```
chati-project/
├── agent/                  # AI brain
│   ├── controller.py       # Main loop (perception → state → reasoning → action)
│   ├── state_machine.py    # Behavioral states (EXPLORING, FOLLOWING, etc.)
│   ├── personality.json    # Chati's character — emotions, reactions, style
│   └── prompts.py          # System prompt builder
├── perception/             # Sensory input (real-time, background threads)
│   ├── scene_analyzer.py   # YOLO11 + motion detection + EasyOCR
│   ├── video_capture.py    # Screen capture via mss (3 FPS)
│   └── audio_capture.py    # WASAPI loopback + Whisper STT
├── mcp_tools/              # Agent capabilities (tool calls)
│   ├── speak.py            # Piper TTS voice output
│   ├── gesture.py          # Avatar emotes (wave, dance, etc.)
│   ├── move.py             # Avatar locomotion
│   ├── look.py             # Head direction + body rotation (turn)
│   ├── memory.py           # Long-term memory (SQLite)
│   ├── environment.py      # Gemma 4 deep scene analysis
│   ├── chatbox.py          # VRChat chatbox (auto-speaks messages)
│   └── world.py            # Join VRChat worlds via deep link
├── vrchat_bridge/          # VRChat communication
│   ├── osc_client.py       # OSC send/receive + push-to-talk
│   └── tts_output.py       # Audio output to VB-Cable with PTT
├── model_server/           # Ollama server config (WSL2)
│   ├── config.yaml         # Server parameters
│   ├── server.py           # Server launcher
│   └── setup_wsl.sh        # WSL2 setup automation
├── models/piper/           # TTS voice models (download separately)
├── ARCHITECTURE.md         # Full technical architecture deep-dive
├── api.md                  # MCP tool API reference
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Can't hear Chati speak | VRChat input → CABLE Output, voice mode → Push-to-Talk |
| Chati doesn't hear players | VRChat output → Speakers (Realtek) |
| cuDNN error on startup | Harmless warning — YOLO/OCR run on CPU |
| Screen capture black | VRChat must be in Borderless Windowed mode |
| Ollama not connecting | Run `ollama serve` in WSL terminal first |
| Chati detects too many players | Motion detection adapts over time, give it ~10s |
| Whisper transcribes noise | Increase SILENCE_THRESHOLD in audio_capture.py |
| Chati repeats itself | Recent actions/chatbox tracking prevents loops |

## License
MIT
