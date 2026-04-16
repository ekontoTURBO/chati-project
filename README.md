# Chati — VRChat AI Companion

A local, real-time AI companion for VRChat. Sees, hears, speaks, moves, and remembers — all on a single RTX 3090.

Chati is **not an AI assistant** and doesn't pretend to be one. The personality is "something older than human, not an AI — bound inside VRChat." Quietly curious, a little wistful, sometimes wry. It watches, joins in when it wants to, and remembers what you've told it.

> Full technical deep-dive: [ARCHITECTURE.md](ARCHITECTURE.md) · Tool API: [api.md](api.md)

## What it does

- **Sees** — screen capture (mss) → YOLO11 person detection + MOG2 motion detection + EasyOCR chatbox reading + frame-diff change detection.
- **Hears** — WASAPI loopback of your speakers → faster-whisper STT with VAD, silence-gap detection, and cooldowns.
- **Speaks** — Piper TTS → VB-Audio Cable → VRChat mic, with auto push-to-talk via OSC. Emoji stripped before synthesis.
- **Acts** — OSC movement, head turning, gestures, chatbox, short- and long-term memory.
- **Interrupts** — if you speak while Chati is talking, TTS stops mid-sentence, the in-flight LLM call is cancelled, and the next response includes both of your utterances.

## Architecture at a glance

```
+-----------------------------------------------------------+
|  PERCEPTION (background threads)                          |
|                                                           |
|  mss screen cap (3 FPS) --+                               |
|                            +-> scene_analyzer             |
|  YOLO11 nano --------------+   (players/text/motion)      |
|  MOG2 motion --------------+                              |
|  EasyOCR ------------------+                              |
|  frame diff ---------------+                              |
|                                                           |
|  WASAPI loopback -> faster-whisper -> transcription       |
+-----------------------------------------------------------+
                          |
                          v
+-----------------------------------------------------------+
|  SignalsHub.snapshot()  (one read per tick)               |
|    perception + latest_speech + tts_playing + frame       |
+-----------------------------------------------------------+
                          |
                          v
+-----------------------------------------------------------+
|  AgentController._tick  (1 Hz)                            |
|                                                           |
|  1. snapshot                                              |
|  2. new speech? -> cancel TTS + cancel gen_task           |
|  3. state_machine.update (hysteresis)                     |
|  4. spatial cue? -> immediate turn                        |
|  5. should_act? -> async _act (cancellable)               |
+-----------------------------------------------------------+
                          |
                          v
+-----------------------------------------------------------+
|  _act  ->  Gemma 4 E4B (AsyncOpenAI -> Ollama)            |
|    system prompt + dialogue + scene + tools               |
|    parse tool_calls + pseudo-calls (Gemma text format)    |
|    dispatch to handlers                                    |
+-----------------------------------------------------------+
                          |
                          v
+-----------------------------------------------------------+
|  ACTION LAYER                                             |
|  send_chatbox (text + TTS)   gesture   move / turn        |
|  look_at (horizontal)        jump      memory read/write  |
+-----------------------------------------------------------+
```

## State machine (3 states + hysteresis)

| State | Entry | Exit |
|-------|-------|------|
| **SOLO** | start, or long absence | player seen ≥3 ticks OR speech heard → ENGAGED |
| **ENGAGED** | sustained player presence or conversation | 20 s silence AND 8 s no player → SOLO |
| **FOLLOWING** | player says "follow me" | "stop following" OR 30 s no sighting → SOLO |

No more flickering between states on single-frame detections. Old `APPROACHING` / `SOCIALIZING` / `CONVERSING` collapsed into `ENGAGED`.

## Interruption model

1. Player speech arrives mid-response.
2. `TTSAudioRouter.cancel()` fires — play loop drains its queue, a `threading.Event` breaks chunk playback.
3. `_gen_id += 1` invalidates the in-flight response; `_gen_task.cancel()` kills the HTTP request to Ollama.
4. The utterance is appended to the dialogue buffer *before* the new generation starts, so Chati's next reply sees the interruption in context.

Late-completing responses check `gen_id` and discard themselves.

## Requirements

- **GPU** NVIDIA RTX 3090 (24 GB) or better
- **OS** Windows 11 (WSL2 for Ollama)
- **VRChat** OSC enabled, voice set to Push-to-Talk
- **VB-Audio Cable** https://vb-audio.com/Cable/ (virtual mic for TTS → VRChat)
- **Piper TTS voice** `en_US-amy-medium.onnx` + `.json` in `models/piper/`

## Audio routing

| Path | Route |
|------|-------|
| Chati → players | Piper TTS → CABLE Input → VRChat mic (auto-PTT via OSC) |
| Players → Chati | VRChat → Speakers (Realtek) → WASAPI loopback → Whisper |
| You → everything | VRChat output → Speakers (Realtek) → headphones |

VRChat settings:
- Output: **Speakers (Realtek)**
- Input: **CABLE Output (VB-Audio Virtual Cable)**
- Voice mode: **Push-to-Talk**

## Install

```powershell
cd chati-project
pip install -r requirements.txt
```

Ollama in WSL2:
```bash
cd /mnt/c/Users/YOUR_USERNAME/Desktop/chati-project/model_server
bash setup_wsl.sh
ollama pull gemma4:e4b
```

Piper voice:
```powershell
mkdir models\piper
# Drop en_US-amy-medium.onnx + .json into models/piper/
```

## Run

WSL2 terminal:
```bash
ollama serve
```

Windows:
```powershell
python -m agent.controller
```

## Voice shortcuts (no LLM round-trip)

| Say | Effect |
|-----|--------|
| "follow me" / "come here" / "come with me" | Force FOLLOWING state |
| "stop following" / "stay here" | Exit FOLLOWING |
| "behind you" / "turn around" | Immediate half-turn before LLM reacts |
| "over here" / "look at me" / "look here" | Spatial cue logged |

## Project structure

```
chati-project/
├── agent/
│   ├── controller.py       # async main loop, interruptible _act, pseudo-call parser
│   ├── signals.py          # SignalsHub — one snapshot per tick
│   ├── state_machine.py    # 3-state FSM with hysteresis
│   ├── dialogue.py         # rolling turn buffer (player/chati, timestamped)
│   ├── prompts.py          # minimal prompt builder, cache-friendly
│   └── personality.json    # identity rules, traits, never_say list
├── perception/
│   ├── scene_analyzer.py   # YOLO + MOG2 + EasyOCR + frame diff → PerceptionState
│   ├── video_capture.py    # mss screen grab
│   └── audio_capture.py    # WASAPI loopback + faster-whisper + VAD
├── mcp_tools/
│   ├── chatbox.py          # /chatbox/input — also triggers TTS
│   ├── gesture.py          # VRCEmote (wave, dance, cheer, …)
│   ├── move.py             # /input/Vertical + /input/Horizontal (timed hold)
│   ├── look.py             # horizontal-only turn + look_at
│   ├── memory.py           # SQLite long-term facts
│   └── speak.py            # Piper TTS primitive (chatbox is the usual path)
├── vrchat_bridge/
│   ├── osc_client.py       # send/receive + push-to-talk hold
│   └── tts_output.py       # pyaudiowpatch out to CABLE Input, cancellable
├── model_server/
│   └── setup_wsl.sh        # Ollama bootstrap for WSL2
├── models/piper/           # TTS voice files (download separately)
└── agent_memory.db         # SQLite memory store (created on first run)
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Chati silent | VRChat input → CABLE Output; voice mode → Push-to-Talk |
| Chati hears nothing | VRChat output → Speakers (Realtek); check loopback device name matches `AudioCaptureProcessor` |
| Screen capture black | Run VRChat in Borderless Windowed |
| Ollama unreachable | `ollama serve` in WSL; confirm port 11434 |
| Head drifts down | Fixed — vertical look targets removed (VRChat LookVertical is velocity, not position) |
| Random VRChat window launched | Fixed — `join_world` tool removed from LLM exposure |
| Chati reads tool syntax out loud | Fixed — pseudo-call parser strips `send_chatbox{text:"…"}` before TTS |
| Chati says "it's quiet in here" | Added to `never_say` in personality.json |

## Known limitations

- `move()` uses blocking `time.sleep(duration)` — stalls perception ticks for up to 10 s. Barge-in can't fire during movement.
- YOLO is trained on COCO, not VRChat avatars. Motion detection is the backstop for non-human avatars.
- Gemma 4 E4B in Ollama emits tool calls as text content instead of proper `tool_calls`; parsed via regex. If Ollama ever fixes this, the parser becomes dead code.
- Head rotation is horizontal-only. VRChat's `LookVertical` is a velocity input, so any non-zero vertical leaves permanent pitch drift.

## License

MIT
