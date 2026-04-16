# API — MCP Tools

All tools are exposed as OpenAI-style function calls to Gemma 4 E4B via Ollama. The controller dispatches them through `_tool_handlers` in `agent/controller.py`.

**Removed from the LLM exposure:**
- `join_world` — model autonomously launched VRChat windows unprompted. File `mcp_tools/world.py` stays on disk but isn't imported.
- `environment_query` — never wired into the single-path controller (legacy from the old 3-tier system).

**Notable quirks:**
- Gemma 4 E4B often emits tool calls as text content instead of `tool_calls`. The controller has a regex parser (`_extract_pseudo_calls`) that catches `name{key:"value"}` and `name{key:<|"|>value<|"|>}` forms.
- `send_chatbox` is the primary "talk" path — it also triggers TTS via the controller. `speak` exists as a standalone primitive but the LLM is told to use `send_chatbox`.

---

## send_chatbox

Primary speech + text path. The text appears over the avatar's head **and** is spoken aloud via Piper TTS (emoji stripped first).

**Parameters**
```json
{
  "text": "string — message to display (≤144 chars; truncated at sentence boundary + '...' if longer)"
}
```

**Success**
```json
{ "success": true, "text": "Hey there! Nice world!" }
```

**Errors**
- `"Empty text"` — text was empty/whitespace.

---

## speak

Standalone Piper TTS primitive. The LLM is instructed to prefer `send_chatbox` instead (which combines chatbox + TTS).

**Parameters**
```json
{ "text": "string — text to speak aloud" }
```

**Errors**
- `"Empty text provided"`
- `"piper-tts not installed..."`
- `"Piper voice model not found..."` — ONNX missing from `models/piper/`.

---

## gesture

Avatar emote via OSC. Plays for ~2 s then auto-resets to idle.

**Parameters**
```json
{
  "type": "string — wave|clap|point|cheer|dance|backflip|die|sadness|sad|happy|thumbsup|bow|reset"
}
```

Mapping of friendly aliases to VRChat emote IDs:

| Name | ID | Name | ID |
|------|----|------|----|
| wave | 1 | dance | 5 |
| clap | 2 | backflip | 6 |
| point / thumbsup | 3 | die / bow | 7 |
| cheer / happy | 4 | sadness / sad | 8 |
| reset | 0 | | |

**Success**
```json
{ "success": true, "gesture": "wave", "emote_id": 1 }
```

**Errors**
- `"Unknown gesture: '...'"`

---

## move

Walks the avatar in a direction for `duration` seconds, then stops. **Blocking — uses `time.sleep(duration)` inside the async controller.** Barge-in cannot fire during a move.

**Parameters**
```json
{
  "direction": "string — forward|backward|back|left|right|forward_left|forward_right|stop",
  "speed":     "number — 0.0..1.0, default 1.0 (>0.7 auto-enables run)",
  "duration":  "number — seconds, clamped 0.1..10.0, default 1.0"
}
```

Directions are avatar-local, not world-local — `forward` means whichever way the head currently points.

**Success**
```json
{ "success": true, "direction": "forward", "speed": 1.0, "duration": 1.0 }
```

**Errors**
- `"Unknown direction: '...'"`

---

## turn

Rotate the avatar's body in place. Yaw-only, uses `/input/LookHorizontal` impulse.

**Parameters**
```json
{
  "direction": "string — left|right",
  "amount":    "string — slight (≈45°) | quarter (≈90°) | half (≈180°), default quarter"
}
```

Durations: slight 0.3 s, quarter 0.6 s, half 1.2 s.

**Success**
```json
{ "success": true, "direction": "left", "amount": "quarter" }
```

---

## look_at

Head direction — **horizontal only**. Vertical targets were removed because VRChat's `/input/LookVertical` is a velocity axis (not a position), so any non-zero vertical left permanent pitch drift.

**Parameters**
```json
{ "target": "string — left|right|center|reset" }
```

Sends a 0.3 s impulse then zeros the axis. `center` and `reset` both send `(0, 0)`.

**Success**
```json
{ "success": true, "target": "left" }
```

**Errors**
- `"Unknown look target: '...'. Available: center, left, reset, right"`

---

## jump

**Parameters**
```json
{}
```

**Success**
```json
{ "success": true, "action": "jump" }
```

---

## memory_write

SQLite long-term store. Persists across sessions. Use for player names, recurring facts.

**Parameters**
```json
{
  "key":   "string — e.g. 'player_alice_name'",
  "value": "string — value to remember"
}
```

**Success**
```json
{ "success": true, "key": "player_alice_name", "value": "Alice" }
```

**Errors**
- `"Empty key"`

---

## memory_read

**Parameters**
```json
{ "key": "string — key to look up" }
```

**Success (found)**
```json
{ "success": true, "found": true, "key": "player_alice_name", "value": "Alice", "category": "general" }
```

**Success (not found)**
```json
{ "success": true, "found": false, "key": "unknown_key" }
```

---

## Out-of-band controls (no tool — handled in `_tick` before LLM)

### Spatial cues → immediate turn
| Phrase | Action |
|--------|--------|
| "behind you" / "behind me" / "turn around" | `turn(right, half)` |
| "over here" / "look at me" / "look here" | logged, no pre-emptive action |

### Follow commands → state transition only
| Phrase | Action |
|--------|--------|
| "follow me" / "come with me" / "come here" | `state_machine.force_follow()` |
| "stop following" / "stop follow" / "stay here" | `state_machine.force_stop_follow()` |

These execute before the LLM sees the utterance; Gemma still receives the text in the dialogue buffer so its reply can acknowledge the command.

---

## Interruption contract

If a new player utterance arrives while Chati is generating OR speaking:

1. `TTSAudioRouter.cancel()` — kills playback mid-chunk.
2. `self._gen_id += 1` — any in-flight response becomes stale.
3. `self._gen_task.cancel()` — cancels the `await AsyncOpenAI.chat.completions.create(...)` and propagates `CancelledError`.
4. `dialogue.add_player(speech)` — new turn appended **before** the next generation.
5. Next `_tick` schedules a fresh `_act(gen_id)`.

Late-completing responses check `gen_id != self._gen_id` and discard. The controller logs `[STALE] gen#N discarded` when this happens.
