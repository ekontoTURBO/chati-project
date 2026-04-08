# API Documentation — MCP Tools

All tools are exposed as function calls to the Gemma 4 model via Ollama's OpenAI-compatible function-calling API. The agent controller handles dispatching tool calls to the appropriate handler.

---

## speak

**Description:** Speak text aloud in VRChat using Piper TTS routed through VB-Audio Cable 16ch.

**Parameters:**
```json
{
  "text": "string — The text to speak aloud"
}
```

**Success Response:**
```json
{
  "success": true,
  "text": "Hello, nice to meet you!"
}
```

**Error Codes:**
- `error: "Empty text provided"` — Text parameter was empty
- `error: "piper-tts not installed..."` — Piper TTS dependency missing
- `error: "Piper voice model not found..."` — ONNX model file not in models/piper/

---

## gesture

**Description:** Trigger an avatar emote/gesture in VRChat via OSC. The emote plays for 2 seconds then resets to idle.

**Parameters:**
```json
{
  "type": "string — Gesture name (wave|clap|point|cheer|dance|backflip|die|sadness|sad|happy|thumbsup|bow|reset)"
}
```

**Success Response:**
```json
{
  "success": true,
  "gesture": "wave",
  "emote_id": 1
}
```

**Error Codes:**
- `error: "Unknown gesture: '...'"` — Invalid gesture name provided

---

## move

**Description:** Move the avatar in a specified direction in VRChat. Movement lasts for the given duration then stops automatically.

**Parameters:**
```json
{
  "direction": "string — Direction (forward|backward|back|left|right|forward_left|forward_right|stop)",
  "speed": "number — Speed 0.0-1.0 (default: 1.0, optional)",
  "duration": "number — Seconds to move (default: 1.0, optional)"
}
```

**Success Response:**
```json
{
  "success": true,
  "direction": "forward",
  "speed": 1.0,
  "duration": 1.0
}
```

**Error Codes:**
- `error: "Unknown direction: '...'"` — Invalid direction name

---

## jump

**Description:** Make the avatar jump.

**Parameters:**
```json
{}
```

**Success Response:**
```json
{
  "success": true,
  "action": "jump"
}
```

---

## look_at

**Description:** Control avatar head/eye look direction. Sends a brief impulse (0.3s) then resets to center to prevent continuous rotation.

**Parameters:**
```json
{
  "target": "string — Direction (left|right|up|down|up_left|up_right|down_left|down_right|center|reset)"
}
```

**Success Response:**
```json
{
  "success": true,
  "target": "left"
}
```

**Error Codes:**
- `error: "Unknown look target: '...'"` — Invalid target name

---

## memory_write

**Description:** Store information in long-term SQLite memory.

**Parameters:**
```json
{
  "key": "string — Memory key (e.g., 'player_alice_name')",
  "value": "string — Value to remember"
}
```

**Success Response:**
```json
{
  "success": true,
  "key": "player_alice_name",
  "value": "Alice"
}
```

**Error Codes:**
- `error: "Empty key"` — Key parameter was empty

---

## memory_read

**Description:** Recall information from long-term memory.

**Parameters:**
```json
{
  "key": "string — Memory key to look up"
}
```

**Success Response (found):**
```json
{
  "success": true,
  "found": true,
  "key": "player_alice_name",
  "value": "Alice",
  "category": "general"
}
```

**Success Response (not found):**
```json
{
  "success": true,
  "found": false,
  "key": "unknown_key"
}
```

---

## environment_query

**Description:** Analyze the current VRChat environment by sending the latest screen capture frame to Gemma 4 for visual analysis. Results are cached for 2 seconds.

**Parameters:**
```json
{}
```

**Success Response:**
```json
{
  "success": true,
  "scene": "A colorful Japanese garden with cherry blossom trees",
  "players_visible": 2,
  "players": ["Player in blue avatar near fountain", "Player dancing"],
  "objects": ["fountain", "cherry trees", "lanterns"],
  "mood": "peaceful and lively",
  "activity": "players socializing and dancing"
}
```

**Error Codes:**
- `error: "No video frame available..."` — Screen capture not running
- `error: "..."` — Ollama API call failure

---

## send_chatbox

**Description:** Send a text message to VRChat's in-game chatbox display (appears above avatar's head).

**Parameters:**
```json
{
  "text": "string — Message to display (max 144 characters)"
}
```

**Success Response:**
```json
{
  "success": true,
  "text": "Hey there! Nice world!"
}
```

**Error Codes:**
- `error: "Empty text"` — Text parameter was empty

---

## join_world

**Description:** Join a VRChat world using the vrchat:// deep link protocol. VRChat must already be running.

**Parameters:**
```json
{
  "world_id": "string — VRChat world ID in wrld_xxxx format (optional)",
  "world_name": "string — Known world name (optional)"
}
```

**Known worlds:** the_black_cat, just_b_club, movie_and_chill, the_great_pug, midnight_rooftop, home

**Success Response:**
```json
{
  "success": true,
  "world_id": "wrld_4cf554b4-430c-4f8f-b53e-1f294eed230b",
  "world_name": "the_black_cat"
}
```

**Error Codes:**
- `error: "Unknown world name: '...'"` — World name not in known list
- `error: "Provide either world_id or world_name"` — Neither parameter provided
