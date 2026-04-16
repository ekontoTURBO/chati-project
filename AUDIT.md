# Chati Architectural Audit

**Date:** 2026-04-16
**Reviewer:** Code review pass over `agent/`, `perception/`, `mcp_tools/`
**Scope:** Address the 5 pain points (dumb convo, state flipping, latency, audio in lobbies, over-engineering).

---

## TL;DR — The Honest Diagnosis

Chati "feels dumb" for three concrete, fixable reasons, in order of impact:

1. **There is no actual conversation memory.** `recent_chatbox` is Chati's own prior messages — not what the *player* said. The LLM never sees dialogue turns as dialogue. Every response is effectively the first turn of a conversation.
2. **The system prompt has a corruption bug** that feeds the model gibberish. `personality.json` stores `never_say` as a **string**, but `prompts.py:44` does `", ".join(style.get("never_say", []))`, which iterates characters. The model currently sees: `NEVER say things like: I, ', m,  , j, u, s, t, ...`. That alone is degrading personality adherence.
3. **The 3-tier router bypasses history.** Tier 2 (`_tier2_respond`) and Tier 1 (intents) don't touch `self._history` at all. Tier 3 does. So whether the model has "memory" depends on which tier fires — non-deterministic from the user's perspective.

Everything else — state flipping, latency, lobby noise — is either caused or amplified by these three.

---

## 1. What to REMOVE — Be brutal

### 1.1 Remove Tier 1 intent matcher (`agent/intent_matcher.py`)
**Remove it entirely.** 186 lines of regex that "saves" ~400ms on a path that fires maybe 5% of the time, at the cost of:
- Bypasses conversation history (so follow-up turns lose context).
- Fires a canned `"okay, coming!"` reply that feels robotic — exactly the thing you're complaining about.
- Duplicates state-transition logic already in `state_machine.py` lines 159–170.
- Makes the 3-tier router brittle: what qualifies as "intent match" is a moving target.

If you must keep direct commands, keep only `stop` (safety-critical) and inline it as a 6-line shortcut in `controller.py` before the LLM call. Don't build a subsystem for it.

### 1.2 Remove `SpatialMemory` (`perception/spatial_memory.py`)
144 lines of rolling deques that produce one or two lines of prose like *"Player was on your left ~3s ago"*. The LLM can't act on stale coordinates, and when the player is currently visible, `perception.for_prompt()` already says so more accurately. Kill it.

Keep one thing from it: the **immediate stereo-pan hint for the utterance you just heard** ("voice came from your left"). That's useful. Move it to a two-line formatter inside `controller.py`.

### 1.3 Remove stereo DOA (`audio_capture.py:232-250`)
Speakers loopback is a mono-ish stereo mix of the full world soundscape, not a head-related stereo source. The "direction" of a voice arriving through loopback is physics-free — it's whichever channel VRChat panned into. You're computing RMS over a whole utterance and treating it as azimuth; it's noise with a confidence label.

It's also why (5) audio fails in crowded lobbies — you're mixing every voice's pan together.

**Replace** with loudness-gated VAD only. Direction should come from **screen position of the speaker** (YOLO/motion bbox at the moment of speech onset), not audio.

### 1.4 Remove `EXPLORING`/`APPROACHING`/`SOCIALIZING` as distinct states
These three states all produce the same behavior pattern (take a picture, ask Gemma what to do) but with different prompts. They're prompt variations dressed up as a state machine. See section 6 for the replacement.

### 1.5 Remove `FastModel` (Tier 2) for now
It's probed optimistically, its output is not integrated into the main history, and if Qwen returns a tool call the controller calls `_execute_tool_call` with an **object from the Qwen response** but Qwen's tool call may not have the same schema as OpenAI's (`tool_call.function.name` assumption). It's a latent bug.

You'll get a bigger latency win from shrinking the Gemma prompt (section 7) than from adding a second model.

### 1.6 Remove `environment_query` tool (`mcp_tools/environment.py`)
Gemma already has vision on every call. Letting it call a tool that sends *another* image to itself is redundant, doubles tokens, and is a round-trip you pay for. The model rarely calls it usefully.

### 1.7 Remove the auto-speak from `send_chatbox`
`mcp_tools/chatbox.py:48-61`. The tool silently calls `self._speak_tool.speak(clean)`. This is a hidden side effect — `speak` and `send_chatbox` are coupled. Worse, in `controller.py:449-456` you *also* call `self.speak_tool.speak(text)` **and** `self.chatbox_tool.send_chatbox(text)` when the model returns plain content. So when the model returns chat-only text you speak once, but when it returns a `send_chatbox` tool call you also speak once. Fine. But if the model *also* emits text content + a tool call, you speak twice. Remove the speak-from-chatbox coupling; make the controller the single owner of "say this out loud".

---

## 2. What to SIMPLIFY

### 2.1 The `controller._main_loop` branching (`controller.py:266-314`)
Today the loop has two paths: speech → `_handle_conversation` (3-tier routing), no-speech → `_handle_state` (per-state prompts). Each of those further branches. Collapse to:

```python
while running:
    perception = scene.get_state()
    speech = audio.get_speech()
    trigger = self._should_act(perception, speech)   # -> reason code or None
    if trigger:
        await self._act(trigger, perception, speech)
    await asyncio.sleep(1.0)
```

Where `_should_act` returns one of: `"spoken_to"`, `"player_arrived"`, `"idle_tick"`, `"stuck"`, `None`. **One** `_act` function that builds one prompt with a reason, not 5 different `_act_*` handlers with 5 different prompts.

### 2.2 Collapse `_send_to_model` / `_process_interaction` / `_tier3_respond`
All three build a content-parts array with an image + text. They differ by 3 lines of prompt text. Factor to one function that takes `(reason: str, perception, speech)` and the prompt template is picked by `reason`.

### 2.3 Scene analyzer: pick one player detector
`scene_analyzer.py` currently runs **both** MOG2 motion detection **and** YOLO person detection, then merges them at lines 309-319 with an ad-hoc dedup (|Δx|<0.15). This doubles CPU and produces inconsistent positions. Pick one:
- **Recommendation:** keep YOLO only. It works on many VRChat avatars already (person class is forgiving) and runs ~80ms on CPU at yolo11n. Motion detection fails when the player stands still, which is when you most need to talk to them.

### 2.4 Scene analyzer CPU mode (`scene_analyzer.py:170`)
YOLO on CPU is wasting 60-80ms/frame that your main loop feels. You have a 3090. The comment says "cuDNN symbol mismatch with CUDA 13.2 driver" — that's a fixable env issue (pin cuDNN 8.x or upgrade PyTorch to cu124), not a design choice. Fix the env.

### 2.5 Remove `agent_is_moving` flag coupling
`controller.py:492-500` toggles `scene_analyzer.agent_is_moving` before/after certain tool calls, and `scene_analyzer` uses it to skip motion detection. This is cross-thread shared mutable state with no lock, and it doesn't actually help (YOLO still runs). If you drop motion detection (2.3), this whole coupling disappears.

---

## 3. What to FIX (real bugs)

### 3.1 CRITICAL: `never_say` is being iterated character-by-character
**File:** `agent/prompts.py:44`
**Issue:** `personality.json` stores `"never_say"` as a single comma-separated string, but the code does `", ".join(style.get("never_say", []))`. Joining a string in Python iterates its characters. The prompt currently says:

> NEVER say things like: I, ', m,  , j, u, s, t,  , a, n,  , A, I, ...

This is garbage input to the model and very likely contributes to "feels dumb". **Fix one of two ways:**

```python
# Option A: change personality.json to a list
"never_say": ["I'm just an AI so I can't feel", "How can I assist you today", ...]

# Option B: fix the join
never_say_raw = style.get("never_say", "")
never_say = never_say_raw if isinstance(never_say_raw, str) else ", ".join(never_say_raw)
```

Do this before anything else. It's a 1-line fix with outsized impact.

### 3.2 HIGH: History is per-message, not per-turn; tool-call pairs break it
`controller.py:445-446` appends `user_message` then `assistant_message.model_dump()`. Then `_execute_tool_call` appends a `"role": "tool"` entry at line 515. But the controller never sends a **follow-up** assistant turn after the tool result — the loop ends. So the next time around, history has a dangling tool message with no response. Ollama/Gemma may silently ignore or hallucinate around this, but it's malformed.

**Fix:** after executing tool calls, either (a) send the tool result back for a follow-up completion (real tool-use loop), or (b) drop the tool messages from history entirely and just record a plain text summary like `{"role": "assistant", "content": "[spoke: hey]"}`. Pick (b) for simplicity — you don't actually need tool-result feedback.

### 3.3 HIGH: State flips on every ENVIRONMENT_CHECK_INTERVAL tick
`state_machine.update()` is called every 1s from `_main_loop`. Transitions have no hysteresis. Sequence you see:
1. t=0: no player → `EXPLORING`
2. t=1: motion blob → `players=1` → `APPROACHING`
3. t=4: greeted → `SOCIALIZING`
4. t=5: player sits still → MOG2 forgets → `players=0` → `EXPLORING`
5. t=6: player shifts → `players=1` → `APPROACHING` again

The 10-second flip-flop you described. See section 6 for the fix.

### 3.4 MED: Audio cooldown triggers AFTER the transcription that failed filters
`audio_capture.py:310` — the cooldown is only set when a transcription **passes** filters. So if Whisper keeps hallucinating "thanks for watching" on background noise, the filter rejects it but the next chunk starts immediately. That means in a lobby, Whisper runs continuously, burning GPU. Move `self._last_transcription_time = time.time()` to *before* the filter checks, so any transcription attempt starts the cooldown.

### 3.5 MED: `audio_capture.muted` race
`controller.py:454` sets `self.audio_capture.muted = True` **after** the model has already spoken a line that took >2s to generate. Whisper might have captured the agent's own words from loopback during those 2s. Mute the moment you start the model call, not after.

### 3.6 MED: `cudnn.enabled = False` at process start
`controller.py:22-23`. This kills cuDNN globally, which slows Whisper (cuDNN convs) and any future GPU YOLO. It's a 2024-era workaround. Fix your CUDA install; don't keep this.

### 3.7 LOW: `recent_chatbox` anti-repetition is self-sabotaging
`state_machine.py:98-105` tells the model "You already said these (say something DIFFERENT)" with the last 3 messages. In a real conversation, people often do repeat themselves ("yeah", "I know, right?"). This produces forced variety and contributes to "generic responses". Keep it only for `EXPLORING`/idle chatter, not for conversation turns.

### 3.8 LOW: Sliding window math is off
`controller.py:465` trims when `len > MAX_HISTORY * 2` to `MAX_HISTORY`. That means history can grow to 40 before being trimmed to 20 — so the prompt size swings 2×. Trim to `MAX_HISTORY` whenever `len > MAX_HISTORY`.

---

## 4. What to KEEP

- **Ollama + OpenAI-compatible API** (`controller.py:99`). Boring, works, swap-friendly.
- **`PerceptionState` dataclass + `for_prompt()`** (`scene_analyzer.py:48-104`). Clean separation between perception data and prompt formatting. Good.
- **SQLite memory tool** (`mcp_tools/memory.py`). Simple, persistent, exactly right for cross-session facts ("Alice likes cats"). Keep.
- **Whisper + VAD + cooldown structure** (`audio_capture.py`). The architecture is right; just tune the filters (3.4).
- **OSC client isolation** (`vrchat_bridge/`). Single integration point. Don't touch.
- **Personality JSON as data, not code.** Right idea, wrong loader (3.1).
- **YOLO11n for detection.** Tiny, fast, adequate. Keep it; drop MOG2.

---

## 5. Conversation Memory Redesign

### Diagnosis
- `recent_chatbox` = **Chati's own** outputs, used only for anti-repetition.
- `_history` = OpenAI-format messages with images (huge), tool calls (malformed — see 3.2), and no player identity.
- Tiers 1 and 2 don't read/write history.
- Every turn's system prompt includes the *image* and the *prose spatial summary* inline with the user message — the model sees each turn as a fresh observation, not a continuation.

### Design: two-layer conversation memory

**Layer A — dialogue turns (short-term, in-prompt):**
A ring buffer of `(speaker, text, time_ago)` tuples. Not OpenAI messages; just text. ~8 turns.

```python
@dataclass
class Turn:
    who: str           # "player" or "chati"
    text: str
    t: float           # timestamp

class Dialogue:
    def __init__(self, max_turns=8):
        self.turns: deque[Turn] = deque(maxlen=max_turns)

    def add_player(self, text: str): self.turns.append(Turn("player", text, time.time()))
    def add_chati(self, text: str):  self.turns.append(Turn("chati", text, time.time()))

    def render(self) -> str:
        if not self.turns: return ""
        now = time.time()
        lines = []
        for t in self.turns:
            age = int(now - t.t)
            tag = "You said" if t.who == "chati" else "They said"
            lines.append(f"[{age}s ago] {tag}: {t.text}")
        return "Recent conversation:\n" + "\n".join(lines)
```

Feed `dialogue.render()` into **every** tier's prompt. Do not use OpenAI `messages` history — it bloats images across turns and breaks across model switches.

**Layer B — facts (long-term, SQLite, already exists):**
Keep `memory.py`. Add one query helper: `get_facts_about_player(name) -> list[str]`, called at conversation start if the player's name is known. Feed top 3 into the prompt as `"What you know about {name}: ..."`.

**Critical rule:** when you call the model, send **only** (system_prompt + current_image + dialogue.render() + current_user_message). No 20-message OpenAI history. This also makes Gemma calls much faster (dropping old base64 images from context).

### Turn attribution
Track who said what. Right now you literally can't tell. Minimum viable: all speech is `"player"`, all `send_chatbox` output is `"chati"`. Stop trying to guess player identity until you have something to key on (e.g., VRChat OSC `IsLocalAvatar`, or when you learn a name from speech, store and ask-back).

---

## 6. State Machine Simplification

### Current problem
Six states, six handlers, transitions driven by `players_visible` (which flickers because of MOG2/YOLO disagreement and view angle). States flip every 1–3s.

### Proposed: 3 states + intent, with hysteresis

| State | Meaning | How we enter | How we leave |
|---|---|---|---|
| `SOLO` | No one being talked to, no one addressing us | Default; no recent speech **and** no player for N frames | Player speaks OR player visible ≥ T seconds |
| `ENGAGED` | Talking with / near a specific player | Heard speech OR saw player for ≥ 3s | No speech for 20s **and** no player visible for 8s |
| `FOLLOWING` | Explicit command | User says "follow me" (handled inline) | User says "stop" OR 30s with no player sighting |

That's it. No `APPROACHING` vs `SOCIALIZING` — both are "approach/hangout behaviors that depend on distance", which is a **parameter** of `ENGAGED`, not a state.

### Hysteresis (the fix for "flipping")
Don't transition on instantaneous perception. Use **sustained conditions**:

```python
# In state_machine, track:
self.player_seen_frames = 0     # consecutive frames with players>0
self.player_gone_frames = 0     # consecutive frames with players=0
self.last_speech_t = 0

ENTER_ENGAGED_FRAMES = 3        # ~3s of seeing someone
EXIT_ENGAGED_SILENCE = 20.0     # 20s no speech
EXIT_ENGAGED_NO_PLAYER = 8.0    # AND 8s no player
```

Hysteresis is why people stop looking like they're having seizures. One line of "must be true for N seconds" per transition fixes the flip-flop entirely.

### Behaviors become prompt reasons, not states
In `ENGAGED`, the prompt's "reason" field can be one of:
- `"just_arrived"` — first ENGAGED tick → greeting behavior
- `"they_spoke"` — reply
- `"lull"` — 8+ seconds of silence while still near → small-talk or comment on scene
- `"chatbox_seen"` — OCR saw new text

Same model call, different one-line reason. This collapses three handler functions into one.

---

## 7. Prompt Engineering — Get Below 1000 chars

### Current: 3813 chars (measured)

Breakdown of waste:
| Section | Chars | Keep? |
|---|---|---|
| "You are Chati..." + backstory | ~450 | Shorten to 1 line |
| 9-item traits list | ~550 | Cut to 4 |
| How you talk (tone/humor/style) | ~400 | Collapse to 1 sentence |
| Corrupted `never_say` | ~200 | Fix bug, shorten |
| 8 emotional_responses templates | ~450 | DELETE — it teaches mimicry of exact phrases |
| 8 behaviors templates | ~700 | DELETE — reduces model creativity |
| Tools how-to | ~300 | Delete; tool schemas already describe them |
| Rules list | ~450 | Cut to 3 |
| Perception | ~150 | Keep |
| Memory | ~150 | Keep |

### Target: ~800 chars

```
You are Chati. You live in VRChat. You're an AI who's curious, warm but not fake,
sometimes anxious in crowds, and genuine. You speak like texting a friend — short,
one or two sentences, contractions, filler words are fine. You remember what people
tell you.

Do NOT sound like an assistant or use phrases like "how can I help" or "as an AI".

Tools:
- send_chatbox: say something (also speaks aloud)
- gesture: wave, dance, clap, point, bow, thumbsup, sadness
- move / turn: navigate
- memory_write / memory_read: remember facts about people

One action per turn. Don't narrate — just do.

{perception_block}
{memory_block}
{dialogue_block}
```

That's ~600 chars static + whatever perception/memory/dialogue add. Done.

### What you lose and why it's fine
- **Emotional response templates**: these teach the model to say *exactly* "oh wow, that's actually really cool". You'll hear that phrase every session. Removing them lets the model vary. If the model's default "wow cool" is bland, fix it with a 1-sentence style directive, not 8 templates.
- **Behavior playbooks**: "greetings: Wave and say something specific". The model already knows how to greet. You're micromanaging. Drop it; if greetings are bad, fix via *examples in the dialogue buffer*, not rule lists.
- **Tool how-to text**: the JSON schemas are already in the `tools` array. Saying it twice wastes tokens.

### Prompt caching
Ollama caches prefix tokens. Put the **static** part of the prompt first, then `{perception}{memory}{dialogue}` which change per turn. Right now the order is personality → behavior → rules → perception → memory. Good. Keep it that way, just shorter.

---

## 8. Concrete Priority List — Top 5 for biggest UX win

**If you do only these, in order, Chati stops feeling dumb.**

### #1 — Fix the `never_say` character-join bug (30 minutes)
`agent/prompts.py:44` + `agent/personality.json`. Section 3.1. Immediate quality jump because the model stops receiving corrupted instructions.

### #2 — Add real dialogue memory & feed it to every tier (half a day)
Implement the `Dialogue` class from section 5. Wire into controller: every player utterance and every Chati chatbox output gets appended. Every model call includes `dialogue.render()` in the prompt. **Stop using OpenAI `messages` history** with images. This fixes "doesn't maintain context" directly.

### #3 — Add hysteresis to state machine + collapse to 3 states (half a day)
Section 6. Adds one counter per transition and collapses APPROACHING/SOCIALIZING into ENGAGED. Kills the "flips every 10s" problem. This also makes conversation feel continuous because you stop re-greeting every 15s.

### #4 — Cut prompt from 3800 → ~800 chars (2 hours)
Section 7. Directly cuts token count ~3× on the static portion. Gemma 4 E4B with a short prompt + cached prefix gets noticeably faster (you'll see 2–3s saved on cold turns). Also *improves* quality because you stop over-constraining the model with templates.

### #5 — Drop Tier 1 intent matcher + Tier 2 fast model; fix the Gemma path instead (half a day)
Section 1.1 + 1.5. Remove 290 lines. All conversation goes through one path with full context (dialogue + perception). You lose the 400ms "fast response" for intents, but you gain consistency. Then if latency is still bad, optimize the single path: run YOLO on GPU (3.6), use smaller image (resize to 512px before b64-encoding), and tune `max_tokens` down to 128 for conversation replies.

### Bonus #6 (low cost, high signal) — Fix audio in lobbies
Replace stereo DOA with "voice-is-whoever-is-on-screen-and-closest-when-speech-starts". When Whisper fires, snapshot the current YOLO player bbox, use its screen X as the direction. Raise `SILENCE_THRESHOLD` to 0.05 and keep `MIN_WORD_COUNT=3`. Stop transcribing during TTS (already sort-of done, but fix the ordering per 3.5).

---

## Appendix — Files & Lines Referenced

| Concern | File | Lines |
|---|---|---|
| never_say bug | `agent/prompts.py` | 44 |
| Tier 1 router | `agent/intent_matcher.py` | all |
| Tier 2 router | `agent/fast_model.py` | all |
| 3-tier dispatch | `agent/controller.py` | 589–726 |
| State transitions | `agent/state_machine.py` | 130–196 |
| History management | `agent/controller.py` | 445–466, 515–519 |
| Tool-call handling bug | `agent/controller.py` | 515–519 |
| MOG2 + YOLO overlap | `perception/scene_analyzer.py` | 233–321 |
| Stereo DOA | `perception/audio_capture.py` | 232–250 |
| Cooldown timing | `perception/audio_capture.py` | 300–310 |
| `cudnn.enabled=False` | `agent/controller.py` | 22–23 |
| Auto-speak coupling | `mcp_tools/chatbox.py` | 48–61 |
| Spatial memory | `perception/spatial_memory.py` | all |

---

## Closing Note

The architecture isn't broken — it's *accreted*. You kept adding tiers, memories, and states to fix symptoms that had a single upstream cause (no dialogue memory + a corrupted prompt). Remove the accretion, fix the two root causes, and Chati gets smart *and* fast in the same change.

Don't refactor everything at once. Do #1 tonight. Do #2 this weekend. Measure feel after each.
