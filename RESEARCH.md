# AI Companion / NPC / VTuber Architecture Research

**Purpose:** Learn from mature AI-companion implementations to inform the Chati VRChat agent design.
**Current Chati stack (baseline):** Gemma 4 E2B Q4 (Ollama) + YOLO11n + EasyOCR + faster-whisper + Piper TTS, driven by an explicit state machine (IDLE / EXPLORING / APPROACHING / SOCIALIZING / CONVERSING) on an RTX 3090.
**Scope:** Architecture, memory, perception, action, prompt engineering, and "aliveness" design across 13 representative projects.

---

## 1. Mantella (Skyrim / Fallout 4 mod) — deep dive

Mantella is the most mature "LLM-in-a-game-NPC" project in the wild and the single closest analog to a VRChat companion.

| Property | Details |
|---|---|
| Architecture pattern | **Event-driven pipeline** orchestrated by a Python HTTP server; the SKSE plugin fires game events, the server responds. No rigid FSM — conversation state is implicit. |
| LLM | User choice via OpenRouter, local koboldcpp, LM Studio, llama.cpp. Recommended: **Llama-3 8B fine-tuned on 8,800 Alpaca-style NPC dialogue pairs** (`art-from-the-machine/Mantella-Skyrim-Llama-3-8B-GGUF`). |
| Memory | **Conversation summaries** written to a per-NPC local text file when conversation ends; reloaded on next talk. A `bios_and_summaries` prompt variable keeps character identity separate from memory. Fork "Pantella" and "CHIM" add ChromaDB vector RAG. |
| Perception | STT (Moonshine / Whisper); **shared vision** (NPC sees what the player sees, optionally local via koboldcpp); in-game event injection (deaths, location changes, time of day). |
| Actions | Small closed set — `Follow`, `Offended`, `Forgiven`, attack, inventory. Triggered by parsing structured tokens the LLM emits in its reply. Disabled by default in MCM. |
| Pipeline | Player speaks → Whisper → prompt assembly (bio + summaries + recent events + vision caption + user turn) → LLM → parse action tokens → Piper/xVASynth/XTTS → in-game audio with lip-sync via SKSE. |
| Latency | Not hard real-time. Multi-second turn-taking is acceptable because NPCs are turn-based by social convention. |
| Prompt approach | **Long system prompt** (character bio) + structured sections + few-shot examples. Users have knob-level control via `prompt_style` settings. |
| Design insight | Success comes from **giving the LLM game context (events, vision, location) not just chat history.** The character feels "aware" because the prompt injects what the game is doing, not because the LLM is smart. |

Sources: [Mantella GitHub](https://github.com/art-from-the-machine/Mantella), [Mantella Nexus](https://www.nexusmods.com/skyrimspecialedition/mods/98631), [Llama-3 8B GGUF fine-tune](https://huggingface.co/art-from-the-machine/Mantella-Skyrim-Llama-3-8B-GGUF), [LLM fine-tuning repo](https://github.com/art-from-the-machine/Mantella-LLM-Fine-Tuning), [HN discussion](https://news.ycombinator.com/item?id=37370815), [Pantella fork](https://github.com/Pathos14489/Pantella).

---

## 2. Herika / CHIM (Skyrim) — predecessor to Mantella

| Property | Details |
|---|---|
| Architecture | **Two-process architecture**: SKSE plugin (C++) collects events and shoves them at a PHP/Python HTTP server ("HerikaServer") which talks to the LLM and returns queued lines back to the plugin. Cleanly separates game-side and AI-side. |
| LLM | Originally ChatGPT; now pluggable (koboldcpp, text-generation-webui, OpenRouter). |
| Memory | **ChromaDB vector store** — conversation turns embedded with OpenAI embeddings; on each turn the N most similar memories are retrieved and stuffed into the prompt. |
| Actions | Queue-based: server pushes dialogue lines + action commands, plugin plays them with lip-sync + TTS sequentially. |
| Insight | The **output queue** is the key trick. LLM generation is slow and non-deterministic; the queue decouples it from the game's real-time tick. |

Sources: [Herika Nexus](https://www.nexusmods.com/skyrimspecialedition/mods/89931), [HerikaServer](https://github.com/nan0bug00/HerikaServer), [HerikaAITools](https://github.com/Elbios/HerikaAITools), [CHIM](https://www.nexusmods.com/skyrimspecialedition/mods/126330).

---

## 3. Neuro-sama (Twitch AI VTuber, Vedal987) — deep dive

The most-watched AI entertainer on Earth. Almost entirely closed-source, but a lot has been revealed across dev streams, papers, and recreations.

| Property | Details |
|---|---|
| Architecture | **Multi-component pipeline, not a single model.** Separate Python processes for STT, vision-to-text, LLM inference, TTS, game agents, and a C#/Unity frontend for Live2D. Communicated via sockets. |
| LLM | **Custom fine-tuned ~2B parameter model, q2_k quantized** (as of early 2025). Trained on curated Twitch-chat transcripts + permissioned collab voice chats. 32k context. Aggressively small — latency prioritized over capability; **personality lives in the weights, not the prompt**. |
| Prompt | System prompt is **~1k tokens** (character, reaction templates, safety rules). Rest of the 32k window is conversation + injected context. |
| Memory | "Within-session context only" was the longtime reality. Long-term memory is an active work area; community believes some RAG is now in place plus manually curated persistent facts. |
| Perception | Twitch chat (text), collab voice (STT), game state converted to text via CV agents (80×60 grayscale frames, per-game custom agents for osu!, Minecraft, Pokémon, Inscryption, etc.). |
| Actions | Speech (Azure TTS "Ashley", +25% pitch), singing (RVC, pre-rendered), Live2D expression triggers, and per-game action protocols (high-level text commands, not keystrokes). |
| Latency | **900 ms → 700 ms optimization pass** mentioned on dev stream. 1–3 s is considered the streaming-survivable range. Achieved via tiny model + streaming TTS + small prompt. |
| State mgmt | **Signals pattern, not FSM.** A shared state object tracks "human speaking", "AI thinking", "chat activity", "time since last interaction". A prompter module continuously evaluates these signals to decide *whether* and *when* to generate. |
| Aliveness | Three layers: (1) weight-level personality via fine-tuning, (2) deliberately-picked reactive quirks (e.g. calling Vedal "mosquito" for weeks — enabled by LT memory), (3) game-playing ability giving her something to *do* on camera. |
| Learning loop | **Iterative batch fine-tuning, not online learning.** Vedal curates transcripts, trains offline, deploys a new checkpoint. Continuous learning was ruled out (catastrophic forgetting + safety). |

Sources: [Neuro-sama Wikipedia](https://en.wikipedia.org/wiki/Neuro-sama), [llm-memory-research / neuro-sama](https://github.com/Lin-Guanguo/llm-memory-research/blob/main/neuro-sama.research.md), [SynchroVerse case study](https://synchroverse.gitbook.io/synchroverse-whitepaper/aria-sentient-influencer/neuro-sama-case-study), [Wiki dev-stream notes](https://neurosama.fandom.com/wiki/Dev_Stream/July_22,_2024), [Grokipedia entry](https://grokipedia.com/page/Neuro-sama).

---

## 4. kimjammer/Neuro (open-source Neuro-sama recreation)

Best reference implementation for a local-only VTuber. Architecture is explicitly documented across 8 dev logs.

| Property | Details |
|---|---|
| Architecture | **Thread-per-subsystem with a shared `signals` object.** Every major component (STT, TTS, LLM, Twitch bot, etc.) runs in its own Python thread with its own event loop; cross-thread talk is queue-based. |
| LLM | Llama 3 8B via text-generation-webui (OpenAI-compatible endpoint). Earlier was Mistral 7B. |
| STT | KoljaB/RealtimeSTT + faster-whisper `tiny.en` — **streams transcription as the user speaks**, so transcription completes within ~100 ms of speech end. |
| TTS | KoljaB/RealtimeTTS + CoquiTTS XTTSv2 — **starts playing audio before the LLM finishes generating.** |
| Prompt | ~1k token system prompt with backstory, example conversations, personality adjectives, reaction templates. Prompt built via an **"Injection" system** — each module contributes a priority-weighted fragment; fragments are sorted and concatenated. |
| Decision to speak | Not a state machine. A `prompter.py` loop watches signals (human speech status, AI thinking flag, Twitch chat recency, idle timer) and decides when to trigger LLM inference. |
| Frontend | SvelteKit + shadcn-svelte control panel, Socket.IO transport on port 8080. |
| Audio routing | **Voicemeeter Potato** as a virtual mixer — captures computer audio + Discord + streams output simultaneously so the AI doesn't hear itself. |
| Hardware | 12 GB VRAM minimum. |
| Key lesson (Dev Log 8) | Earlier builds had "a mix of multithreading, event loops, and other badness." Moving to **one event loop per thread** plus queues dramatically simplified the code. |
| Key lesson (Dev Log 4) | Aligned base models (Llama-3-Instruct) resist certain prompted behaviors (swearing, rudeness) no matter what the system prompt says — **fine-tuning is required for strong personality on aligned models**. |

Sources: [kimjammer/Neuro](https://github.com/kimjammer/Neuro), [Dev Log 1](https://blog.kimjammer.com/neuro-dev-log-1/), [Dev Log 4](https://blog.kimjammer.com/neuro-dev-log-4/), [Dev Log 8](https://blog.kimjammer.com/neuro-dev-log-8/).

---

## 5. Open-LLM-VTuber

Most-featureful open-source AI VTuber framework in 2026. Runs fully local.

| Property | Details |
|---|---|
| Architecture | Swappable modules for ASR, LLM, TTS, avatar. Client-server split so the avatar can run on one machine and inference on another. |
| LLM | Anything with OpenAI-compat: Ollama, LM Studio, vLLM, Claude, Gemini, DeepSeek, GGUF. |
| ASR | sherpa-onnx, FunASR, Faster-Whisper, Whisper.cpp, Azure, Groq. |
| TTS | sherpa-onnx, MeloTTS, Coqui, GPT-SoVITS, Bark, CosyVoice, Edge TTS, Fish, Azure. |
| Voice interruption | **Client-side VAD** via `ricky0123/vad-web`. AI speech is barge-in interruptible. **AI doesn't hear its own voice** (key trick — filtered at the VAD layer). |
| Streaming | Producer-consumer: sentences are synthesized as soon as they arrive from the LLM; playback and synthesis are independent queues. |
| Memory | Persistent chat-log storage; long-term memory is being reworked. |
| Extras | Live2D expression mapping, AI "inner thoughts" display, proactive speaking, touch feedback. |

Sources: [Open-LLM-VTuber](https://github.com/Open-LLM-VTuber/Open-LLM-VTuber), [project docs](http://docs.llmvtuber.com/en/docs/intro/), [awesome-ai-vtubers list](https://github.com/proj-airi/awesome-ai-vtubers).

---

## 6. AITuberKit (tegnike)

| Property | Details |
|---|---|
| Architecture | Next.js web-app frontend + backend adapter to various LLM/TTS APIs. "Soul system" (dialogue) / "Shell system" (avatar + voice) split. |
| LLM | OpenAI, Anthropic, Gemini, local via adapters. |
| TTS | VOICEVOX, Koeiromap, Google TTS, ElevenLabs. |
| Avatar | VRM (3D) or Live2D (2D). |
| Insight | The **"Soul vs Shell"** separation is the cleanest conceptual model for AI-Vtuber design: keep personality/reasoning totally decoupled from presentation. |

Sources: [AITuberKit GitHub](https://github.com/tegnike/aituber-kit), [docs](https://docs.aituberkit.com/en/), [dev blog post](https://medium.com/@nikechan/creating-an-aituber-using-ai-generated-videos-15fa2a2979ff).

---

## 7. ChatVRM / LocalChatVRM (Pixiv)

| Property | Details |
|---|---|
| Architecture | Pure browser: **Web Speech API** (STT) → ChatGPT API (original) or **Chrome Built-in AI** (LocalChatVRM) → Koeiromap TTS → @pixiv/three-vrm avatar. |
| LLM | Remote (original) or in-browser (LocalChatVRM uses Chrome's built-in Gemini Nano-equivalent). |
| Memory | In-session only (browser state). |
| Insight | Demonstrates how **small, well-composed components** give a high-quality feel. No FSM, no event loop — just request/response driving an animation. Works because it's a single-user, turn-based experience. |

Sources: [ChatVRM GitHub (archived)](https://github.com/pixiv/ChatVRM), [local-chat-vrm](https://github.com/pixiv/local-chat-vrm), [Gigazine writeup](https://gigazine.net/gsc_news/en/20230508-chatvrm/).

---

## 8. Voyager (MineDojo / NVIDIA) — embodied Minecraft agent

| Property | Details |
|---|---|
| Architecture | Three pillars: (1) **automatic curriculum** (proposes tasks that maximize exploration), (2) **skill library** — a growing collection of *executable JavaScript functions* indexed by an embedding so the agent can retrieve past skills by semantic query, (3) **iterative prompting** with environment feedback + execution errors + self-verification. |
| LLM | GPT-4 via API (black-box). No fine-tuning. |
| Action space | **Code, not button presses.** LLM writes Mineflayer JS that the environment executes. |
| Memory | Vector-indexed skill library — this is the long-term memory. |
| Key result | 3.3× more items, 2.3× longer travel, 15.3× faster tech-tree progress than prior SOTA. Skills transfer across worlds. |
| Insight for VRChat | **Treat learned behaviors as retrievable code snippets.** The "skill library" pattern generalizes: a VRChat agent could retrieve "how to do the dance emote" as a snippet/function rather than reasoning from scratch. |

Sources: [Voyager paper (arXiv)](https://arxiv.org/abs/2305.16291), [project site](https://voyager.minedojo.org/), [Voyager GitHub](https://github.com/MineDojo/Voyager), [TDS walkthrough](https://towardsdatascience.com/tool-use-agents-and-the-voyager-paper-5a0e548f8b38/).

---

## 9. Mindcraft / mindcraft-ce — multi-agent Minecraft LLM bots

| Property | Details |
|---|---|
| Architecture | Node.js + **Mineflayer** (MC bot lib). Central **mindserver** hub coordinates multiple agent instances; a localhost:8080 web UI is the dashboard. |
| Agent profile | JSON file (`andy.json`) — **specifies different models for chat, coding, and embeddings** (e.g. cheap model for chat, big model for code generation, small model for embeddings). |
| LLM | OpenAI, Gemini, Anthropic, Ollama (local). |
| Extensions | "minecraft-ai" fork adds **daily self-generated task lists, reflective behavior cycles, dynamic agent profiles** (Generative-Agents-style). |
| Insight | **Route different jobs to different-sized models.** Don't burn your big model on trivial acknowledgements. Embeddings and "should I speak?" checks can run on tiny models. |

Sources: [mindcraft-bots/mindcraft](https://github.com/mindcraft-bots/mindcraft), [mindcraft-ce](https://github.com/mindcraft-ce/mindcraft-ce), [aeromechanic/minecraft-ai](https://github.com/aeromechanic000/minecraft-ai), [MineCollab paper](https://mindcraft-minecollab.github.io/).

---

## 10. AI Town (a16z)

The canonical open-source "Generative Agents"-style sim.

| Property | Details |
|---|---|
| Architecture | **Tick-based simulation.** Central game engine (Convex) loads full game state each tick; only the engine mutates tables; agents submit inputs via a queue. Long LLM calls go through `startOperation` so the sim doesn't block. |
| LLM | Llama-3 default; any OpenAI-compat endpoint. |
| Embeddings | `mxbai-embed-large` for memory retrieval. |
| Memory | After each conversation GPT summarizes → summary embedded → stored in Convex vector DB. On new conversations the **top-3 most similar memories** are retrieved into the prompt. |
| Conversation flow | Explicit three-stage FSM: **starting → continuing → exiting.** |
| Prompt | Structured: personality block + retrieved memories + recent turns. |
| Insight | **Strict single-writer state discipline** is what keeps a multi-agent sim sane. Only the engine mutates state; everything else submits inputs. |

Sources: [a16z-infra/ai-town](https://github.com/a16z-infra/ai-town), [ARCHITECTURE.md](https://github.com/a16z-infra/ai-town/blob/main/ARCHITECTURE.md), [cat-town fork](https://github.com/ykhli/cat-town).

---

## 11. tuckerisapizza/VRChat-AI-Bot

Real VRChat bot; minimal architecture but useful reference.

| Property | Details |
|---|---|
| Architecture | **Single main loop** + 3 daemon threads. Global flags (`is_talking`, `isemoting`, `movementpaused`, `listencount`) track state. `stop_event` coordinates interruption. |
| LLM | **Character.AI** via `pycai` SDK (no local LLM). |
| STT | Google Cloud Speech Recognition (1.5 s timeout, 8 s max phrase). |
| VRChat | Pure **OSC over UDP to 127.0.0.1:9000.** Commands like `/input/Jump`, `/input/MoveForward`, `/avatar/parameters/VRCEmote`, `/chatbox/input`. |
| Prompt | Raw user speech sent directly to Character.AI; status strings appended to the VRChat chatbox. |
| Insight | Shows the **OSC contract with VRChat is trivially simple** — the hard work is everything upstream of OSC. |

Source: [VRChat-AI-Bot](https://github.com/tuckerisapizza/VRChat-AI-Bot), [botscript](https://github.com/tuckerisapizza/VRChat-AI-Bot/blob/main/botscript_torelease.py).

---

## 12. AIAvatarKit (uezo) — VRChat-compatible companion framework

| Property | Details |
|---|---|
| Architecture | Modular: VAD + STT + LLM + TTS with swappable providers. VRChat and other metaverse platforms listed as targets. |
| LLM | ChatGPT, Gemini, Claude, any LiteLLM/Dify-supported model. |
| Insight | Closest thing to an off-the-shelf framework for what Chati is building. Worth reading the repo for module-boundary conventions. |

Source: [aiavatarkit](https://github.com/uezo/aiavatarkit).

---

## 13. Discord AI bots (Animus, hc20k/LLMChat, Najmul190/Discord-AI-Selfbot, Caellwyn/long-memory-character-chat)

| Property | Common Patterns |
|---|---|
| Concurrency | Per-channel (or per-user) conversation objects, so multiple simultaneous conversations don't bleed into each other. |
| Memory | **Two-tier is standard**: (1) rolling scratchpad summary of the last N turns, (2) vector DB (Chroma / Pinecone / FAISS / Milvus) of embedded fact chunks, retrieved by semantic similarity at prompt time. |
| Personality | **First-reply sampling trick** — cache an early response and paste it back in every prompt as a "speaking style example". Keeps tone stable over 100+ turns. |
| Prompt | System prompt (persona) + retrieved facts + rolling summary + recent turns. |
| Insight | **Conversation memory is solved.** Rolling summary + vector RAG works; don't reinvent it. |

Sources: [Animus](https://animus.velocode.io), [hc20k/LLMChat](https://github.com/hc20k/LLMChat), [Najmul190/Discord-AI-Selfbot](https://github.com/Najmul190/Discord-AI-Selfbot), [Caellwyn/long-memory-character-chat](https://github.com/Caellwyn/long-memory-character-chat), [Medium: LangChain + Milvus](https://medium.com/@zilliz_learn/building-a-conversational-ai-agent-with-long-term-memory-using-langchain-and-milvus-0c4120ad7426), [Adding Memory to Conversational AI](https://getclaw.sh/blog/adding-memory-to-conversational-ai-companion).

---

## Comparison Matrix

| Project | Arch pattern | Model | Context | Memory | Perception | Actions | Latency | State mgmt |
|---|---|---|---|---|---|---|---|---|
| **Mantella** | Event-driven HTTP server | Llama-3 8B fine-tune / any OpenRouter | ~8k | Per-NPC summary files + bios | STT + vision + game events | Follow/Offended/Forgiven + inventory | Turn-based, not RT | Implicit |
| **Herika/CHIM** | Two-process client/server | ChatGPT / local | ~4–8k | ChromaDB RAG | STT + events | Queued dialogue + actions | Turn-based | Queue-driven |
| **Neuro-sama** | Multi-proc signals | Custom 2B q2_k | 32k | RAG + curated LT (emerging) | STT + CV + Twitch chat + game text | TTS + Live2D + game protocol | 700 ms target | Signals object |
| **kimjammer/Neuro** | Thread-per-module + signals | Llama-3 8B | 8k–32k | Session only | STT + Twitch | TTS + Live2D | ~1 s | Signals, prompter loop |
| **Open-LLM-VTuber** | Swappable modular pipeline | Any OpenAI-compat | Variable | Chat logs | STT + VAD + optional vision | TTS + Live2D | Streaming | Modular |
| **AITuberKit** | Web Next.js adapter | Any | Variable | Variable | Text/voice | TTS + VRM/Live2D | Network-bound | Soul/Shell |
| **ChatVRM** | Browser only | ChatGPT / in-browser | Small | Session only | Web Speech API | Koeiromap TTS + VRM | Near-instant | None |
| **Voyager** | Loop + skill library | GPT-4 | Large | Vector-indexed code skills | Text game state | Executable JS (Mineflayer) | Minutes/task | None (planner loop) |
| **Mindcraft** | Mineflayer + LLM router | Multi-model per task | Variable | Generative-agents-style | Text game state | Mineflayer actions | Seconds | Agent profile JSON |
| **AI Town** | Tick-based sim | Llama-3 + mxbai | 8k | Summary + vector DB | Proximity + chat | Movement + chat | Non-real-time | Explicit 3-stage FSM |
| **VRChat-AI-Bot** | Main loop + 3 threads | Character.AI | N/A | Cloud-managed | Google STT | VRChat OSC | Google STT bound | Global flags |
| **AIAvatarKit** | VAD→STT→LLM→TTS | Pluggable | Variable | Variable | VAD + STT | OSC / avatar | Streaming | Modular |
| **Discord bots** | Per-channel session objects | Varies | 8k–32k | Rolling summary + vector | Text | Text reply | Async | Per-conversation |
| **Chati (current)** | Explicit FSM + YOLO + OCR + Ollama | Gemma 4 E2B Q4 | ~8k | — (TBD) | YOLO + OCR + STT + audio DOA | OSC + Piper TTS | ~5 s | 5-state FSM |

---

## Answers to the Key Questions

### 1. What are the most common successful architecture patterns?

Across the 13 projects, two dominant patterns:

- **Thread-per-subsystem + shared signal object** (Neuro-sama, kimjammer, Open-LLM-VTuber). STT, LLM, TTS, perception each run independently; a small set of flags coordinates them. No explicit FSM.
- **Event-driven pipeline with a queue between generation and playback** (Mantella, Herika). Game events produce requests; LLM output is queued; playback drains the queue. Decouples slow, non-deterministic LLM inference from the real-time game tick.

The **rarer** pattern is strict FSM — only AI Town really uses one, and only for conversation lifecycle (start/continue/exit). Chati's 5-state FSM is an outlier among successful systems.

### 2. What makes a chatbot feel "alive" in a virtual space?

Looking at what Neuro-sama, Mantella, and Voyager have that the failed projects don't:

1. **The LLM sees game context, not just chat.** Mantella injects vision + events; Neuro has per-game CV agents; Voyager has environment feedback. If the only input is chat text, the character feels like a chatbot in a costume.
2. **Proactive behavior between user turns.** Idle animations, occasional unsolicited comments, reactions to things that happened. kimjammer's "time since last interaction" signal exists precisely to trigger this.
3. **Persistent callbacks.** Neuro calling Vedal "mosquito" for weeks; Herika remembering an argument — long-term memory enables the social payoff.
4. **Personality in the weights, scaffolding in the prompt.** Aligned base models refuse quirky behavior no matter how long the system prompt. Fine-tuning or careful base-model selection is non-optional for strong characters.
5. **Deliberate imperfection.** Vedal's keeping Neuro at 2B q2_k — she makes mistakes, hallucinates, fixates — and that's charming, not broken.

### 3. Should a VRChat agent use explicit state machines or a conversation loop?

**Hybrid, leaning toward signals.** Explicit FSMs help for **locomotion and social proxemics** (idle vs. approaching vs. conversing is a real physical state) but are the wrong tool for conversation flow, which every successful project treats as **implicit and signal-driven**.

Concrete recommendation for Chati: **keep the FSM for body behaviors only** (IDLE/EXPLORING/APPROACHING are about navigation, not dialogue). **Replace CONVERSING logic with a kimjammer-style prompter loop** that evaluates signals (user speaking? LLM busy? silence timer? new perception event?) to decide *whether* to generate. This removes the bottleneck of "I'm in state X so I can't react to Y."

### 4. How do successful projects handle "always listening + avoiding interruption"?

Five techniques, usually combined:

1. **Echo avoidance via audio routing** — Voicemeeter virtual mixer (kimjammer) or a client-side VAD that explicitly filters the AI's own output channel (Open-LLM-VTuber). **Don't let the AI hear itself.**
2. **Streaming STT that finalizes on silence** — RealtimeSTT with faster-whisper tiny.en; 200–300 ms silence = end-of-turn.
3. **Barge-in friendly TTS** — TTS synthesis and playback in separate queues (Open-LLM-VTuber producer-consumer); user VAD activity cancels the playback queue and flushes.
4. **Continuous signal evaluation, not state gating** — don't refuse to listen because you're "in SPEAKING state"; always listen, and let the signals loop arbitrate.
5. **A silence-break timer, not a constant talker** — kimjammer's "time since last interaction" signal causes proactive speech only when the room is genuinely quiet.

### 5. What's the prompt length sweet spot for local LLMs?

From the research and the Neuro-sama case in particular:

- **System prompt: 500–1500 tokens.** Neuro-sama runs ~1k; kimjammer ~1k. More than this and the model starts losing instruction fidelity on small quantized checkpoints.
- **Effective context usage: 4k–16k tokens.** For 7B/8B Q4 models, real quality stays high up to 16k–32k tokens; beyond that, the "lost in the middle" problem kicks in hard. The advertised 128k windows on local models are largely theoretical.
- **For 2B models** (Chati's Gemma-4 E2B is in this class): keep **total prompt under ~4k tokens**. The Neuro-sama 2B q2_k is proof that aggressive compression is viable *if* the personality is in the weights, not the prompt.

### 6. How do projects balance scripted behaviors with LLM-driven decisions?

Universal answer: **LLM decides *what* and *why*; scripted code decides *how*.**

- Mantella: LLM writes dialogue; SKSE plugin handles lip-sync, camera, audio mixing.
- Voyager: LLM writes JS snippets; Mineflayer actually moves the character.
- VRChat-AI-Bot: LLM produces text; OSC sends `/input/MoveForward`.
- Neuro-sama: LLM produces speech tokens; Azure TTS + Live2D runtime handle presentation.

Practical rule: **any behavior that runs > 10 Hz should be scripted** (idle animations, lip-sync, head-tracking, walk cycles). **Any behavior that happens < 1 Hz can be LLM-gated** (conversation turns, target selection, emote choice, navigation goals). The FSM owns the fast loop; the LLM owns the slow loop.

### 7. Common mistakes / over-engineering traps

From all projects surveyed, the repeated failure modes are:

1. **Putting the LLM in the hot loop.** Anyone who tried calling the LLM every frame/tick hit latency walls. The queue-and-stream pattern (Mantella, kimjammer, Open-LLM-VTuber) exists because of this.
2. **One giant FSM for everything.** State explosion is real. AI Town keeps its FSM minimal (3 stages for conversation only); Chati's current 5-state mixes dialogue and locomotion, which will ossify.
3. **Overtrusting the prompt for personality.** kimjammer Dev Log 4 and the Neuro-sama research paper both say the same thing: **aligned models resist strong persona prompts.** Either fine-tune or pick a less-aligned base (Mistral, Qwen, Nous-Hermes).
4. **Shared mutable state across threads without discipline.** Dev Log 8 explicitly calls this out as "badness". Enforce: one queue in, one queue out per module.
5. **Memory as an append-only chat log.** Every mature project (AI Town, Herika, Discord bots) uses **summary + vector retrieval**. Raw chat logs hit the context wall within minutes.
6. **Ignoring audio echo.** Without Voicemeeter routing or a VAD echo filter, the bot will talk to itself forever. Treat this as a day-one problem, not a polish item.
7. **Big model when small is enough.** Neuro-sama runs 2B. Mindcraft routes small models to small jobs. Chati's Gemma-4 E2B is correctly sized; don't scale up without a specific reason.
8. **Forgetting the "why now?" question.** Many bots have a system for *what* to say but none for *when* to say it. kimjammer's prompter loop and Mantella's event injection are the two good patterns.
9. **Coupling perception to reasoning.** YOLO, OCR, and STT should each write into a shared perception buffer; the LLM reads that buffer. Don't wire YOLO directly into a prompt-building function inside the FSM — it forces a redesign every time perception changes.

---

## Specific Recommendations for Chati (VRChat)

Given the current stack (Gemma 4 E2B Q4 / YOLO11n / EasyOCR / faster-whisper / Piper / 5-state FSM on RTX 3090), the highest-leverage changes in priority order:

1. **Split the FSM into body-FSM and a conversation signals loop.**
   - Body FSM keeps IDLE/EXPLORING/APPROACHING — they're physical states driving OSC movement.
   - Replace SOCIALIZING/CONVERSING with a **kimjammer-style prompter loop** reading signals: `user_speaking`, `llm_busy`, `silence_duration`, `new_perception_event`, `nearby_players`, `recent_chatbox_messages`. The prompter decides *when* to call the LLM. This unblocks reacting to things during movement.

2. **Introduce an output queue between the LLM and Piper/OSC.**
   - LLM streams sentences → TTS queue → OSC `/chatbox/input` + Piper audio playback. Cancellable on user barge-in.
   - Matches Open-LLM-VTuber's producer-consumer pattern and Mantella's dialogue queue.

3. **Add two-tier memory** (copy from Discord-bot and AI-Town patterns):
   - Tier 1 — Rolling summary (last N turns, 200–400 tokens).
   - Tier 2 — Vector store (ChromaDB; embed with `mxbai-embed-large` or `nomic-embed-text` via Ollama). Summarize each closed conversation, embed, store. On new prompt, retrieve top 3.
   - Per-user (or per-avatar-ID) scoped. Critical for VRChat because the same usernames come back.

4. **Echo avoidance.**
   - Option A: Voicemeeter Banana/Potato routing (kimjammer's approach).
   - Option B: VAD-level filter — while Piper is speaking, mute the STT input channel. Simpler, fewer moving parts.
   - Do this before adding any more features. Without it, every other improvement is wasted.

5. **Perception buffer pattern.**
   - YOLO, OCR, stereo DOA, chatbox, audio — each module writes timestamped events to a ring buffer.
   - Prompter loop composes the prompt from the buffer at generation time.
   - Decouples perception cadence from LLM cadence; lets you add/remove sensors without touching the FSM.

6. **Prompt budget for Gemma-4 E2B Q4 (~2B class):**
   - System prompt: ≤ 800 tokens (persona + VRChat etiquette + action format).
   - Retrieved memories: ≤ 400 tokens (top-3 summaries).
   - Perception snapshot: ≤ 400 tokens (who's nearby, what OCR saw, recent DOA).
   - Recent dialogue: ≤ 1000 tokens.
   - Total budget ~2.5–3k tokens. Leaves headroom and keeps latency in the 2–3 s range (well below Neuro-sama's 700 ms target, but acceptable for VRChat social pacing).

7. **Action vocabulary as a fixed token set** (Mantella pattern):
   - Have the LLM emit structured actions like `[EMOTE:wave]`, `[MOVE:follow player42]`, `[LOOK:nearest_speaker]` at the end of its reply.
   - A small parser in Python maps these to OSC messages. This is the Mantella/Voyager insight: narrow the action space to something reliable.

8. **Don't fight Gemma's alignment — pick a less-aligned base if you want quirks.**
   - If Gemma-4 E2B refuses to be rude / sarcastic / weird even with strong prompting, switch to a less-aligned 2B–3B model (Qwen 2.5 1.5B/3B, Phi-3 mini, or a Nous-Hermes tune). Documented in kimjammer Dev Log 4 and the Neuro-sama research page.

9. **Reserve "fine-tuning" for v2.** Mantella's fine-tune dataset is 8,800 Alpaca-style pairs. That's ~10 hours of curation. Skip for now; strong prompt + good base model + fine-tuned persona-example injection (the Discord-bot "first-reply caching" trick) will get 80% of the value.

10. **What *not* to add right now:**
    - Do not add a behavior tree on top of the FSM. Overkill for a single agent.
    - Do not add RL. Every project surveyed uses pure prompting or offline fine-tuning.
    - Do not add a second LLM for "planning". Mindcraft gets away with multi-model routing because its tasks are computationally heterogeneous; Chati's aren't.
    - Do not chase 128k context. Stay under 8k, keep it fast.

---

## Final Sources Index

- [Mantella GitHub](https://github.com/art-from-the-machine/Mantella)
- [Mantella Nexus](https://www.nexusmods.com/skyrimspecialedition/mods/98631)
- [Mantella Llama-3 8B GGUF](https://huggingface.co/art-from-the-machine/Mantella-Skyrim-Llama-3-8B-GGUF)
- [Mantella fine-tune repo](https://github.com/art-from-the-machine/Mantella-LLM-Fine-Tuning)
- [Pantella fork](https://github.com/Pathos14489/Pantella)
- [Herika Nexus](https://www.nexusmods.com/skyrimspecialedition/mods/89931)
- [HerikaServer](https://github.com/nan0bug00/HerikaServer)
- [CHIM](https://www.nexusmods.com/skyrimspecialedition/mods/126330)
- [Neuro-sama Wikipedia](https://en.wikipedia.org/wiki/Neuro-sama)
- [Neuro-sama research doc](https://github.com/Lin-Guanguo/llm-memory-research/blob/main/neuro-sama.research.md)
- [Neuro-sama case study](https://synchroverse.gitbook.io/synchroverse-whitepaper/aria-sentient-influencer/neuro-sama-case-study)
- [kimjammer/Neuro](https://github.com/kimjammer/Neuro)
- [Neuro Dev Log 1](https://blog.kimjammer.com/neuro-dev-log-1/)
- [Neuro Dev Log 4](https://blog.kimjammer.com/neuro-dev-log-4/)
- [Neuro Dev Log 8](https://blog.kimjammer.com/neuro-dev-log-8/)
- [Open-LLM-VTuber](https://github.com/Open-LLM-VTuber/Open-LLM-VTuber)
- [Open-LLM-VTuber docs](http://docs.llmvtuber.com/en/docs/intro/)
- [awesome-ai-vtubers](https://github.com/proj-airi/awesome-ai-vtubers)
- [AITuberKit](https://github.com/tegnike/aituber-kit)
- [AITuberKit docs](https://docs.aituberkit.com/en/)
- [ChatVRM (archived)](https://github.com/pixiv/ChatVRM)
- [LocalChatVRM](https://github.com/pixiv/local-chat-vrm)
- [Voyager paper (arXiv 2305.16291)](https://arxiv.org/abs/2305.16291)
- [Voyager site](https://voyager.minedojo.org/)
- [Voyager GitHub](https://github.com/MineDojo/Voyager)
- [mindcraft-bots/mindcraft](https://github.com/mindcraft-bots/mindcraft)
- [mindcraft-ce](https://github.com/mindcraft-ce/mindcraft-ce)
- [aeromechanic/minecraft-ai](https://github.com/aeromechanic000/minecraft-ai)
- [a16z-infra/ai-town](https://github.com/a16z-infra/ai-town)
- [AI Town ARCHITECTURE.md](https://github.com/a16z-infra/ai-town/blob/main/ARCHITECTURE.md)
- [VRChat-AI-Bot](https://github.com/tuckerisapizza/VRChat-AI-Bot)
- [aiavatarkit](https://github.com/uezo/aiavatarkit)
- [hc20k/LLMChat](https://github.com/hc20k/LLMChat)
- [Najmul190/Discord-AI-Selfbot](https://github.com/Najmul190/Discord-AI-Selfbot)
- [Caellwyn/long-memory-character-chat](https://github.com/Caellwyn/long-memory-character-chat)
- [Animus Discord bot](https://animus.velocode.io)
- [Local LLM context length guide](https://local-ai-zone.github.io/guides/context-length-optimization-ultimate-guide-2025.html)
- [Context Kills VRAM](https://medium.com/@lyx_62906/context-kills-vram-how-to-run-llms-on-consumer-gpus-a785e8035632)
- [Voice AI interruption handling](https://callbotics.ai/blog/ai-voice-agent-interruption-handling)
- [Speechmatics on interruption](https://www.speechmatics.com/company/articles-and-news/your-ai-assistant-keeps-cutting-you-off-im-fixing-that)
- [State Machine Game Programming Patterns](https://gameprogrammingpatterns.com/state.html)
- [Adding memory to AI companion](https://getclaw.sh/blog/adding-memory-to-conversational-ai-companion)
- [LangChain + Milvus long-term memory](https://medium.com/@zilliz_learn/building-a-conversational-ai-agent-with-long-term-memory-using-langchain-and-milvus-0c4120ad7426)
