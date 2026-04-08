# How the Unmute Codebase Works

> A complete walkthrough of every component in the Unmute voice conversation system — how they connect, communicate, and process data from browser microphone to spoken AI response.

---

## Table of Contents

1. [Architecture at a Glance](#architecture-at-a-glance)
2. [Two Entry Points](#two-entry-points)
3. [The WebSocket Server — `main_websocket.py`](#the-websocket-server--main_websocketpy)
4. [The Handler — `unmute_handler.py`](#the-handler--unmute_handlerpy)
5. [The Quest Manager — Async Lifecycle](#the-quest-manager--async-lifecycle)
6. [STT Pipeline — Speech to Text](#stt-pipeline--speech-to-text)
7. [LLM Pipeline — The Brain](#llm-pipeline--the-brain)
8. [TTS Pipeline — Text to Speech](#tts-pipeline--text-to-speech)
9. [The Event Protocol](#the-event-protocol)
10. [Service Discovery](#service-discovery)
11. [Voice System](#voice-system)
12. [Recording & Debugging Tools](#recording--debugging-tools)
13. [Configuration & Constants](#configuration--constants)
14. [Supporting Modules](#supporting-modules)
15. [Full Request Lifecycle — Start to Finish](#full-request-lifecycle--start-to-finish)
16. [File-by-File Reference Table](#file-by-file-reference-table)

---

## Architecture at a Glance

Unmute is a **modular voice conversation system** with three external AI services (STT, LLM, TTS) orchestrated by a Python backend, served to users via WebSocket.

```
┌──────────────┐          ┌─────────────────────────────────────────────────────────┐
│   BROWSER    │          │                  PYTHON BACKEND                         │
│              │  Opus    │                                                         │
│  Microphone ─┼──audio──►│  main_websocket.py          unmute_handler.py           │
│              │  over    │  ┌─────────────┐           ┌──────────────────┐         │
│              │  WebSocket│  │receive_loop │──frames──►│ receive()        │         │
│              │          │  │             │           │   │               │         │
│              │          │  │emit_loop    │◄──events──│   ├─► STT ──────────►[STT Server]
│              │          │  │             │           │   │               │         │
│  Speaker  ◄──┼──Opus────│  │             │           │   ├─► LLM ──────────►[LLM Server]
│              │  audio   │  └─────────────┘           │   │               │         │
│              │          │                            │   └─► TTS ──────────►[TTS Server]
│  UI State ◄──┼──JSON────│                            └──────────────────┘         │
│              │  events  │                                                         │
└──────────────┘          └─────────────────────────────────────────────────────────┘
```

**Key design principle:** Each user connection gets its own `UnmuteHandler` instance. The backend scales by running multiple server processes (avoids GIL), not by handling many connections per process. Default limit: 4 concurrent connections per process.

---

## Two Entry Points

Unmute can run in two modes:

### 1. WebSocket Mode — `main_websocket.py` (Production)

A **FastAPI** server that speaks an OpenAI Realtime API–compatible WebSocket protocol. The browser frontend connects here.

```
Browser ◄──WebSocket──► FastAPI (main_websocket.py) ──► UnmuteHandler
```

**How to run:**
```bash
fastapi dev unmute/main_websocket.py
```

### 2. Gradio Mode — `main_gradio.py` (Development)

A **Gradio**-based UI using FastRTC for WebRTC audio streaming. Simpler but includes a built-in debug dashboard (chat history, debug dict, live waveform plot).

```
Browser ◄──WebRTC──► Gradio/FastRTC (main_gradio.py) ──► UnmuteHandler
```

**How to run:**
```bash
python unmute/main_gradio.py
```

Both modes use the exact same `UnmuteHandler` — they differ only in transport (WebSocket vs WebRTC) and UI.

---

## The WebSocket Server — `main_websocket.py`

This is the production server. Let's trace what happens when a user connects.

### Startup & Server Config

```python
app = FastAPI()

# Only 4 concurrent connections per process (GIL avoidance strategy)
MAX_CLIENTS = 4
SEMAPHORE = asyncio.Semaphore(MAX_CLIENTS)

# CORS for local development
CORS_ALLOW_ORIGINS = ["http://localhost", "http://localhost:3000"]
```

### REST Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Health ping — returns `{"message": "You've reached the Unmute backend server."}` |
| `/v1/health` | GET | Checks if STT, TTS, LLM, and voice cloning servers are reachable. Returns a `HealthStatus` JSON |
| `/v1/voices` | GET | Returns the list of "good" voices from `voices.yaml` |
| `/v1/voices` | POST | Upload an audio file for voice cloning. Returns a `custom:UUID` voice name |
| `/v1/voice-donation` | GET | Get a verification text for voice donation |
| `/v1/voice-donation` | POST | Submit a voice donation recording |

### The WebSocket Route — `/v1/realtime`

This is where the real action happens:

```python
@app.websocket("/v1/realtime")
async def websocket_route(websocket: WebSocket):
    async with SEMAPHORE:          # Rate limiting
        await websocket.accept(subprotocol="realtime")  # OpenAI-compatible
        
        handler = UnmuteHandler()
        async with handler:        # Opens QuestManager context
            await handler.start_up()   # Connects to STT
            await _run_route(websocket, handler)
```

`_run_route()` performs a health check, then launches **three concurrent tasks** using `asyncio.TaskGroup`:

```python
async def _run_route(websocket, handler):
    health = await get_health()
    if not health.ok:
        await websocket.close(...)
        return

    async with asyncio.TaskGroup() as tg:
        tg.create_task(receive_loop(...))          # Reads from WebSocket
        tg.create_task(emit_loop(...))             # Writes to WebSocket
        tg.create_task(handler.quest_manager.wait()) # Bubbles up Quest exceptions
        tg.create_task(debug_running_tasks())       # Periodic debug logging
```

If **any** task throws an exception, the `TaskGroup` cancels all others. This is how errors in STT/TTS/LLM tear down the whole connection cleanly.

### `receive_loop()` — Reading User Input

Reads WebSocket messages, decodes them, and routes them:

```
WebSocket Text Message
    │
    ▼
Parse JSON → validate as ClientEvent (via Pydantic TypeAdapter)
    │
    ├── InputAudioBufferAppend:
    │     Decode base64 → Opus bytes → PCM via sphn.OpusStreamReader
    │     Wait for first valid Opus packet (bit flag check: opus_bytes[5] & 2)
    │     Call handler.receive((SAMPLE_RATE, pcm[np.newaxis, :]))
    │
    ├── SessionUpdate:
    │     Update handler instructions/voice
    │     Send SessionUpdated back
    │
    └── Other: Log and ignore
```

**Important detail:** The Opus decoder maintains state between calls. The `wait_for_first_opus` flag ensures we skip stale OGG packets from a previous connection (the browser sometimes resends old buffered data on reconnect).

### `emit_loop()` — Sending Events to Browser

Reads from two sources and sends to the WebSocket:

```
Priority 1: emit_queue (explicit events from receive_loop — e.g., SessionUpdated)
    │
    ▼
Priority 2: handler.emit() — polls the handler's output_queue
    │
    ├── AdditionalOutputs → wraps as UnmuteAdditionalOutputs (debug data)
    ├── CloseStream → closes WebSocket gracefully
    ├── ServerEvent → sends as JSON directly
    └── (SAMPLE_RATE, audio) → encodes to Opus via sphn.OpusStreamWriter
                                → wraps as ResponseAudioDelta (base64 Opus)
```

The emit loop also records all outgoing events via `handler.recorder`.

### Error Handling

Errors are caught and sent as `ora.Error` events before closing the WebSocket:

| Error | User sees |
|-------|-----------|
| `MissingServiceAtCapacity` | "Too many people are connected to service 'X'. Please try again later." |
| `MissingServiceTimeout` | "Service 'X' timed out. Please try again later." |
| `WebSocketClosedError` | (silent — client disconnected) |
| Any other exception | "Internal server error :( Complain to Kyutai" |

CORS headers are injected even into error responses to prevent confusing browser CORS errors that mask the real issue.

---

## The Handler — `unmute_handler.py`

`UnmuteHandler` is the **brain of the system**. It extends FastRTC's `AsyncStreamHandler` and orchestrates all three AI services.

### Initialization

```python
class UnmuteHandler(AsyncStreamHandler):
    def __init__(self):
        super().__init__(
            input_sample_rate=24000,
            output_frame_size=480,     # IMPORTANT: higher values = choppy audio
            output_sample_rate=24000,
        )
        self.n_samples_received = 0        # Audio clock
        self.output_queue = asyncio.Queue() # All outputs go here
        self.chatbot = Chatbot()           # Conversation state machine
        self.openai_client = get_openai_client()
        self.quest_manager = QuestManager()
        self.turn_transition_lock = asyncio.Lock()
        # ... debug dicts, timers, etc.
```

### The Audio Clock

Instead of using `time.time()`, the handler uses **received audio samples** as a clock:

```python
def audio_received_sec(self) -> float:
    return self.n_samples_received / self.input_sample_rate
```

This means timing is tied to the audio stream, not wall-clock time. If audio processing stalls, the clock stalls too — which is actually desirable because all the timing logic (pause detection, long silence, etc.) should be relative to what the STT has actually processed.

### `receive()` — The Main Audio Processing Loop

Called once per audio frame from the browser. This is the heart of everything:

```python
async def receive(self, frame: tuple[int, np.ndarray]):
    stt = self.stt                    # Get current STT instance
    array = frame[1][0]               # Extract mono audio
    self.n_samples_received += len(array)

    # Always send audio to STT (even during flush!)
    await stt.send_audio(array)
    
    if self.stt_end_of_flush_time is None:
        # ══════════ NORMAL MODE ══════════
        await self.detect_long_silence()
        
        if self.determine_pause():
            # Pause detected! Start flushing STT → then generate response
            # (See UNDERSTAND.md for pause detection details)
            ...
        elif (bot_speaking and pause_prediction < 0.4 and time > 3s):
            # VAD-based interruption
            await self.interrupt_bot()
    else:
        # ══════════ FLUSH MODE ══════════
        if stt.current_time > self.stt_end_of_flush_time:
            # Flush complete → generate LLM response
            await self._generate_response()
```

There's also special handling for:
- **`TTS_DEBUGGING_TEXT`** — if set, ignores the mic and sends a fixed string to the LLM (for testing TTS latency)
- **`AUDIO_INPUT_OVERRIDE`** — replaces mic input with audio from a file (for reproducible testing)
- **Initial response** — on the very first frame after instructions are set, it triggers the first LLM response (the bot's greeting)

> **📖 For how pause detection works in detail, see [UNDERSTAND.md](./UNDERSTAND.md).**

### `emit()` — Output Polling

FastRTC calls this to get the next output. It drains the `output_queue`:

```python
async def emit(self):
    output = await wait_for_item(self.output_queue)
    if output is not None:
        return output
    else:
        # If idle for >1 second, send a debug update
        if self.last_additional_output_update < self.audio_received_sec() - 1:
            return self.get_gradio_update()
```

### `_generate_response()` — Kicking off the LLM

This is a two-step process:

1. **Immediate:** Add an empty assistant message to mark state as `"bot_speaking"`
2. **Async:** Create a Quest that runs `_generate_response_task()`

```python
async def _generate_response(self):
    await self.add_chat_message_delta("", "assistant")  # state → bot_speaking
    quest = Quest.from_run_step("llm", self._generate_response_task)
    await self.quest_manager.add(quest)
```

### `_generate_response_task()` — The Full LLM→TTS Pipeline

This runs as an async Quest. The flow:

```
1. Send ResponseCreated event (notifies frontend)
2. Start TTS WebSocket connection (via start_up_tts)
3. Create VLLMStream (OpenAI-compatible client)
4. Stream LLM tokens:
    for each word from LLM:
        → Send UnmuteResponseTextDeltaReady to frontend
        → Send word to TTS server
        → Check for interruption (has a new message been added?)
5. Send ResponseTextDone with full text
6. Send TTSClientEosMessage to TTS (end of stream)
```

**Key detail:** The LLM response is re-chunked into whole words via `rechunk_to_words()` before being sent to TTS, because the TTS mispronounces split words.

### `_stt_loop()` — Processing STT Output

Runs as a concurrent Quest. Iterates over words from the STT:

```python
async for data in stt:
    # Skip markers
    # Send transcription delta to frontend
    
    if data.text == "":
        continue  # Skip empty first message
    
    if conversation_state == "bot_speaking":
        await self.interrupt_bot()  # STT-word interruption!
    
    is_new_message = await self.add_chat_message_delta(data.text, "user")
    if is_new_message:
        stt.pause_prediction.value = 0.0  # Reset: don't pause after first word
```

### `_tts_loop()` — Processing TTS Output

Runs as a concurrent Quest. Iterates over audio/text from TTS:

```python
async for message in tts:
    if len(chat_history) > generating_message_i:
        break  # Interrupted!

    if isinstance(message, TTSAudioMessage):
        # Push PCM audio to output_queue
        await output_queue.put((SAMPLE_RATE, audio))

    elif isinstance(message, TTSTextMessage):
        # Push text delta to frontend + update chat history
        await output_queue.put(ora.ResponseTextDelta(delta=...))
        await self.add_chat_message_delta(text, "assistant")
```

After TTS finishes:
1. Push silence frame to flush Opus state
2. Send `ResponseAudioDone`
3. Add empty user message → state goes to `"waiting_for_user"`
4. Check if the bot said "Bye!" → if so, close the stream
5. Reset the `waiting_for_user_start_time` for long-silence detection

### `interrupt_bot()` — Cutting Off The Bot

When the user interrupts:

```python
async def interrupt_bot(self):
    # 1. Append "—" (em-dash) to the assistant's message
    await self.add_chat_message_delta(INTERRUPTION_CHAR, "assistant")
    
    # 2. Clear FastRTC's internal audio buffer
    self._clear_queue()
    
    # 3. Replace output_queue (old TTS loop now writes to a dead queue)
    self.output_queue = asyncio.Queue()
    
    # 4. Push silence to flush Opus + send interruption event
    await self.output_queue.put(ora.UnmuteInterruptedByVAD())
    
    # 5. Cancel TTS and LLM Quests
    await self.quest_manager.remove("tts")
    await self.quest_manager.remove("llm")
```

The trick with replacing `output_queue` is clever — the old TTS loop captured a reference to the old queue (`output_queue = self.output_queue` at the start of `_tts_loop`), so it harmlessly writes into a queue nobody reads.

### `detect_long_silence()` — Handling Silence

If the user hasn't spoken for 7 seconds after the bot finished:

```python
async def detect_long_silence(self):
    if (
        conversation_state == "waiting_for_user"
        and (audio_received_sec - waiting_for_user_start_time) > 7.0
    ):
        await self.add_chat_message_delta("...", "user")
```

Adding `"..."` changes state to `"user_speaking"`. Since there's no actual speech, `pause_prediction` stays high → `determine_pause()` returns True → the bot responds to the silence (the system prompt tells it how to handle `"..."` messages).

### `update_session()` — Client Settings

Called when the frontend sends a `SessionUpdate`:

```python
async def update_session(self, session):
    if session.instructions:
        self.chatbot.set_instructions(session.instructions)
    if session.voice:
        self.tts_voice = session.voice
    if not session.allow_recording:
        # Delete the recording file
        await self.recorder.shutdown(keep_recording=False)
        self.recorder = None
```

---

## The Quest Manager — Async Lifecycle

`QuestManager` (`quest_manager.py`) solves a common async problem: **how do you manage the lifecycle of concurrent tasks that depend on resources?**

### The Problem

The handler needs to:
1. Connect to STT → run a loop → close connection when done (or on error)
2. Connect to TTS → run a loop → close connection when done (or interrupted)
3. Start LLM generation → stream tokens → clean up on interruption

Each of these has init/run/close phases and can be cancelled at any time.

### The Solution: `Quest`

A Quest wraps three callbacks:

```python
class Quest[T]:
    def __init__(self, name, init, run, close):
        self.init = init    # () -> T          (e.g., connect to TTS)
        self.run = run      # (T) -> None      (e.g., TTS streaming loop)
        self.close = close  # (T) -> None      (e.g., close TTS connection)
```

**Lifecycle:**

```
Quest created → init() runs → sets self._data → run() starts → (eventually) close()
                     │                                               │
                     ▼                                               ▼
              await quest.get()                            Happens automatically
              returns T (blocks until                      on cancel or when
              init completes)                              QuestManager exits
```

**Key behaviors:**
- If you `add()` a Quest with the same name as an existing one, the old one is **cancelled** first
- `Quest.from_run_step()` creates a simplified Quest with no init/close (used for LLM)
- `quest.get()` returns a `Future[T]` — you can `await` it to wait for init, or call `get_nowait()` to check if ready

### QuestManager

```python
class QuestManager:
    def __init__(self):
        self.quests: dict[str, Quest] = {}
```

Used as an async context manager. When the context exits (`__aexit__`), all Quests are cancelled and closed — even if they raise `MissingServiceAtCapacity` or `MissingServiceTimeout` (those are swallowed).

The `wait()` method returns a Future that is set when any Quest raises an unexpected exception, allowing errors to propagate up to the `TaskGroup` in `_run_route()`.

### Real Usage in the Handler

```python
# STT Quest — long-lived, runs for the entire session
Quest("stt", init=connect_to_stt, run=stt_loop, close=stt_shutdown)

# TTS Quest — per-response, replaced on each new response
Quest("tts", init=connect_to_tts, run=tts_loop, close=tts_shutdown)

# LLM Quest — per-response, no explicit cleanup needed
Quest.from_run_step("llm", generate_response_task)
```

When the user interrupts the bot:
```python
await self.quest_manager.remove("tts")   # Cancels TTS streaming + closes connection
await self.quest_manager.remove("llm")   # Cancels LLM generation
# STT Quest stays alive — it's needed for the next turn
```

---

## STT Pipeline — Speech to Text

### `SpeechToText` class (`unmute/stt/speech_to_text.py`)

Connects to the DSM-ASR server via WebSocket, sends raw audio, and receives transcription + pause probability signals.

**Protocol:** Uses **msgpack** for binary serialization (not JSON) for efficiency.

### Message Types from STT Server

| Message | Fields | Meaning |
|---------|--------|---------|
| `STTReadyMessage` | `type: "Ready"` | Server is ready to accept audio |
| `STTStepMessage` | `step_idx`, `prs: list[float]` | One frame (80ms) processed. `prs[2]` = pause probability |
| `STTWordMessage` | `text`, `start_time` | A word was transcribed |
| `STTEndWordMessage` | `stop_time` | End-of-word timing info |
| `STTMarkerMessage` | `id` | Echo of a marker we sent (for pipeline timing) |
| `STTErrorMessage` | `message` | Server error (usually at capacity) |

### Connection Lifecycle

```python
async def start_up(self):
    self.websocket = await websockets.connect(
        self.stt_instance + "/api/asr-streaming",
        additional_headers={"kyutai-api-key": "public_token"}
    )
    message = await self.websocket.recv()  # Wait for Ready or Error
```

### Sending Audio

```python
async def send_audio(self, audio: np.ndarray):
    self.sent_samples += len(audio)
    await self._send({"type": "Audio", "pcm": audio.tolist()})
```

Audio is sent as a list of floats via msgpack. `send_marker()` can also send marker IDs for pipeline latency measurement.

### Receiving (`__aiter__`)

The `__aiter__` method is the core loop, yielding only words and markers:

```python
async def __aiter__(self):
    n_steps_to_wait = 12  # Skip first 12 frames (~960ms warm-up)

    async for message_bytes in self.websocket:
        message = decode(message_bytes)
        
        match message:
            case STTStepMessage():
                self.current_time += FRAME_TIME_SEC  # Advance time
                if n_steps_to_wait > 0:
                    n_steps_to_wait -= 1  # Skip warm-up
                else:
                    self.pause_prediction.update(dt=0.08, new_value=message.prs[2])
                    
            case STTWordMessage():
                self.received_words += 1
                yield message  # This is what _stt_loop() receives
                
            case STTMarkerMessage():
                yield message
```

`STTStepMessage` updates the pause prediction EMA but is **not yielded** — it's consumed internally. Only words and markers are passed to the handler.

> **📖 For how pause prediction and the EMA work, see [UNDERSTAND.md](./UNDERSTAND.md).**

### `DummySpeechToText` (`dummy_speech_to_text.py`)

A testing stub that never sends words and keeps `pause_prediction` permanently at 1.0. Useful for benchmarking TTS latency without a real STT server.

---

## LLM Pipeline — The Brain

### `Chatbot` class (`unmute/llm/chatbot.py`)

Manages the conversation history and state machine.

#### Conversation State

Determined by the last message in `chat_history`:

```python
def conversation_state(self) -> ConversationState:
    last = self.chat_history[-1]
    
    if last["role"] == "assistant":
        return "bot_speaking"
    elif last["role"] == "user":
        if last["content"].strip() != "":
            return "user_speaking"
        else:
            return "waiting_for_user"
    elif last["role"] == "system":
        return "waiting_for_user"
```

**State transitions:**

```
             User says word              Bot says "Bye!"
  ┌──────► waiting_for_user ──────────► (connection closed)
  │            │                              ▲
  │            │ add_chat_message_delta        │ add_chat_message_delta
  │            │ ("word", "user")              │ ("", "user")
  │            ▼                              │
  │       user_speaking ──────────────────────┤
  │            │                              │
  │            │ determine_pause() → True     │
  │            │ → _generate_response()       │
  │            │ → add_chat_message_delta     │
  │            │   ("", "assistant")          │
  │            ▼                              │
  │       bot_speaking ───────────────────────┘
  │            │
  │            │ interrupt_bot()
  │            │ → add_chat_message_delta("—", "assistant")
  │            │ → add_chat_message_delta("", "user")
  │            ▼
  └──────── (back to waiting_for_user or user_speaking)
```

#### `add_chat_message_delta()` — Building Messages Incrementally

Words arrive one at a time from STT. This method appends them intelligently:

```python
async def add_chat_message_delta(self, delta, role):
    if chat_history[-1]["role"] != role:
        # New message
        chat_history.append({"role": role, "content": delta})
        return True  # is_new_message = True
    else:
        # Append to existing message, auto-inserting spaces
        chat_history[-1]["content"] += " " + delta  # (simplified)
        return last_was_empty  # True only if message was ""
```

#### `preprocessed_messages()` — Cleaning Up for the LLM

Before sending to the LLM, the chat history is cleaned:

```python
def preprocessed_messages(self):
    messages = preprocess_messages_for_llm(self.chat_history)
    return messages
```

`preprocess_messages_for_llm()` (in `llm_utils.py`) does:
1. **Removes empty/interruption-only messages** — messages that are just "—" are dropped
2. **Strips interruption characters** — trailing "—" removed from interrupted responses
3. **Merges consecutive same-role messages** — sometimes happens due to race conditions
4. **Adds dummy user message** — if chat is just `[system, assistant]`, inserts `{"role": "user", "content": "Hello."}` because some LLMs (like Gemma) choke without a user message before the assistant
5. **Removes silence markers** — if user message starts with `"..."` but has real content after, strips the `"..."`

### `VLLMStream` (`llm_utils.py`)

Wraps the OpenAI client for streaming LLM responses:

```python
class VLLMStream:
    def __init__(self, client: AsyncOpenAI, temperature: float):
        self.model = autoselect_model()  # Auto-detects model from vLLM
    
    async def chat_completion(self, messages):
        stream = await self.client.chat.completions.create(
            model=self.model, messages=messages, stream=True,
            temperature=self.temperature,
        )
        async for chunk in stream:
            yield chunk.choices[0].delta.content
```

**Model auto-selection:** If `KYUTAI_LLM_MODEL` env var is not set, `autoselect_model()` queries the LLM server's `/v1/models` endpoint. If exactly one model is available, it uses that. Otherwise, it raises an error.

**Temperature:** First message uses 0.7 (more creative greeting), subsequent messages use 0.3 (more focused responses).

### `rechunk_to_words()` — Word Boundary Re-chunking

The LLM streams tokens that might not align with word boundaries. This generator re-chunks them:

```
Input:  "He" "llo," " how" " are" " you"
Output: "Hello,", " how", " are", " you"
```

Spaces are carried with the **next** word. Multiple whitespace characters are collapsed to a single space. This is critical because the TTS mispronounces split words.

### System Prompts (`system_prompt.py`)

Six instruction types, all producing a system prompt from a template:

| Type | Purpose |
|------|---------|
| `ConstantInstructions` | Default — generic conversational prompt |
| `SmalltalkInstructions` | Casual chat with time awareness and conversation starters |
| `GuessAnimalInstructions` | 20-questions animal guessing game |
| `QuizShowInstructions` | Quiz show host personality (5 random questions) |
| `NewsInstructions` | Discuss tech news from The Verge (via NewsAPI) |
| `UnmuteExplanationInstructions` | Explain how Unmute works to the user |

The template includes rules for:
- Spoken format (no markdown, no emojis)
- Language switching (English/French only, due to TTS)
- Handling transcription errors gracefully
- "WHO ARE YOU" — explains the modular STT→LLM→TTS architecture
- Silence handling — what to do with `"..."` messages
- Conversation ending — say "Bye!" to close the stream

### NewsAPI (`newsapi.py`)

Fetches headlines from The Verge via [newsapi.org](https://newsapi.org). Results are cached for 4 hours (via `LocalCache` or `RedisCache`). If the API key isn't configured, the news mode falls back to smalltalk.

---

## TTS Pipeline — Text to Speech

### `TextToSpeech` class (`unmute/tts/text_to_speech.py`)

Connects to the Kyutai TTS server via WebSocket, sends text words, and receives audio + word-timing data.

### Connection Setup

```python
async def start_up(self):
    url = self.tts_instance + "/api/tts_streaming" + query_params
    self.websocket = await websockets.connect(url, additional_headers=HEADERS)
    
    # Send voice embedding for custom voices
    if self.voice.startswith("custom:"):
        voice_embedding = voice_embeddings_cache.get(self.voice)
        await self.websocket.send(voice_embedding)
    
    # Wait for Ready message (may need to skip stale packets)
    for _ in range(10):
        message = await self.websocket.recv()
        if isinstance(message, TTSReadyMessage):
            return
```

**Query parameters** include voice name, temperature, top_k, cfg_alpha (1.5), and format (`PcmMessagePack`).

### Sending Text

```python
async def send(self, message: str | TTSClientMessage):
    if isinstance(message, str):
        message = TTSClientTextMessage(text=prepare_text_for_tts(message))
    await self.websocket.send(msgpack.packb(message.model_dump()))
```

`prepare_text_for_tts()` strips unpronounceable characters (`*`, `_`, backtick), normalizes smart quotes, and removes ` : ` patterns.

### Receiving Audio — The RealtimeQueue

The TTS generates audio **faster than real-time** — it produces the entire response's audio before the user has finished hearing the first word. If released immediately, the text captions and audio would desynchronize.

Solution: **`RealtimeQueue`** — a heap-based priority queue that releases items at their scheduled timestamps:

```python
AUDIO_BUFFER_SEC = FRAME_TIME_SEC * 4  # Buffer 4 frames ahead

async def __aiter__(self):
    async for message in self.websocket:
        if isinstance(message, TTSAudioMessage):
            # Don't yield immediately! Queue with a timestamp.
            output_queue.put(
                message,
                self.received_samples / SAMPLE_RATE - AUDIO_BUFFER_SEC
            )
            self.received_samples += len(message.pcm)

        elif isinstance(message, TTSTextMessage):
            # Queue text at its spoken timestamp
            output_queue.put(message, message.start_s)

        # Release items whose timestamps have passed
        for _, message in output_queue.get_nowait():
            yield message
```

**Why text is delayed too:** Text messages from the TTS have `start_s` and `stop_s` timestamps — when the word is *spoken* in the audio. These are "from the future" relative to real-time playback, so text is also queued and released at the right moment for lip-sync with the audio.

### Message Types

**Client → TTS Server:**
| Message | Purpose |
|---------|---------|
| `TTSClientTextMessage` | "Turn this text into speech" |
| `TTSClientVoiceMessage` | Voice embedding for cloning |
| `TTSClientEosMessage` | "I'm done sending text" |

**TTS Server → Client:**
| Message | Purpose |
|---------|---------|
| `TTSReadyMessage` | "Ready to accept text" |
| `TTSAudioMessage` | PCM audio chunk (float32) |
| `TTSTextMessage` | Word with timing (`text`, `start_s`, `stop_s`) |
| `TTSErrorMessage` | Error (usually at capacity) |

---

## The Event Protocol

`openai_realtime_api_events.py` defines all messages exchanged between the WebSocket server and the browser frontend. The protocol is **inspired by OpenAI's Realtime API** but with custom extensions.

### BaseEvent

All events inherit from `BaseEvent[Literal["event.type"]]`:

```python
class BaseEvent(BaseModel, Generic[T]):
    type: T   # Auto-set from the Literal type parameter
    event_id: str  # Random unique ID like "event_BJhGUIswO2u7vA2Cxw3Jy"
```

The `type` field is automatically set by a model validator — you never pass it explicitly.

### Client → Server Events

| Event | Purpose |
|-------|---------|
| `SessionUpdate` | Set instructions, voice, recording consent |
| `InputAudioBufferAppend` | Base64 Opus audio from microphone |

### Server → Client Events

| Event | Purpose |
|-------|---------|
| `Error` | Error with type (warning/fatal), code, message |
| `SessionUpdated` | Acknowledges session config change |
| `InputAudioBufferSpeechStarted` | STT detected user started speaking |
| `InputAudioBufferSpeechStopped` | Pause detected — bot will respond |
| `ResponseCreated` | Bot started generating (includes chat history + voice) |
| `ResponseTextDelta` | Word from TTS (displayed as captions) |
| `ResponseTextDone` | Full text of the response |
| `ResponseAudioDelta` | Base64 Opus audio chunk |
| `ResponseAudioDone` | Audio stream complete |
| `UnmuteAdditionalOutputs` | Debug data (chat history, timing, etc.) |
| `UnmuteResponseTextDeltaReady` | Word ready for TTS (before audio is generated) |
| `UnmuteResponseAudioDeltaReady` | Audio chunk ready (before realtime queuing) |
| `UnmuteInterruptedByVAD` | VAD interrupted the bot |

The `Unmute*` events are custom extensions not in the OpenAI spec.

---

## Service Discovery

`service_discovery.py` handles finding and connecting to STT/TTS/LLM servers.

### DNS-based Discovery

```python
SERVICES = {
    "tts": TTS_SERVER,    # e.g., "ws://localhost:8089"
    "stt": STT_SERVER,    # e.g., "ws://localhost:8090"
    "llm": LLM_SERVER,    # e.g., "http://localhost:8091"
}

async def get_instances(service_name):
    url = SERVICES[service_name]
    hostname = extract_hostname(url)
    ips = await dns_resolve(hostname)  # socket.gethostbyname_ex
    random.shuffle(ips)               # Load balancing
    return [f"{protocol}://{ip}:{port}" for ip in ips]
```

In production (Docker Swarm), a service hostname like `tts` resolves to multiple IPs — one per replica. The function shuffles them for crude load balancing.

### `find_instance()` — Connecting with Retry

```python
async def find_instance(service_name, client_factory, timeout_sec=0.5, max_trials=3):
    instances = await get_instances(service_name)
    for instance in instances:
        client = client_factory(instance)
        try:
            async with asyncio.timeout(timeout_sec):
                await client.start_up()
        except MissingServiceAtCapacity:
            continue   # This one is full, try next
        except TimeoutError:
            continue   # This one is slow, try next
        except Exception:
            continue   # Something else went wrong, try next
        
        return client  # Found a working instance!
    
    raise MissingServiceAtCapacity(service_name)
```

**Important:** For TTS, there are retry loops with exponential backoff *on top of* this — see `start_up_tts()` in the handler, which retries up to 5 times with growing sleeps.

### DNS Caching

DNS lookups are cached for 0.5 seconds using `async_ttl_cached()`:

```python
@partial(async_ttl_cached, ttl_sec=0.5)
async def _resolve(hostname):
    *_, ipaddrlist = await asyncio.to_thread(socket.gethostbyname_ex, hostname)
    return ipaddrlist
```

---

## Voice System

### Voice List (`voices.py`)

Voices are defined in `voices.yaml` at the project root. Each voice has:

```yaml
- name: "Friendly Female"
  good: true              # Only "good" voices shown to users
  source:
    source_type: freesound # or "file"
    freesound_id: 12345
    path_on_server: "friendly_female.wav"
  instructions:           # Optional custom personality
    type: smalltalk
```

The `VoiceList` class loads this YAML file and provides:
- `voices()` REST endpoint → returns good voices
- `upload_to_server()` → downloads from Freesound, optionally uses Adobe podcast-enhanced versions, uploads to dev/prod servers

### Voice Cloning (`voice_cloning.py`)

User-uploaded voices go through a cloning pipeline:

```
User uploads WAV → POST /v1/voices → clone_voice()
    → POST to voice cloning server (/api/voice)
    → Receives msgpack voice embedding
    → Stores in cache with key "custom:UUID"
    → Returns voice name "custom:UUID"
```

The voice embedding is cached for 1 hour (via `LocalCache` or `RedisCache`). When TTS starts up, it sends the cached embedding to the TTS server as the first message.

### Voice Donation (`voice_donation.py`)

Users can donate their voice:

1. **GET `/v1/voice-donation`** → returns a verification text (consent prefix + two random sentences) and a unique ID
2. User records themselves reading the text
3. **POST `/v1/voice-donation`** → submits the recording with:
   - Email (for withdrawal requests)
   - Nickname
   - Verification ID (must match, and must be within 5 minutes)
   - License: CC0 only
4. Audio saved to `VOICE_DONATION_DIR` as `{verification_id}.wav`
5. Metadata saved as `{verification_id}.json`

---

## Recording & Debugging Tools

### `Recorder` (`recorder.py`)

Records all WebSocket events to a JSONL file for debugging:

```python
class Recorder:
    def __init__(self, recordings_dir):
        self.path = recordings_dir / "2026-04-08_14-30-00_a1b2.jsonl"
    
    async def add_event(self, sender: "client"|"server", data: Event):
        await self.opened_file.write(
            RecorderEvent(timestamp_wall=..., event_sender=sender, data=data)
            .model_dump_json() + "\n"
        )
```

**Privacy:** User audio (`InputAudioBufferAppend`) is NOT recorded. Instead, `UnmuteInputAudioBufferAppendAnonymized` records only the *number of samples* (for timing reconstruction).

**Consent:** Recording starts immediately but can be deleted retroactively if the user doesn't consent:
```python
await self.recorder.shutdown(keep_recording=False)  # Deletes the file
```

### `process_recording.py`

A standalone CLI script that converts recorded JSONL files into time-aligned visualization data:

```bash
python unmute/process_recording.py recording.msgpack output.json --audio-output-path combined.ogg
```

It reconstructs:
- Per-step waveform amplitude (RMS)
- Word timing (when words were transcribed vs when they were spoken)
- TTS audio timing (when audio was generated vs when it was played)
- It also extracts user + assistant audio as a two-channel audio file

### `AudioStreamSaver` (`audio_stream_saver.py`)

A debug utility that collects audio chunks and saves them to WAV files at regular intervals:

```python
saver = AudioStreamSaver(interval_sec=1.0, max_saves=1)
saver.add(audio_chunk)  # Accumulates until 1 second, then saves
```

---

## Configuration & Constants

### `kyutai_constants.py`

All server URLs and audio parameters:

| Constant | Value | Source |
|----------|-------|--------|
| `STT_SERVER` | `ws://localhost:8090` | `KYUTAI_STT_URL` env var |
| `TTS_SERVER` | `ws://localhost:8089` | `KYUTAI_TTS_URL` env var |
| `LLM_SERVER` | `http://localhost:8091` | `KYUTAI_LLM_URL` env var |
| `KYUTAI_LLM_MODEL` | (auto-detect) | `KYUTAI_LLM_MODEL` env var |
| `KYUTAI_LLM_API_KEY` | (none) | `KYUTAI_LLM_API_KEY` env var |
| `VOICE_CLONING_SERVER` | `http://localhost:8092` | `KYUTAI_VOICE_CLONING_URL` env var |
| `REDIS_SERVER` | (none, uses local dict) | `KYUTAI_REDIS_URL` env var |
| `SAMPLE_RATE` | 24,000 Hz | Hardcoded |
| `SAMPLES_PER_FRAME` | 1,920 | Hardcoded (80ms frames) |
| `FRAME_TIME_SEC` | 0.08s | Computed |
| `STT_DELAY_SEC` | 0.5s | Hardcoded |
| `RECORDINGS_DIR` | (none) | `KYUTAI_RECORDINGS_DIR` env var |
| `MAX_VOICE_FILE_SIZE_MB` | 4 MB | Hardcoded |

### Handler Constants (`unmute_handler.py`)

| Constant | Value | Purpose |
|----------|-------|---------|
| `USER_SILENCE_TIMEOUT` | 7.0s | Seconds before bot responds to silence |
| `FIRST_MESSAGE_TEMPERATURE` | 0.7 | LLM temperature for greeting |
| `FURTHER_MESSAGES_TEMPERATURE` | 0.3 | LLM temperature for responses |
| `UNINTERRUPTIBLE_BY_VAD_TIME_SEC` | 3s | VAD interruption grace period (echo cancellation) |
| `TTS_DEBUGGING_TEXT` | None | If set, bypasses mic and sends this string |
| `AUDIO_INPUT_OVERRIDE` | None | If set, replaces mic with audio file |

---

## Supporting Modules

### `cache.py` — Dual-backend Cache

Two implementations behind the same interface:

| Backend | When | How |
|---------|------|-----|
| `LocalCache` | `KYUTAI_REDIS_URL` not set | Python dict with TTL-based expiry |
| `RedisCache` | `KYUTAI_REDIS_URL` is set | Redis with `SETEX` for TTL |

Used for voice embeddings (1h TTL), news cache (4h TTL), and voice donation verification (1h TTL).

### `timer.py` — Stopwatch Utilities

**`Stopwatch`** — measures elapsed time:
```python
sw = Stopwatch()           # Starts immediately
sw = Stopwatch(autostart=False)  # Start later with .start_if_not_started()
elapsed = sw.time()        # Seconds since start
elapsed = sw.stop()        # Stops and returns elapsed (returns None if re-called)
```

**`PhasesStopwatch`** — tracks multiple phases of a pipeline:
```python
ps = PhasesStopwatch(["stt", "llm", "tts"])
ps.time_phase_if_not_started("stt")   # Records current time
ps.time_phase_if_not_started("llm")
ps.phase_dict()  # {"stt": 1.2, "llm": 1.5, "tts": None}
```

**Important:** Both use `asyncio.get_event_loop().time()`, not `time.time()`. This is the event loop's monotonic clock.

### `exceptions.py` — Custom Exceptions

```python
MissingServiceAtCapacity  # Service is running but full
MissingServiceTimeout     # Service didn't respond in time
WebSocketClosedError      # Client disconnected (not an error, just cleanup)

make_ora_error(type, message)  # Constructor for ora.Error events
```

### `websocket_utils.py` — URL Conversion

```python
http_to_ws("http://localhost:8090")  → "ws://localhost:8090"
http_to_ws("https://api.example.com") → "wss://api.example.com"
ws_to_http("ws://localhost:8090")    → "http://localhost:8090"
```

### `webrtc_utils.py` — Cloudflare TURN Credentials

Fetches ICE server credentials from Cloudflare's Calls API for WebRTC NAT traversal. Used only in Gradio mode.

### `metrics.py` — Prometheus Monitoring

Comprehensive Prometheus metrics for all services:

**Counters:**
- Sessions, service misses, interruptions, hard errors, voice donations

**Gauges:**
- Active sessions (per service)

**Histograms (with custom bins):**
- Session/audio duration
- TTFT (Time to First Token) for STT, TTS, LLM
- Request/reply length (in words)
- Ping time and find time for service discovery
- Generation duration

Exposed via `/metrics` endpoint (auto-configured by `prometheus-fastapi-instrumentator`).

---

## Full Request Lifecycle — Start to Finish

Here's everything that happens from the user opening the page to hearing the bot speak:

### Phase 1: Connection Setup

```
1. Browser connects to ws://server/v1/realtime
2. Server accepts with subprotocol "realtime"
3. UnmuteHandler created
4. QuestManager context enters
5. STT Quest created: connects to STT server, starts _stt_loop()
6. TaskGroup spawns: receive_loop, emit_loop, quest_manager.wait()
```

### Phase 2: Session Configuration

```
7. Browser sends SessionUpdate (instructions, voice, recording consent)
8. Handler sets instructions → system prompt generated
9. Handler sets tts_voice
10. SessionUpdated sent back to browser
```

### Phase 3: Initial Bot Greeting

```
11. First audio frame arrives from browser
12. receive() detects: chat_history has 1 message (system) + instructions set
13. _generate_response() called → empty assistant message added → state = bot_speaking
14. LLM Quest starts: connects to vLLM, streams greeting
15. TTS Quest starts: connects to TTS, receives voice
16. LLM tokens → rechunked to words → sent to TTS
17. TTS produces audio → queued in RealtimeQueue → released at correct time
18. Audio encoded to Opus → sent to browser as ResponseAudioDelta
19. Text captions sent as ResponseTextDelta
20. TTS finishes → ResponseAudioDone → empty user message → state = waiting_for_user
```

### Phase 4: User Speaks

```
21. Audio frames arrive continuously from browser
22. Each frame sent to STT via stt.send_audio()
23. STT processes and returns STTStepMessage (updating pause_prediction)
24. STT returns STTWordMessage when a word is transcribed
25. _stt_loop() adds word to chat history → state = user_speaking
26. First word resets pause_prediction to 0.0
27. InputAudioBufferSpeechStarted sent to browser
```

### Phase 5: Pause Detection & Response

```
28. User stops speaking
29. pause_prediction rises above 0.6
30. determine_pause() returns True
31. InputAudioBufferSpeechStopped sent to browser
32. 8 silence frames pushed to STT (flush)
33. Flush completes → _generate_response() 
34. Back to Phase 3 (but with real user message)
```

### Phase 5b: Interruption (Alternative)

```
28. Bot is speaking, user starts talking
29a. STT produces a word → STT-word interruption (immediate)
29b. pause_prediction < 0.4 and time > 3s → VAD interruption
30. interrupt_bot(): cancel LLM/TTS, clear output, add "—" to assistant message
31. Add empty user message → state = waiting_for_user or user_speaking
32. Back to Phase 4
```

### Phase 6: Session End

```
33a. Bot says "Bye!" → CloseStream sent → WebSocket closed
33b. User disconnects → WebSocketClosedError → cleanup
33c. 7 seconds of silence → "..." message → bot responds (go to Phase 5)
34. QuestManager __aexit__: cancel all Quests, close all connections
35. Recorder flushed and closed
```

---

## File-by-File Reference Table

| File | Lines | Purpose |
|------|-------|---------|
| **Entry Points** | | |
| `main_websocket.py` | 623 | FastAPI WebSocket server (production) |
| `main_gradio.py` | 68 | Gradio WebRTC UI (development) |
| **Core** | | |
| `unmute_handler.py` | 652 | Main orchestrator — audio→STT→LLM→TTS pipeline |
| `quest_manager.py` | 179 | Async lifecycle management (init/run/close pattern) |
| `openai_realtime_api_events.py` | 203 | WebSocket protocol event definitions |
| **STT** | | |
| `stt/speech_to_text.py` | 229 | STT WebSocket client |
| `stt/exponential_moving_average.py` | 38 | EMA filter for pause probability |
| `stt/dummy_speech_to_text.py` | 66 | Testing stub |
| **LLM** | | |
| `llm/chatbot.py` | 122 | Conversation state machine + chat history |
| `llm/llm_utils.py` | 166 | OpenAI/vLLM client, message preprocessing |
| `llm/system_prompt.py` | 392 | System prompt templates (6 instruction types) |
| `llm/newsapi.py` | 91 | The Verge news fetcher |
| `llm/quiz_show_questions.py` | ~200 | Trivia questions database |
| **TTS** | | |
| `tts/text_to_speech.py` | 356 | TTS WebSocket client + realtime queuing |
| `tts/realtime_queue.py` | 83 | Timestamp-based priority queue |
| `tts/voices.py` | 242 | Voice catalog management |
| `tts/voice_cloning.py` | 34 | Voice cloning client |
| `tts/voice_donation.py` | 140 | Voice donation workflow |
| **Infrastructure** | | |
| `service_discovery.py` | 148 | DNS-based service finding with retry |
| `cache.py` | 112 | Local dict or Redis cache |
| `metrics.py` | 129 | Prometheus metrics |
| `recorder.py` | 75 | Event recording (JSONL) |
| `timer.py` | 98 | Stopwatch utilities |
| `exceptions.py` | 27 | Custom exceptions |
| `websocket_utils.py` | 42 | HTTP↔WS URL conversion |
| `webrtc_utils.py` | 26 | Cloudflare TURN credentials |
| **Debug/Tools** | | |
| `process_recording.py` | 450 | Recording → visualization converter |
| `audio_stream_saver.py` | 58 | Debug audio dumper |
| `audio_input_override.py` | 35 | Replace mic with file input |
