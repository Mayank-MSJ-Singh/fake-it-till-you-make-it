"""
Test: Gemma → Pocket TTS (sentence streaming)

Pipeline:
  Gemma streams tokens → buffer into sentences → pocket_tts speaks each sentence
  Audio plays via continuous OutputStream (no gaps between chunks)

  Gemma: "Hello! " → "I can help. " → "What do you need?"
              ↓               ↓                  ↓
        speak("Hello!")  speak("I can...")  speak("What...")
              ↓               ↓                  ↓
         🔊 plays        🔊 plays next      🔊 plays last
"""
import queue
import time
import numpy as np
import sounddevice as sd
from stream2sentence import generate_sentences

from llm.llm_engine import LLMEngine
from tts.tts_engine import TTSEngine
from core.config import LLM_MODEL_ID, TTS_VOICE, SAMPLES_PER_FRAME

SAMPLE_RATE = 24000


# ─────────────────────────────────────────────
# Continuous audio player (no gaps between chunks)
# ─────────────────────────────────────────────
class AudioPlayer:
    """Feeds audio chunks into a continuous sounddevice OutputStream."""

    def __init__(self, sample_rate=SAMPLE_RATE):
        self._q      = queue.Queue()
        self._buf    = np.array([], dtype=np.float32)
        self._stream = sd.OutputStream(
            samplerate=sample_rate,
            channels=1,
            dtype='float32',
            blocksize=SAMPLES_PER_FRAME,
            callback=self._callback,
        )

    def _callback(self, outdata, frames, time_info, status):
        needed  = frames
        written = 0
        while written < needed:
            if len(self._buf) == 0:
                try:
                    chunk = self._q.get_nowait()
                    if chunk is None:
                        outdata[written:] = 0
                        raise sd.CallbackStop()
                    self._buf = chunk.flatten().astype(np.float32)
                except queue.Empty:
                    outdata[written:] = 0
                    return
            to_copy = min(len(self._buf), needed - written)
            outdata[written:written + to_copy, 0] = self._buf[:to_copy]
            self._buf = self._buf[to_copy:]
            written  += to_copy

    def start(self):
        self._stream.start()

    def put(self, audio: np.ndarray):
        self._q.put(audio.astype(np.float32))

    def finish_and_wait(self):
        self._q.put(None)
        while self._stream.active:
            time.sleep(0.05)
        self._stream.stop()
        self._stream.close()


def speak(tts: TTSEngine, text: str, player: AudioPlayer):
    """Stream text through TTS and push audio chunks to the player."""
    for chunk in tts.synthesize_stream(text):
        player.put(chunk)


# ─────────────────────────────────────────────
# Load models
# ─────────────────────────────────────────────
print(f"LLM   : {LLM_MODEL_ID}")
print(f"Voice : {TTS_VOICE}\n")

print("Loading LLM...")
llm = LLMEngine()
llm.load_model()

print("\nLoading Pocket TTS...")
tts = TTSEngine()
tts.load_model()
print(f"Sample rate: {tts.model.sample_rate} Hz\n")

# ─────────────────────────────────────────────
# Chat loop
# ─────────────────────────────────────────────
print("=" * 55)
print("Gemma → Pocket TTS  |  Type 'quit' to exit")
print("=" * 55 + "\n")

history = []

while True:
    user_input = input("[YOU] ")
    if user_input.strip().lower() in ("quit", "exit", "q"):
        break

    history.append({"role": "user", "content": user_input})

    t_start       = time.time()
    t_first_audio = None
    response_parts = []   # mutable so inner generator can append
    n_sentences   = 0

    player = AudioPlayer()
    player.start()

    print("[BOT] ", end="", flush=True)

    def _llm_stream():
        for chunk in llm.generate_stream(history):
            print(chunk, end="", flush=True)
            response_parts.append(chunk)
            yield chunk

    for sentence in generate_sentences(
        _llm_stream(),
        minimum_sentence_length=12,
        minimum_first_fragment_length=10,
        force_first_fragment_after_words=7,
        quick_yield_single_sentence_fragment=True,
        cleanup_text_links=True,
        cleanup_text_emojis=True,
    ):
        if not sentence.strip():
            continue
        if t_first_audio is None:
            t_first_audio = time.time()
        speak(tts, sentence.strip(), player)
        n_sentences += 1

    player.finish_and_wait()

    full_response = "".join(response_parts)

    first_ms = (t_first_audio - t_start) * 1000 if t_first_audio else 0
    print(f"\n  ⏱  First audio: {first_ms:.0f}ms | "
          f"Total: {time.time() - t_start:.2f}s | "
          f"Sentences: {n_sentences}\n")

    history.append({"role": "assistant", "content": full_response.strip()})

print("\n👋 Bye!")