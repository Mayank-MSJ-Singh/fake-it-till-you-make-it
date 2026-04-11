"""
Test: Gemma LLM streaming → Pocket TTS ONNX streaming → Gapless Speaker

Pipeline:
  LLM stream → stream2sentence (smart chunking) → tts.stream(sentence)
  → single persistent OutputStream (NO gaps, NO device open/close per chunk)

stream2sentence triggers TTS after just 10 chars or 7 words — much lower
first-audio latency vs waiting for a full sentence with punctuation.
"""
import sys
import queue
import threading
import time
import numpy as np
import sounddevice as sd
from stream2sentence import generate_sentences

sys.path.insert(0, "pocket-tts-onnx")
from pocket_tts_onnx import PocketTTSOnnx

from llm.llm_engine import LLMEngine
from core.config import LLM_MODEL_ID

VOICE_REF   = "pocket-tts-onnx/reference_sample.wav"
SAMPLE_RATE = 24000
BLOCKSIZE   = 1920   # 80ms — same as ONNX frame size


# ──────────────────────────────────────────────────────────
# Gapless audio player
# ──────────────────────────────────────────────────────────
class GaplessPlayer:
    """Single OutputStream that pulls chunks from a queue.

    No device open/close between chunks = no gaps or pops.
    Call put(audio) to enqueue, finish() to signal done.
    """
    def __init__(self, sample_rate=SAMPLE_RATE, blocksize=BLOCKSIZE):
        self._q    = queue.Queue()
        self._buf  = np.zeros(0, dtype=np.float32)
        self._done = threading.Event()
        self._stream = sd.OutputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            blocksize=blocksize,
            callback=self._callback,
            finished_callback=self._done.set,
        )

    def _callback(self, outdata, frames, time_info, status):
        needed  = frames
        written = 0

        while written < needed:
            if len(self._buf) == 0:
                try:
                    item = self._q.get_nowait()
                    if item is None:           # sentinel → stop
                        outdata[written:] = 0
                        raise sd.CallbackStop()
                    self._buf = np.asarray(item, dtype=np.float32).flatten()
                except queue.Empty:
                    outdata[written:] = 0      # underrun → silence
                    return

            n = min(len(self._buf), needed - written)
            outdata[written:written + n, 0] = self._buf[:n]
            self._buf = self._buf[n:]
            written  += n

    def start(self):
        self._stream.start()
        return self

    def put(self, audio: np.ndarray):
        self._q.put(audio)

    def finish_and_wait(self):
        self._q.put(None)          # sentinel
        self._done.wait(timeout=30)
        self._stream.stop()
        self._stream.close()


# ──────────────────────────────────────────────────────────
# Load models
# ──────────────────────────────────────────────────────────
print(f"LLM  : {LLM_MODEL_ID}")
print(f"Voice: {VOICE_REF}\n")

print("Loading LLM...")
llm = LLMEngine()
llm.load_model()

print("\nLoading TTS (ONNX)...")
tts = PocketTTSOnnx(precision="int8", device="auto")
print(f"TTS ready on {tts.device}")

print("Encoding voice reference...")
voice_emb = tts.encode_voice(VOICE_REF)
print(f"Voice encoded: shape {voice_emb.shape}\n")

# ──────────────────────────────────────────────────────────
# Chat loop
# ──────────────────────────────────────────────────────────
print("=" * 58)
print("LLM → ONNX TTS streaming!  Type 'quit' to exit.")
print("=" * 58 + "\n")

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

    player = GaplessPlayer().start()
    print("[BOT] ", end="", flush=True)

    def _llm_stream():
        for chunk in llm.generate_stream(history):
            print(chunk, end="", flush=True)
            response_parts.append(chunk)
            yield chunk

    # stream2sentence wraps the LLM generator and yields smart sentence fragments:
    #   - after 10 chars minimum (minimum_first_fragment_length)
    #   - forced after 7 words even without punctuation
    #   - handles markdown, emojis, edge-cases automatically
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
        for audio_chunk in tts.stream(
            sentence.strip(),
            voice=voice_emb,
            first_chunk_frames=2,
        ):
            player.put(audio_chunk)
        n_sentences += 1

    player.finish_and_wait()

    full_response = "".join(response_parts)

    first_ms = (t_first_audio - t_start) * 1000 if t_first_audio else 0
    print(f"\n  ⏱  Total: {time.time() - t_start:.2f}s | "
          f"First audio: {first_ms:.0f}ms | "
          f"Sentences: {n_sentences}\n")

    history.append({"role": "assistant", "content": full_response.strip()})

print("\n👋 Bye!")
