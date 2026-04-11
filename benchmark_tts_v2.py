"""
Benchmark v2: Apples-to-Apples TTS comparison

1. LLM generates response ONCE → stored as token list
2. Same tokens replayed through BOTH TTS approaches
3. Measures time-to-first-audio at the same point (first chunk in queue)

His approach first, then ours — with identical input.
"""
import queue
import threading
import time
import numpy as np
import sounddevice as sd
from stream2sentence import generate_sentences

from llm.llm_engine import LLMEngine
from tts.tts_engine import TTSEngine
from core.config import LLM_MODEL_ID, TTS_VOICE, SAMPLES_PER_FRAME

SAMPLE_RATE   = 24000
TOKEN_DELAY_S = 0.015   # simulate ~67 tokens/sec (realistic LLM speed)
_STOP         = object()
TEST_PROMPT   = "Tell me a long story."


# ─────────────────────────────────────────────
# Shared: GaplessPlayer with accurate timing
# ─────────────────────────────────────────────
class GaplessPlayer:
    def __init__(self):
        self._q   = queue.Queue()
        self._buf = np.array([], dtype=np.float32)
        self.t_first_chunk = None
        self._done = threading.Event()
        self._stream = sd.OutputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32",
            blocksize=SAMPLES_PER_FRAME,
            callback=self._callback,
            finished_callback=self._done.set,
        )

    def _callback(self, outdata, frames, time_info, status):
        written = 0
        while written < frames:
            if len(self._buf) == 0:
                try:
                    item = self._q.get_nowait()
                    if item is None:
                        outdata[written:] = 0
                        raise sd.CallbackStop()
                    self._buf = np.asarray(item, dtype=np.float32).flatten()
                except queue.Empty:
                    outdata[written:] = 0
                    return
            n = min(len(self._buf), frames - written)
            outdata[written:written + n, 0] = self._buf[:n]
            self._buf = self._buf[n:]
            written += n

    def start(self):
        self._stream.start()
        return self

    def put(self, audio):
        if self.t_first_chunk is None:
            self.t_first_chunk = time.time()
        self._q.put(np.asarray(audio, dtype=np.float32))

    def finish_and_wait(self):
        self._q.put(None)
        self._done.wait(timeout=30)
        self._stream.stop()
        self._stream.close()


def make_token_stream(tokens):
    """Replay stored tokens with realistic delay to simulate LLM streaming."""
    for tok in tokens:
        time.sleep(TOKEN_DELAY_S)
        yield tok


# ─────────────────────────────────────────────
# Approach B: His threaded (inline frag1, worker frag2+)
# ─────────────────────────────────────────────
def _tts_worker(tts, player, text_q):
    while True:
        item = text_q.get()
        try:
            if item is _STOP:
                return
            for chunk in tts.synthesize_stream(item):
                player.put(chunk)
        finally:
            text_q.task_done()


def run_threaded(tts, tokens, label="B-threaded"):
    t_start    = time.time()
    n_frags    = 0
    first_done = False
    worker     = None
    text_q     = queue.Queue(maxsize=50)
    player     = GaplessPlayer().start()

    print(f"\n[{label}] ", end="", flush=True)

    def _stream():
        for tok in make_token_stream(tokens):
            print(tok, end="", flush=True)
            yield tok

    for sentence in generate_sentences(
        _stream(),
        minimum_sentence_length=12,
        minimum_first_fragment_length=10,
        force_first_fragment_after_words=7,
        quick_yield_single_sentence_fragment=True,
        cleanup_text_links=True,
        cleanup_text_emojis=True,
    ):
        frag = sentence.strip()
        if not frag:
            continue

        if not first_done:
            first_done = True
            for chunk in tts.synthesize_stream(frag):
                player.put(chunk)
            n_frags += 1
            worker = threading.Thread(target=_tts_worker, args=(tts, player, text_q), daemon=True)
            worker.start()
        else:
            text_q.put(frag)
            n_frags += 1

    if worker:
        text_q.put(_STOP)
        text_q.join()
        worker.join()

    player.finish_and_wait()
    first_ms = (player.t_first_chunk - t_start) * 1000 if player.t_first_chunk else 0
    return first_ms, time.time() - t_start, n_frags


# ─────────────────────────────────────────────
# Approach A: Our sequential
# ─────────────────────────────────────────────
def run_sequential(tts, tokens, label="A-ours"):
    t_start = time.time()
    n_frags = 0
    player  = GaplessPlayer().start()

    print(f"\n[{label}] ", end="", flush=True)

    def _stream():
        for tok in make_token_stream(tokens):
            print(tok, end="", flush=True)
            yield tok

    for sentence in generate_sentences(
        _stream(),
        minimum_sentence_length=12,
        minimum_first_fragment_length=10,
        force_first_fragment_after_words=7,
        quick_yield_single_sentence_fragment=True,
        cleanup_text_links=True,
        cleanup_text_emojis=True,
    ):
        frag = sentence.strip()
        if not frag:
            continue
        for chunk in tts.synthesize_stream(frag):
            player.put(chunk)
        n_frags += 1

    player.finish_and_wait()
    first_ms = (player.t_first_chunk - t_start) * 1000 if player.t_first_chunk else 0
    return first_ms, time.time() - t_start, n_frags


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
print(f"LLM   : {LLM_MODEL_ID}")
print(f"Voice : {TTS_VOICE}")
print(f"Prompt: \"{TEST_PROMPT}\"\n")

print("Loading LLM...")
llm = LLMEngine()
llm.load_model()

print("\nLoading Pocket TTS...")
tts = TTSEngine()
tts.load_model()
print(f"Sample rate: {tts.model.sample_rate} Hz\n")

# ── Step 1: Generate LLM response ONCE ──
print("Generating LLM response once (this is the shared input)...")
history = [{"role": "user", "content": TEST_PROMPT}]
tokens = []
print("  ", end="", flush=True)
for tok in llm.generate_stream(history):
    print(tok, end="", flush=True)
    tokens.append(tok)
full_text = "".join(tokens).strip()
print(f"\n  → {len(tokens)} tokens stored\n")

# ── Step 2: Run BOTH on the SAME tokens ──
print("=" * 58)
print("APPROACH B — His threaded  (runs first)")
print("=" * 58)
b_first, b_total, b_frags = run_threaded(tts, tokens)
print(f"\n  ⏱  First audio : {b_first:.0f} ms")
print(f"  ⏱  Total       : {b_total:.2f} s")
print(f"  📦 Fragments   : {b_frags}")

print()
print("=" * 58)
print("APPROACH A — Our sequential  (runs second)")
print("=" * 58)
a_first, a_total, a_frags = run_sequential(tts, tokens)
print(f"\n  ⏱  First audio : {a_first:.0f} ms")
print(f"  ⏱  Total       : {a_total:.2f} s")
print(f"  📦 Fragments   : {a_frags}")

print()
print("=" * 58)
print("RESULT  (same LLM text, same token replay speed)")
print("=" * 58)
w_first = "B (his) " if b_first < a_first else "A (ours)"
w_total = "B (his) " if b_total < a_total else "A (ours)"
print(f"  Faster first audio : {w_first}  ({b_first:.0f}ms vs {a_first:.0f}ms)")
print(f"  Faster total time  : {w_total}  ({b_total:.2f}s vs {a_total:.2f}s)")
print()
