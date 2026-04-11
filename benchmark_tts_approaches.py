"""
Benchmark: Our approach vs Threaded approach for Gemma → Pocket TTS

Measures REAL time-to-first-audio (when first audio chunk hits the queue)
using the same measurement point for both approaches.

Run: python benchmark_tts_approaches.py
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

SAMPLE_RATE = 24000
_STOP = object()

# Fixed prompt so both runs are comparable
TEST_PROMPT = "Tell me a short 2-3 sentence story."

# ─────────────────────────────────────────────
# Shared: GaplessPlayer with accurate first-audio tracking
# ─────────────────────────────────────────────
class GaplessPlayer:
    def __init__(self, sample_rate=SAMPLE_RATE):
        self._q = queue.Queue()
        self._buf = np.array([], dtype=np.float32)
        self.t_first_chunk = None   # set when FIRST AUDIO CHUNK hits queue
        self._done = threading.Event()
        self._stream = sd.OutputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            blocksize=SAMPLES_PER_FRAME,
            callback=self._callback,
            finished_callback=self._done.set,
        )

    def _callback(self, outdata, frames, time_info, status):
        needed = frames
        written = 0
        while written < needed:
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
            n = min(len(self._buf), needed - written)
            outdata[written:written + n, 0] = self._buf[:n]
            self._buf = self._buf[n:]
            written += n

    def start(self):
        self._stream.start()
        return self

    def put(self, audio: np.ndarray):
        if self.t_first_chunk is None:
            self.t_first_chunk = time.time()  # ← accurate: when audio actually queued
        self._q.put(audio.astype(np.float32))

    def finish_and_wait(self):
        self._q.put(None)
        self._done.wait(timeout=30)
        self._stream.stop()
        self._stream.close()


# ─────────────────────────────────────────────
# Approach A: Our sequential approach
# ─────────────────────────────────────────────
def run_our_approach(llm, tts, history):
    t_start = time.time()
    response_parts = []
    n_fragments = 0

    player = GaplessPlayer().start()
    print("\n[OUR] ", end="", flush=True)

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
        for chunk in tts.synthesize_stream(sentence.strip()):
            player.put(chunk)
        n_fragments += 1

    player.finish_and_wait()

    first_ms = (player.t_first_chunk - t_start) * 1000 if player.t_first_chunk else 0
    total_s  = time.time() - t_start
    return first_ms, total_s, n_fragments, "".join(response_parts).strip()


# ─────────────────────────────────────────────
# Approach B: His threaded approach
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


def run_threaded_approach(llm, tts, history):
    t_start = time.time()
    response_parts = []
    n_fragments = 0

    player = GaplessPlayer().start()
    print("\n[HIS] ", end="", flush=True)

    def _llm_stream():
        for chunk in llm.generate_stream(history):
            print(chunk, end="", flush=True)
            response_parts.append(chunk)
            yield chunk

    text_q = queue.Queue(maxsize=50)
    first_done = False
    worker = None

    for sentence in generate_sentences(
        _llm_stream(),
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
            # Fragment 1: inline
            first_done = True
            for chunk in tts.synthesize_stream(frag):
                player.put(chunk)
            n_fragments += 1
            # Start worker AFTER first fragment queued (audio playing while worker runs)
            worker = threading.Thread(target=_tts_worker, args=(tts, player, text_q), daemon=True)
            worker.start()
        else:
            # Fragments 2+: background worker
            text_q.put(frag)
            n_fragments += 1

    if worker:
        text_q.put(_STOP)
        text_q.join()
        worker.join()

    player.finish_and_wait()

    first_ms = (player.t_first_chunk - t_start) * 1000 if player.t_first_chunk else 0
    total_s  = time.time() - t_start
    return first_ms, total_s, n_fragments, "".join(response_parts).strip()


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

history = [{"role": "user", "content": TEST_PROMPT}]

print("=" * 58)
print("APPROACH A — Our sequential  (stream2sentence, no thread)")
print("=" * 58)
a_first, a_total, a_frags, _ = run_our_approach(llm, tts, history)
print(f"\n  ⏱  First audio : {a_first:.0f} ms")
print(f"  ⏱  Total       : {a_total:.2f} s")
print(f"  📦 Fragments   : {a_frags}")

print()
print("=" * 58)
print("APPROACH B — His threaded  (inline frag1, worker frag2+)")
print("=" * 58)
b_first, b_total, b_frags, _ = run_threaded_approach(llm, tts, history)
print(f"\n  ⏱  First audio : {b_first:.0f} ms")
print(f"  ⏱  Total       : {b_total:.2f} s")
print(f"  📦 Fragments   : {b_frags}")

print()
print("=" * 58)
print("RESULT")
print("=" * 58)
winner_first = "A (ours)" if a_first < b_first else "B (his) "
winner_total = "A (ours)" if a_total < b_total else "B (his) "
print(f"  Faster first audio : {winner_first}  ({a_first:.0f}ms vs {b_first:.0f}ms)")
print(f"  Faster total time  : {winner_total}  ({a_total:.2f}s vs {b_total:.2f}s)")
print()
