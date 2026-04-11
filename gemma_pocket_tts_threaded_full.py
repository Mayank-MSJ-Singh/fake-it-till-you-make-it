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


class AudioPlayer:
    def __init__(self, sample_rate=SAMPLE_RATE):
        self._q = queue.Queue()
        self._buf = np.array([], dtype=np.float32)
        self.first_audio_time = None

        self._stream = sd.OutputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            blocksize=SAMPLES_PER_FRAME,
            callback=self._callback,
        )

    def _callback(self, outdata, frames, time_info, status):
        needed = frames
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
            written += to_copy

    def start(self):
        self._stream.start()

    def put(self, audio: np.ndarray):
        if self.first_audio_time is None and audio is not None and len(audio) > 0:
            self.first_audio_time = time.time()
        self._q.put(audio.astype(np.float32))

    def finish_and_wait(self):
        self._q.put(None)
        while self._stream.active:
            time.sleep(0.05)
        self._stream.stop()
        self._stream.close()


def speak_inline(tts: TTSEngine, text: str, player: AudioPlayer):
    for chunk in tts.synthesize_stream(text):
        player.put(chunk)


def tts_worker(tts: TTSEngine, player: AudioPlayer, text_q: queue.Queue):
    while True:
        item = text_q.get()
        try:
            if item is _STOP:
                return
            for chunk in tts.synthesize_stream(item):
                player.put(chunk)
        finally:
            text_q.task_done()


def main():
    print(f"LLM   : {LLM_MODEL_ID}")
    print(f"Voice : {TTS_VOICE}\n")

    print("Loading LLM...")
    llm = LLMEngine()
    llm.load_model()

    print("\nLoading Pocket TTS...")
    tts = TTSEngine()
    tts.load_model()
    print(f"Sample rate: {tts.model.sample_rate} Hz\n")

    history = []

    while True:
        user_input = input("[YOU] ")
        if user_input.strip().lower() in ("quit", "exit", "q"):
            break

        history.append({"role": "user", "content": user_input})

        t_start = time.time()
        response_parts = []
        n_fragments = 0

        player = AudioPlayer()
        player.start()

        # Queue + worker for fragments AFTER the first one
        text_q = queue.Queue(maxsize=50)
        worker = None
        worker_started = False

        print("[BOT] ", end="", flush=True)

        def _llm_stream():
            for token_chunk in llm.generate_stream(history):
                print(token_chunk, end="", flush=True)
                response_parts.append(token_chunk)
                yield token_chunk

        first_fragment_done_inline = False

        for fragment in generate_sentences(
            _llm_stream(),
            minimum_sentence_length=12,
            minimum_first_fragment_length=10,
            force_first_fragment_after_words=7,
            quick_yield_single_sentence_fragment=True,
            cleanup_text_links=True,
            cleanup_text_emojis=True,
        ):
            frag = fragment.strip()
            if not frag:
                continue

            # Fragment #1: inline (fastest first audio)
            if not first_fragment_done_inline:
                first_fragment_done_inline = True
                speak_inline(tts, frag, player)
                n_fragments += 1

                # Start worker AFTER first fragment has begun speaking
                worker = threading.Thread(target=tts_worker, args=(tts, player, text_q), daemon=True)
                worker.start()
                worker_started = True
                continue

            # Fragments #2+: enqueue for background TTS
            text_q.put(frag)
            n_fragments += 1

        # Finish worker if it started
        if worker_started:
            text_q.put(_STOP)
            text_q.join()
            worker.join()

        player.finish_and_wait()

        full_response = "".join(response_parts).strip()
        history.append({"role": "assistant", "content": full_response})

        first_ms = (player.first_audio_time - t_start) * 1000 if player.first_audio_time else 0
        print(
            f"\n  ⏱  First audio: {first_ms:.0f}ms | "
            f"Total: {time.time() - t_start:.2f}s | "
            f"Fragments: {n_fragments}\n"
        )

    print("\nBye!")


if __name__ == "__main__":
    main()