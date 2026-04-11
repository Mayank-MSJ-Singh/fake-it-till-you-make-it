"""
Threaded LLM → Kyutai TTS streaming

Fixes vs your previous threaded version:
1) IMPORTANT: Keep LMGen.streaming_forever(1) object alive
   - If you don't store it, it can get GC'd / not keep streaming state stable
   - This can manifest as "voice breaks" / conditioning sounds wrong

2) Add backpressure (bounded word_queue) so LLM doesn't outrun TTS
3) Reuse CUDA input tensor to reduce allocator churn
4) (Optional) disable CUDA graphs with DISABLE_CUDA_GRAPHS if you hit capture errors

Copy-paste ready.
"""
import threading
import queue
import time
import numpy as np
import sounddevice as sd
import torch

from moshi.models.loaders import CheckpointInfo
from moshi.models.tts import TTSModel, script_to_entries
from moshi.conditioners import dropout_all_conditions
from moshi.models.lm import LMGen
from moshi.utils.compile import no_cuda_graph  # Moshi 0.2.13

from llm.llm_engine import LLMEngine
from core.config import LLM_MODEL_ID

TTS_REPO    = "kyutai/tts-1.6b-en_fr"
VOICE       = "expresso/ex03-ex01_happy_001_channel1_334s.wav"
DEVICE      = "cuda"
SAMPLE_RATE = 24000
BLOCKSIZE   = 1920  # try 4800 if you still get underruns

# Safety vs speed:
DISABLE_CUDA_GRAPHS = False  # True = safer (no capture issues), slower

# Backpressure:
WORD_QUEUE_MAXSIZE = 30      # smaller => LLM waits more, steadier TTS
PCM_QUEUE_MAXSIZE  = 80      # ~80 * 80ms = ~6.4s buffer max


def make_entries(tts_model, text: str):
    return script_to_entries(
        tts_model.tokenizer,
        tts_model.machine.token_ids,
        tts_model.mimi.frame_rate,
        [text],
        multi_speaker=False,
        padding_between=1,
    )


class TTSWorker:
    """
    Dedicated TTS thread:
    - consumes words from word_queue
    - runs moshi TTS step() on CUDA
    - decodes PCM and pushes to pcm_queue
    """

    def __init__(self, tts_model: TTSModel, cond_attrs, pcm_queue: "queue.Queue[np.ndarray]"):
        self.tts_model = tts_model
        self.cond_attrs = cond_attrs
        self.pcm_queue = pcm_queue

        # BOUNDED queue => backpressure => LLM can't outrun TTS
        self.word_queue: "queue.Queue[str | None]" = queue.Queue(maxsize=WORD_QUEUE_MAXSIZE)

        self.thread: threading.Thread | None = None
        self.running = False

        # per-utterance state
        self.offset = 0
        self.state = tts_model.machine.new_state([])

        # timing hooks
        self._turn_started_at = None
        self._first_audio_at = None
        self._timing_lock = threading.Lock()

        # Keep streaming context objects alive (IMPORTANT)
        self._lm_streaming_ctx = None

        # build LMGen once
        attrs = [cond_attrs]
        if tts_model.cfg_coef != 1.0:
            attrs = attrs + dropout_all_conditions([cond_attrs])

        prepared = tts_model.lm.condition_provider.prepare(attrs)
        self.condition_tensors = tts_model.lm.condition_provider(prepared)

        def _on_text_logits_hook(text_logits):
            if tts_model.padding_bonus:
                text_logits[..., tts_model.machine.token_ids.pad] += tts_model.padding_bonus
            return text_logits

        def _on_audio_hook(audio_tokens):
            for q in range(audio_tokens.shape[1]):
                delay = tts_model.lm.delays[q + tts_model.lm.audio_offset]
                if self.offset < delay + tts_model.delay_steps:
                    audio_tokens[:, q] = tts_model.machine.token_ids.zero

        def _on_text_hook(text_tokens):
            tokens = text_tokens.tolist()
            out_tokens = []
            for token in tokens:
                out_token, _ = tts_model.machine.process(self.offset, self.state, token)
                out_tokens.append(out_token)
            text_tokens[:] = torch.tensor(out_tokens, dtype=torch.long, device=text_tokens.device)

        tts_model.lm.dep_q = tts_model.n_q

        self.lm_gen = LMGen(
            tts_model.lm,
            temp=tts_model.temp, temp_text=tts_model.temp,
            cfg_coef=tts_model.cfg_coef,
            condition_tensors=self.condition_tensors,
            on_text_logits_hook=_on_text_logits_hook,
            on_text_hook=_on_text_hook,
            on_audio_hook=_on_audio_hook,
            cfg_is_masked_until=None,
            cfg_is_no_text=True,
        )

        # reuse input tensor (reduces allocator churn)
        missing = tts_model.lm.n_q - tts_model.lm.dep_q
        self._inp = torch.full(
            (1, missing, 1),
            tts_model.machine.token_ids.zero,
            dtype=torch.long,
            device=tts_model.lm.device,
        )

    def reset_utterance(self):
        self.offset = 0
        self.state = self.tts_model.machine.new_state([])

        with self._timing_lock:
            self._turn_started_at = time.time()
            self._first_audio_at = None

    def timing_snapshot(self):
        with self._timing_lock:
            return self._turn_started_at, self._first_audio_at

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        try:
            self.word_queue.put(None, block=False)
        except queue.Full:
            # if full, unblock by removing one then put None
            try:
                _ = self.word_queue.get_nowait()
            except queue.Empty:
                pass
            self.word_queue.put(None, block=False)

        if self.thread:
            self.thread.join(timeout=5)

    def push_word(self, word: str):
        # blocks when queue is full -> natural backpressure
        self.word_queue.put(word, block=True)

    def end_turn(self):
        self.word_queue.put("__EOS__", block=True)

    def _step(self):
        if DISABLE_CUDA_GRAPHS:
            with torch.inference_mode(), no_cuda_graph():
                frame = self.lm_gen.step(self._inp)
        else:
            with torch.inference_mode():
                frame = self.lm_gen.step(self._inp)

        self.offset += 1
        return frame

    def _decode_and_enqueue(self, frame):
        if (frame != -1).all():
            if DISABLE_CUDA_GRAPHS:
                with torch.inference_mode(), no_cuda_graph():
                    pcm = self.tts_model.mimi.decode(frame[:, 1:, :]).detach().cpu().numpy()
            else:
                with torch.inference_mode():
                    pcm = self.tts_model.mimi.decode(frame[:, 1:, :]).detach().cpu().numpy()

            chunk = np.clip(pcm[0, 0], -1, 1).astype(np.float32)

            with self._timing_lock:
                if self._first_audio_at is None:
                    self._first_audio_at = time.time()

            # Keep queue bounded; if full, drop oldest to keep real-time
            try:
                self.pcm_queue.put_nowait(chunk)
            except queue.Full:
                try:
                    _ = self.pcm_queue.get_nowait()
                except queue.Empty:
                    pass
                self.pcm_queue.put_nowait(chunk)

    def _flush_utterance(self):
        # flush pipeline for this utterance
        while len(self.state.entries) > 0 or self.state.end_step is not None:
            frame = self._step()
            if frame is not None:
                self._decode_and_enqueue(frame)

        extra = self.tts_model.delay_steps + max(self.tts_model.lm.delays) + 8
        for _ in range(extra):
            frame = self._step()
            if frame is not None:
                self._decode_and_enqueue(frame)

    def _run(self):
        # All streaming contexts live INSIDE this thread
        with self.tts_model.mimi.streaming(1):
            # IMPORTANT: store this object so it stays alive for the whole thread lifetime
            self._lm_streaming_ctx = self.lm_gen.streaming_forever(1)

            while self.running:
                item = self.word_queue.get()
                if item is None:
                    break

                if item == "__EOS__":
                    self._flush_utterance()
                    continue

                word = item.strip()
                if not word:
                    continue

                for entry in make_entries(self.tts_model, word):
                    self.state.entries.append(entry)

                # process while we have enough lookahead
                while len(self.state.entries) > self.tts_model.machine.second_stream_ahead:
                    frame = self._step()
                    if frame is not None:
                        self._decode_and_enqueue(frame)


def main():
    print(f"LLM : {LLM_MODEL_ID}")
    print(f"TTS : {TTS_REPO}\n")

    print("Loading LLM...")
    llm = LLMEngine()
    llm.load_model()

    print("\nLoading Kyutai TTS...")
    info = CheckpointInfo.from_hf_repo(TTS_REPO)
    tts_model = TTSModel.from_checkpoint_info(info, n_q=32, temp=0.6, device=DEVICE)

    print(f"Loading voice '{VOICE}'...")
    voice_path = tts_model.get_voice_path(VOICE)
    cond_attrs = tts_model.make_condition_attributes([voice_path], cfg_coef=2.0)
    print("Voice ready.\n")

    print("=" * 60)
    print("LLM → Kyutai TTS streaming (TTS in dedicated thread). Type 'quit' to exit.")
    print("=" * 60 + "\n")

    history = []

    # bounded PCM queue
    pcm_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=PCM_QUEUE_MAXSIZE)

    tts = TTSWorker(tts_model, cond_attrs, pcm_queue)
    tts.start()

    def audio_callback(outdata, frames, time_info, status):
        try:
            pcm_data = pcm_queue.get(block=False)
            if pcm_data.shape[0] != frames:
                if pcm_data.shape[0] < frames:
                    pad = np.zeros(frames - pcm_data.shape[0], dtype=np.float32)
                    pcm_data = np.concatenate([pcm_data, pad])
                else:
                    pcm_data = pcm_data[:frames]
            outdata[:, 0] = pcm_data
        except queue.Empty:
            outdata[:] = 0

    with sd.OutputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCKSIZE,
        channels=1,
        callback=audio_callback,
    ):
        while True:
            user_input = input("[YOU] ")
            if user_input.strip().lower() in ("quit", "exit", "q"):
                break

            history.append({"role": "user", "content": user_input})

            # reset per turn
            tts.reset_utterance()
            while not pcm_queue.empty():
                try:
                    pcm_queue.get_nowait()
                except queue.Empty:
                    break

            print("[BOT] ", end="", flush=True)
            full_response = ""
            word_buffer = ""

            # IMPORTANT: if you updated llm_engine.py with throttle support,
            # you can also throttle here. Otherwise, this still works due to
            # bounded word_queue in TTSWorker.
            for llm_chunk in llm.generate_stream(history):
                print(llm_chunk, end="", flush=True)
                full_response += llm_chunk
                word_buffer += llm_chunk

                while " " in word_buffer or "\n" in word_buffer:
                    sep_pos_space = word_buffer.find(" ")
                    sep_pos_nl = word_buffer.find("\n")
                    seps = [p for p in (sep_pos_space, sep_pos_nl) if p != -1]
                    if not seps:
                        break
                    sep_pos = min(seps)

                    word = word_buffer[:sep_pos]
                    word_buffer = word_buffer[sep_pos + 1 :]

                    if word.strip():
                        tts.push_word(word.strip())

            if word_buffer.strip():
                tts.push_word(word_buffer.strip())

            tts.end_turn()

            # optional timing print
            t0 = time.time()
            while True:
                started, first = tts.timing_snapshot()
                if first is not None or time.time() - t0 > 10:
                    break
                time.sleep(0.01)

            started, first = tts.timing_snapshot()
            first_ms = (first - started) * 1000 if (started and first) else 0
            print(f"\n  ⏱️  First audio: {first_ms:.0f}ms\n")

            history.append({"role": "assistant", "content": full_response.strip()})

    tts.stop()


if __name__ == "__main__":
    main()