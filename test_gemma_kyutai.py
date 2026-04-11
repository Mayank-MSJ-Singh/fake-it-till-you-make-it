"""
Test: Gemma 3 → Kyutai TTS 1.6B - True End-to-End Streaming

Pipeline:
  Gemma streams tokens → buffer into words → script_to_entries(word)
  → gen.append_entry() → gen.process() → on_frame() callback
  → audio queue → sd.OutputStream plays continuously

The TTS model starts generating audio after ~1.28s (16 audio frames delay),
then stays fully pipelined with the LLM output.
"""
import queue
import threading
import time
import numpy as np
import sounddevice as sd
import torch

from moshi.models.loaders import CheckpointInfo
from moshi.models.tts import (
    TTSModel,
    script_to_entries,
    DEFAULT_DSM_TTS_REPO,
    DEFAULT_DSM_TTS_VOICE_REPO,
)
from moshi.conditioners import dropout_all_conditions
from moshi.models.lm import LMGen

from llm.llm_engine import LLMEngine
from core.config import LLM_MODEL_ID

TTS_REPO    = "kyutai/tts-1.6b-en_fr"
VOICE       = "expresso/ex03-ex01_happy_001_channel1_334s.wav"
DEVICE      = "cuda"
SAMPLE_RATE = 24000
BLOCKSIZE   = 1920  # 80ms per frame


# ─────────────────────────────────────────────
# TTSGen — same as test_kyutai_tts.py
# ─────────────────────────────────────────────
class TTSGen:
    """Streaming TTS generator."""

    def __init__(self, tts_model, condition_attributes, on_frame=None):
        self.tts_model = tts_model
        self.on_frame  = on_frame
        self.offset    = 0
        self.state     = tts_model.machine.new_state([])

        attributes = list(condition_attributes)
        if tts_model.cfg_coef != 1.0:
            nulled     = dropout_all_conditions(list(condition_attributes))
            attributes = attributes + nulled

        prepared          = tts_model.lm.condition_provider.prepare(attributes)
        condition_tensors = tts_model.lm.condition_provider(prepared)

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
            tokens     = text_tokens.tolist()
            out_tokens = []
            for token in tokens:
                out_token, _ = tts_model.machine.process(self.offset, self.state, token)
                out_tokens.append(out_token)
            text_tokens[:] = torch.tensor(out_tokens, dtype=torch.long,
                                          device=text_tokens.device)

        # Reset any leftover streaming state from a previous turn
        # Must happen BEFORE LMGen.streaming_forever() is called
        try:
            tts_model.lm.reset_streaming()
        except Exception:
            pass

        tts_model.lm.dep_q = tts_model.n_q
        self.lm_gen = LMGen(
            tts_model.lm,
            temp=tts_model.temp, temp_text=tts_model.temp,
            cfg_coef=tts_model.cfg_coef,
            condition_tensors=condition_tensors,
            on_text_logits_hook=_on_text_logits_hook,
            on_text_hook=_on_text_hook,
            on_audio_hook=_on_audio_hook,
            cfg_is_masked_until=None,
            cfg_is_no_text=True,
        )
        self.lm_gen.streaming_forever(1)

    def append_entry(self, entry):
        self.state.entries.append(entry)

    def process(self):
        while len(self.state.entries) > self.tts_model.machine.second_stream_ahead:
            self._step()

    def process_last(self):
        while len(self.state.entries) > 0 or self.state.end_step is not None:
            self._step()
        extra = self.tts_model.delay_steps + max(self.tts_model.lm.delays) + 8
        for _ in range(extra):
            self._step()

    def _step(self):
        missing = self.tts_model.lm.n_q - self.tts_model.lm.dep_q
        inp = torch.full((1, missing, 1), self.tts_model.machine.token_ids.zero,
                         dtype=torch.long, device=self.tts_model.lm.device)
        frame = self.lm_gen.step(inp)
        self.offset += 1
        if frame is not None and self.on_frame is not None:
            self.on_frame(frame)


def make_entries(tts_model, text: str):
    """Convert a text fragment to TTS entries."""
    return script_to_entries(
        tts_model.tokenizer,
        tts_model.machine.token_ids,
        tts_model.mimi.frame_rate,
        [text],
        multi_speaker=False,
        padding_between=1,
    )


# ─────────────────────────────────────────────
# Load models
# ─────────────────────────────────────────────
print(f"LLM : {LLM_MODEL_ID}")
print(f"TTS : {TTS_REPO}\n")

print("Loading LLM...")
llm = LLMEngine()
llm.load_model()

print("\nLoading Kyutai TTS...")
t0 = time.time()
checkpoint_info = CheckpointInfo.from_hf_repo(TTS_REPO)
tts_model = TTSModel.from_checkpoint_info(checkpoint_info, n_q=32, temp=0.6, device=DEVICE)
print(f"TTS loaded in {time.time() - t0:.1f}s")

print(f"Loading voice '{VOICE}'...")
voice_path = tts_model.get_voice_path(VOICE)
cond_attrs  = tts_model.make_condition_attributes([voice_path], cfg_coef=2.0)
print("Voice ready.\n")

# ─────────────────────────────────────────────
# Chat loop
# ─────────────────────────────────────────────
print("=" * 60)
print("Gemma → Kyutai TTS streaming! Type 'quit' to exit.")
print("(You'll hear audio ~1.3s after the first word is generated)")
print("=" * 60 + "\n")

history = []

while True:
    user_input = input("[YOU] ")
    if user_input.strip().lower() in ("quit", "exit", "q"):
        break

    history.append({"role": "user", "content": user_input})

    t_start       = time.time()
    t_first_audio = None
    full_response = ""

    # Audio queue → OutputStream
    pcm_queue = queue.Queue()

    def on_frame(frame):
        global t_first_audio
        if (frame != -1).all():
            pcm = tts_model.mimi.decode(frame[:, 1:, :]).cpu().detach().numpy()
            chunk = np.clip(pcm[0, 0], -1, 1)
            if t_first_audio is None:
                t_first_audio = time.time()
            pcm_queue.put_nowait(chunk)

    def audio_callback(outdata, frames, time_info, status):
        try:
            pcm_data = pcm_queue.get(block=False)
            outdata[:, 0] = pcm_data
        except queue.Empty:
            outdata[:] = 0

    gen = TTSGen(tts_model, [cond_attrs], on_frame=on_frame)

    print("[BOT] ", end="", flush=True)

    with (
        sd.OutputStream(samplerate=SAMPLE_RATE, blocksize=BLOCKSIZE,
                        channels=1, callback=audio_callback),
        tts_model.mimi.streaming(1),
    ):
        word_buffer = ""

        for llm_chunk in llm.generate_stream(history):
            print(llm_chunk, end="", flush=True)
            full_response += llm_chunk
            word_buffer   += llm_chunk

            # Feed each word (split on space) as a separate TTS entry
            # This way TTS gets text as fast as LLM produces it
            while " " in word_buffer or "\n" in word_buffer:
                # Split off the first complete word
                for sep in (" ", "\n"):
                    if sep in word_buffer:
                        word, word_buffer = word_buffer.split(sep, 1)
                        if word.strip():
                            for entry in make_entries(tts_model, word.strip()):
                                gen.append_entry(entry)
                                gen.process()
                        break

        # Feed any remaining partial word
        if word_buffer.strip():
            for entry in make_entries(tts_model, word_buffer.strip()):
                gen.append_entry(entry)

        # Flush the TTS pipeline
        gen.process_last()

        # Wait for audio queue to drain
        timeout = time.time() + 30
        while pcm_queue.qsize() > 0 and time.time() < timeout:
            time.sleep(0.05)

    first_ms = (t_first_audio - t_start) * 1000 if t_first_audio else 0
    print(f"\n  ⏱️  First audio: {first_ms:.0f}ms | Total: {time.time() - t_start:.2f}s\n")

    history.append({"role": "assistant", "content": full_response.strip()})

print("\n👋 Bye!")
