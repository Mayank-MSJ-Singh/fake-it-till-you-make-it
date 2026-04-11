"""
Test: Kyutai TTS 1.6B - True Streaming Text-to-Speech

Uses moshi.models.tts.TTSModel which supports real streaming:
  - Feed text word-by-word via append_entry()
  - Audio frames come out immediately via on_frame callback
  - No need to wait for full sentence!

Based on the official script:
  https://github.com/kyutai-labs/delayed-streams-modeling/blob/main/scripts/tts_pytorch_streaming.py
"""
import queue
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

HF_REPO = "kyutai/tts-1.6b-en_fr"
DEVICE = "cuda"

# ── Helper to build TTSGen (simplified from the official script) ──
class TTSGen:
    """Streaming TTS generator. Feed entries, get audio frames."""

    def __init__(self, tts_model: TTSModel, condition_attributes, on_frame=None):
        self.tts_model = tts_model
        self.on_frame = on_frame
        self.offset = 0
        self.state = tts_model.machine.new_state([])

        # CFG distillation setup
        attributes = list(condition_attributes)
        if tts_model.cfg_coef != 1.0:
            nulled = dropout_all_conditions(list(condition_attributes))
            attributes = attributes + nulled

        assert tts_model.lm.condition_provider is not None
        prepared = tts_model.lm.condition_provider.prepare(attributes)
        condition_tensors = tts_model.lm.condition_provider(prepared)

        def _on_text_logits_hook(text_logits):
            if tts_model.padding_bonus:
                text_logits[..., tts_model.machine.token_ids.pad] += tts_model.padding_bonus
            return text_logits

        def _on_audio_hook(audio_tokens):
            audio_offset = tts_model.lm.audio_offset
            delays = tts_model.lm.delays
            for q in range(audio_tokens.shape[1]):
                delay = delays[q + audio_offset]
                if self.offset < delay + tts_model.delay_steps:
                    audio_tokens[:, q] = tts_model.machine.token_ids.zero

        def _on_text_hook(text_tokens):
            tokens = text_tokens.tolist()
            out_tokens = []
            for token in tokens:
                out_token, _ = tts_model.machine.process(self.offset, self.state, token)
                out_tokens.append(out_token)
            text_tokens[:] = torch.tensor(out_tokens, dtype=torch.long, device=text_tokens.device)

        # Reset any leftover streaming state from a previous turn
        try:
            tts_model.lm.reset_streaming()
        except Exception:
            pass

        tts_model.lm.dep_q = tts_model.n_q
        self.lm_gen = LMGen(
            tts_model.lm,
            temp=tts_model.temp,
            temp_text=tts_model.temp,
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
        """Feed a text entry to the generator."""
        self.state.entries.append(entry)

    def process(self):
        """Run LM steps until enough text has been consumed."""
        while len(self.state.entries) > self.tts_model.machine.second_stream_ahead:
            self._step()

    def process_last(self):
        """Drain all remaining text and flush the pipeline."""
        while len(self.state.entries) > 0 or self.state.end_step is not None:
            self._step()
        additional_steps = (
            self.tts_model.delay_steps + max(self.tts_model.lm.delays) + 8
        )
        for _ in range(additional_steps):
            self._step()

    def _step(self):
        missing = self.tts_model.lm.n_q - self.tts_model.lm.dep_q
        input_tokens = torch.full(
            (1, missing, 1),
            self.tts_model.machine.token_ids.zero,
            dtype=torch.long,
            device=self.tts_model.lm.device,
        )
        frame = self.lm_gen.step(input_tokens)
        self.offset += 1
        if frame is not None and self.on_frame is not None:
            self.on_frame(frame)


# ── Load model ──
print(f"Loading TTS model from {HF_REPO}...")
t0 = time.time()
checkpoint_info = CheckpointInfo.from_hf_repo(HF_REPO)
tts_model = TTSModel.from_checkpoint_info(
    checkpoint_info, n_q=32, temp=0.6, device=DEVICE
)
print(f"Model loaded in {time.time() - t0:.1f}s")

# ── Load voice ──
print(f"Loading voice from {DEFAULT_DSM_TTS_VOICE_REPO}...")
voice = "expresso/ex03-ex01_happy_001_channel1_334s.wav"
voice_path = tts_model.get_voice_path(voice)
condition_attributes = tts_model.make_condition_attributes([voice_path], cfg_coef=2.0)
print(f"Voice ready.")

# ── Test texts ──
TEST_SENTENCES = [
    "Hello! I am Kyutai's streaming text to speech model.",
    "I can start speaking before I even know the full sentence.",
    "This is true real-time streaming, not sentence-by-sentence.",
]

print(f"\nSample rate: {tts_model.mimi.sample_rate} Hz")
print("=" * 60)
print("Testing streaming TTS. You should hear audio immediately!\n")

for text in TEST_SENTENCES:
    print(f"Speaking: \"{text}\"")

    pcms = queue.Queue()
    t_first_audio = None
    t_start = time.time()

    def on_frame(frame):
        global t_first_audio
        if (frame != -1).all():
            pcm = tts_model.mimi.decode(frame[:, 1:, :]).cpu().detach().numpy()
            chunk = np.clip(pcm[0, 0], -1, 1)
            if t_first_audio is None:
                t_first_audio = time.time()
            pcms.put_nowait(chunk)

    def audio_callback(outdata, frames, time_info, status):
        try:
            pcm_data = pcms.get(block=False)
            outdata[:, 0] = pcm_data
        except queue.Empty:
            outdata[:] = 0

    gen = TTSGen(tts_model, [condition_attributes], on_frame=on_frame)

    with (
        sd.OutputStream(
            samplerate=tts_model.mimi.sample_rate,
            blocksize=1920,
            channels=1,
            callback=audio_callback,
        ),
        tts_model.mimi.streaming(1),
    ):
        entries = script_to_entries(
            tts_model.tokenizer,
            tts_model.machine.token_ids,
            tts_model.mimi.frame_rate,
            [text],
            multi_speaker=False,
            padding_between=1,
        )
        for entry in entries:
            gen.append_entry(entry)
            gen.process()
        gen.process_last()

        # Wait for audio queue to drain
        while pcms.qsize() > 0:
            time.sleep(0.05)

    first_ms = (t_first_audio - t_start) * 1000 if t_first_audio else 0
    total_s = time.time() - t_start
    print(f"  ✅ First audio: {first_ms:.0f}ms | Total: {total_s:.2f}s\n")

print("Done! 🎉")
