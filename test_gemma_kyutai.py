"""
Gemma (or any LLM) → Kyutai TTS 1.6B streaming (single-GPU safe)

Fixes:
- Avoids starting moshi streaming multiple times ("is already streaming!")
- Enters LMGen.streaming_forever(1) ONCE for the lifetime of the program
- Enters tts_model.mimi.streaming(1) ONCE for the lifetime of the program
- Resets only per-utterance state per user input
- Fixes t_first_audio scoping (no incorrect 'global')
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
)
from moshi.conditioners import dropout_all_conditions
from moshi.models.lm import LMGen

from llm.llm_engine import LLMEngine
from core.config import LLM_MODEL_ID


TTS_REPO    = "kyutai/tts-1.6b-en_fr"
VOICE       = "expresso/ex03-ex01_happy_001_channel1_334s.wav"
DEVICE      = "cuda"         # TTS on GPU
SAMPLE_RATE = 24000
BLOCKSIZE   = 1920           # 80ms per frame


class TTSGen:
    """Long-lived streaming TTS generator (streaming_forever entered once)."""

    def __init__(self, tts_model, condition_attributes, on_frame=None):
        self.tts_model = tts_model
        self.on_frame  = on_frame

        # Per-utterance state (will be reset each turn)
        self.offset = 0
        self.state  = tts_model.machine.new_state([])

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
            text_tokens[:] = torch.tensor(out_tokens, dtype=torch.long, device=text_tokens.device)

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

        # Enter long-lived streaming mode ONCE.
        # This is what was crashing when you recreated TTSGen each turn.
        self._streaming_ctx = self.lm_gen.streaming_forever(1)

    def reset_utterance(self):
        """Reset per-utterance state but keep streaming session alive."""
        self.offset = 0
        self.state  = self.tts_model.machine.new_state([])

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
        inp = torch.full(
            (1, missing, 1),
            self.tts_model.machine.token_ids.zero,
            dtype=torch.long,
            device=self.tts_model.lm.device,
        )
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


def main():
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

    print("=" * 60)
    print("Gemma → Kyutai TTS streaming! Type 'quit' to exit.")
    print("(You'll hear audio ~1.3s after the first word is generated)")
    print("=" * 60 + "\n")

    history = []

    # --- Per-program audio queue and per-turn reset ---
    pcm_queue: queue.Queue[np.ndarray] = queue.Queue()

    # We'll update these per-turn
    timing = {"t_start": None, "t_first_audio": None}
    current_tts_gen = {"gen": None}  # holds the long-lived gen instance

    def on_frame(frame):
        # frame: audio tokens (or -1 sentinel frames)
        if (frame != -1).all():
            pcm = tts_model.mimi.decode(frame[:, 1:, :]).detach().cpu().numpy()
            chunk = np.clip(pcm[0, 0], -1, 1).astype(np.float32)
            if timing["t_first_audio"] is None:
                timing["t_first_audio"] = time.time()
            pcm_queue.put_nowait(chunk)

    def audio_callback(outdata, frames, time_info, status):
        try:
            pcm_data = pcm_queue.get(block=False)
            # Ensure correct shape
            if pcm_data.shape[0] != frames:
                # If mismatch (rare), pad or trim
                if pcm_data.shape[0] < frames:
                    pad = np.zeros(frames - pcm_data.shape[0], dtype=np.float32)
                    pcm_data = np.concatenate([pcm_data, pad])
                else:
                    pcm_data = pcm_data[:frames]
            outdata[:, 0] = pcm_data
        except queue.Empty:
            outdata[:] = 0

    # Create ONE long-lived TTS generator
    gen = TTSGen(tts_model, [cond_attrs], on_frame=on_frame)
    current_tts_gen["gen"] = gen

    # Enter long-lived streaming for Mimi ONCE as well.
    # This prevents repeatedly opening/closing streaming state.
    with (
        sd.OutputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCKSIZE,
            channels=1,
            callback=audio_callback,
        ),
        tts_model.mimi.streaming(1),
    ):
        while True:
            user_input = input("[YOU] ")
            if user_input.strip().lower() in ("quit", "exit", "q"):
                break

            history.append({"role": "user", "content": user_input})

            # Reset per-turn state
            gen.reset_utterance()
            while not pcm_queue.empty():
                try:
                    pcm_queue.get_nowait()
                except queue.Empty:
                    break

            timing["t_start"] = time.time()
            timing["t_first_audio"] = None
            full_response = ""

            print("[BOT] ", end="", flush=True)

            word_buffer = ""

            # Stream LLM output -> feed words into TTS
            for llm_chunk in llm.generate_stream(history):
                print(llm_chunk, end="", flush=True)
                full_response += llm_chunk
                word_buffer += llm_chunk

                # Feed each word (split on space or newline) as a separate TTS entry
                while " " in word_buffer or "\n" in word_buffer:
                    sep_pos_space = word_buffer.find(" ")
                    sep_pos_nl = word_buffer.find("\n")

                    # choose earliest separator that exists
                    seps = [p for p in (sep_pos_space, sep_pos_nl) if p != -1]
                    if not seps:
                        break
                    sep_pos = min(seps)

                    word = word_buffer[:sep_pos]
                    word_buffer = word_buffer[sep_pos + 1 :]

                    if word.strip():
                        for entry in make_entries(tts_model, word.strip()):
                            gen.append_entry(entry)
                            gen.process()

            # Remaining partial word
            if word_buffer.strip():
                for entry in make_entries(tts_model, word_buffer.strip()):
                    gen.append_entry(entry)

            # Flush the TTS pipeline
            gen.process_last()

            # Wait for audio queue to drain a bit
            timeout = time.time() + 30
            while pcm_queue.qsize() > 0 and time.time() < timeout:
                time.sleep(0.05)

            t_end = time.time()
            first_ms = (timing["t_first_audio"] - timing["t_start"]) * 1000 if timing["t_first_audio"] else 0
            print(f"\n  ⏱️  First audio: {first_ms:.0f}ms | Total: {t_end - timing['t_start']:.2f}s\n")

            history.append({"role": "assistant", "content": full_response.strip()})


if __name__ == "__main__":
    main()