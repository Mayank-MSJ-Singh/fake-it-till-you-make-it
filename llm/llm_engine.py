"""
LLM Engine — Language Model for Response Generation
=====================================================

Loads a HuggingFace causal LM and generates conversational responses.
The model downloads automatically on first run and is cached locally.

All configuration lives in core/config.py:
  - LLM_MODEL_ID        → which model to download
  - LLM_DEVICE          → "cuda" or "cpu"
  - LLM_DTYPE           → precision (bfloat16, float16, float32)
  - LLM_MAX_NEW_TOKENS  → max response length
  - LLM_TEMPERATURE     → creativity (0=deterministic, 1=creative)
  - LLM_SYSTEM_PROMPT   → personality of the bot

To swap models, just change LLM_MODEL_ID in config.py. No code changes needed.

Two generation modes:
  generate()        — blocking, returns full response string
  generate_stream() — yields text chunks as they're generated (for TTS)
"""
import threading
from typing import Iterator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from core.config import (
    LLM_MODEL_ID,
    LLM_DEVICE,
    LLM_DTYPE,
    LLM_MAX_NEW_TOKENS,
    LLM_TEMPERATURE,
    LLM_TOP_P,
    LLM_REPETITION_PENALTY,
    LLM_SYSTEM_PROMPT,
)


# Map config string → torch dtype
_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


class LLMEngine:
    """Loads and runs a HuggingFace language model for chat responses.

    Lifecycle:
        1. load_model()       — downloads model from HF (cached after first run)
        2. generate()         — blocking, returns full response
        3. generate_stream()  — yields chunks as they're generated
    """

    def __init__(self):
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Download and load the LLM onto the configured device.

        Uses AutoModelForCausalLM which supports most HF causal models:
        Gemma, Qwen, Llama, Mistral, Phi, etc.

        The model is downloaded to HuggingFace's cache (~/.cache/huggingface/)
        on first run. Subsequent runs load from cache instantly.
        """
        dtype = _DTYPE_MAP.get(LLM_DTYPE, torch.bfloat16)

        print(f"Loading LLM: {LLM_MODEL_ID} on {LLM_DEVICE} ({LLM_DTYPE})...")

        # Download and load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)

        # Download and load model
        # device_map=LLM_DEVICE puts the whole model on one device.
        # For CPU: no VRAM used, slower but works alongside GPU STT.
        # For CUDA: fast but shares VRAM with STT (~2.3GB already used).
        self.model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_ID,
            dtype=dtype,
            device_map=LLM_DEVICE,
        )
        self.model.eval()

        # Report memory usage
        if LLM_DEVICE == "cuda":
            vram = torch.cuda.memory_allocated() / 1024**3
            print(f"LLM loaded! Total GPU memory: {vram:.1f} GB")
        else:
            print(f"LLM loaded on CPU (no GPU memory used)")

    def _prepare_inputs(self, messages: list[dict[str, str]]) -> torch.Tensor:
        """Format conversation history into model input tokens.

        Prepends the system prompt and applies the model's chat template
        (handles Gemma's <start_of_turn>, Qwen's <|im_start|>, etc.).
        """
        full_messages = [
            {"role": "system", "content": LLM_SYSTEM_PROMPT},
            *messages,
        ]
        input_ids = self.tokenizer.apply_chat_template(
            full_messages,
            return_tensors="pt",
            return_dict=False,
            add_generation_prompt=True,
        ).to(self.model.device)
        return input_ids

    def generate(self, messages: list[dict[str, str]]) -> str:
        """Generate a full response (blocking). Returns the complete text.

        Args:
            messages: List of {"role": "user"/"assistant", "content": "..."}

        Returns:
            The generated response text (string).
        """
        input_ids = self._prepare_inputs(messages)
        input_length = input_ids.shape[1]

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=LLM_MAX_NEW_TOKENS,
                temperature=LLM_TEMPERATURE,
                top_p=LLM_TOP_P,
                repetition_penalty=LLM_REPETITION_PENALTY,
                do_sample=LLM_TEMPERATURE > 0,
            )

        response_ids = output_ids[0, input_length:]
        return self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()

    def generate_stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        """Generate a response, yielding text chunks as they're produced.

        Uses TextIteratorStreamer — the model runs in a background thread
        and pushes decoded text chunks into an iterator that we yield from.

        This is essential for voice: we can start TTS on the first few
        words while the rest of the response is still being generated.

        Args:
            messages: List of {"role": "user"/"assistant", "content": "..."}

        Yields:
            Text chunks (strings) as the model generates them.
            Each chunk is typically a few tokens / one word.
        """
        input_ids = self._prepare_inputs(messages)

        # TextIteratorStreamer decodes tokens as they're generated
        # and lets us iterate over the text chunks.
        # skip_prompt=True means we only get the NEW text, not the input.
        # skip_special_tokens=True strips <eos>, <pad>, etc.
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        # model.generate() is blocking, so we run it in a thread.
        # The streamer receives tokens from the thread and yields them here.
        gen_kwargs = dict(
            inputs=input_ids,
            max_new_tokens=LLM_MAX_NEW_TOKENS,
            temperature=LLM_TEMPERATURE,
            top_p=LLM_TOP_P,
            repetition_penalty=LLM_REPETITION_PENALTY,
            do_sample=LLM_TEMPERATURE > 0,
            streamer=streamer,
        )
        thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()

        # Yield chunks as the model generates them.
        # The streamer blocks until the next chunk is ready.
        for chunk in streamer:
            if chunk:  # Skip empty chunks
                yield chunk

        thread.join()
