"""Test: Get TEXT logits from the model and compute P(PAD) as pause signal.

The UNDERSTAND.md doc says prs comes from text token probabilities, not extra_heads.
Let's check if P(PAD token=3) gives a cleaner pause signal."""
import asyncio
import torch
import torch.nn as nn
import torch.nn.functional as F
import moshi.models
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from core.audio_io import MicrophoneInput

# Monkey-patch LMGen.step to also return logits
_orig_step = None

async def main():
    print("Loading model...")
    info = moshi.models.loaders.CheckpointInfo.from_hf_repo("kyutai/stt-1b-en_fr")
    mimi = info.get_mimi(device="cuda")
    lm = info.get_moshi(device="cuda", dtype=torch.bfloat16)
    lm_gen = moshi.models.LMGen(lm, temp=0, temp_text=0.0)
    tokenizer = info.get_text_tokenizer()

    # We need to see what methods LMGen has
    print("\nLMGen methods:")
    for attr in dir(lm_gen):
        if not attr.startswith("_"):
            print(f"  {attr}")
    
    print("\nLM model text_linear:", lm.text_linear)
    print("LM model text_card:", getattr(lm, 'text_card', 'N/A'))
    
    # Check if we can hook into the forward to get logits
    # The text_linear layer maps hidden states to text logits
    # We can register a forward hook to capture logits
    
    captured_logits = {}
    
    def hook_fn(module, input, output):
        captured_logits['logits'] = output.detach()
    
    hook = lm.text_linear.register_forward_hook(hook_fn)
    
    # Warmup
    print("\nWarming up...")
    with mimi.streaming(1), lm_gen.streaming(1):
        silence = torch.zeros((1, 1, 1920), device="cuda")
        for _ in range(15):
            tokens = mimi.encode(silence)
            lm_gen.step(tokens)
    
    # Test with mic
    mic = MicrophoneInput()
    mic.start()
    
    print("\n🎤 Speak! Comparing text logits vs extra_heads. Ctrl+C to stop.\n")
    print(f"{'frm':>4} | {'word':>10} | {'P(PAD)':>8} | {'P(t=0)':>8} | {'P(real)':>8} | {'token':>5}")
    print("-" * 70)
    
    frame_count = 0
    with mimi.streaming(1), lm_gen.streaming(1):
        try:
            while True:
                frame_np = await mic.get_frame()
                audio_tensor = torch.from_numpy(frame_np).to("cuda").unsqueeze(0).unsqueeze(0)
                
                audio_tokens = mimi.encode(audio_tensor)
                text_tokens = lm_gen.step(audio_tokens)
                
                frame_count += 1
                if frame_count <= 12:
                    continue
                
                text_token = text_tokens[0, 0, 0].cpu().item()
                word = None
                if text_token not in (0, 3):
                    word = tokenizer.id_to_piece(int(text_token)).replace("▁", " ")
                
                # Get the captured logits
                if 'logits' in captured_logits:
                    logits = captured_logits['logits'].float()  # shape: [1, 1, 8000]
                    probs = F.softmax(logits[0, 0, :], dim=0)  # softmax over 8000 classes
                    
                    p_pad = probs[3].item()       # P(token 3 = PAD)
                    p_word = probs[0].item()      # P(token 0 = word boundary)
                    p_real = 1.0 - p_pad - p_word # P(any real text token)
                    
                    word_str = word if word else ""
                    if frame_count % 3 == 0 or word:
                        print(f"{frame_count:4d} | {word_str:>10} | {p_pad:8.4f} | {p_word:8.4f} | {p_real:8.4f} | {int(text_token):5d}")
                    
        except KeyboardInterrupt:
            print("\nDone!")
        finally:
            hook.remove()
            mic.stop()


asyncio.run(main())
