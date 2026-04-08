"""Test: Log ALL 6 values from vad_heads[2] during real speech.
This bypasses STTEngine to see the raw head output."""
import asyncio
import torch
import torch.nn as nn
import torch.nn.functional as F
import moshi.models
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from core.audio_io import MicrophoneInput


async def main():
    # Load model with extra heads
    print("Loading model...")
    info = moshi.models.loaders.CheckpointInfo.from_hf_repo("kyutai/stt-1b-en_fr")
    mimi = info.get_mimi(device="cuda")
    lm = info.get_moshi(device="cuda", dtype=torch.bfloat16)
    
    candle_path = hf_hub_download("kyutai/stt-1b-en_fr-candle", "model.safetensors")
    with safe_open(candle_path, framework="pt") as f:
        for i in range(4):
            weight = f.get_tensor(f"extra_heads.{i}.weight")
            head = nn.Linear(2048, 6, bias=False, dtype=torch.bfloat16, device="cuda")
            head.weight.data = weight.to(device="cuda", dtype=torch.bfloat16)
            lm.extra_heads.append(head)
    
    lm_gen = moshi.models.LMGen(lm, temp=0, temp_text=0.0)
    tokenizer = info.get_text_tokenizer()
    
    # Warmup
    print("Warming up...")
    with mimi.streaming(1), lm_gen.streaming(1):
        silence = torch.zeros((1, 1, 1920), device="cuda")
        for _ in range(15):
            tokens = mimi.encode(silence)
            lm_gen.step_with_extra_heads(tokens)
    
    # Start mic
    mic = MicrophoneInput()
    mic.start()
    
    print("\n🎤 Speak! Logging head[2] raw values + softmax. Ctrl+C to stop.\n")
    print(f"{'frm':>4} | {'word':>10} | {'raw[0]':>7} {'raw[1]':>7} {'raw[2]':>7} {'raw[3]':>7} {'raw[4]':>7} {'raw[5]':>7} | {'sm[0]':>6} {'sm[1]':>6}")
    print("-" * 110)
    
    frame_count = 0
    with mimi.streaming(1), lm_gen.streaming(1):
        try:
            while True:
                frame_np = await mic.get_frame()
                audio_tensor = torch.from_numpy(frame_np).to("cuda").unsqueeze(0).unsqueeze(0)
                
                audio_tokens = mimi.encode(audio_tensor)
                text_tokens, vad_heads = lm_gen.step_with_extra_heads(audio_tokens)
                
                frame_count += 1
                if frame_count <= 12:
                    continue
                
                text_token = text_tokens[0, 0, 0].cpu().item()
                word = None
                if text_token not in (0, 3):
                    word = tokenizer.id_to_piece(text_token).replace("▁", " ")
                
                # Get ALL 6 values from head 2
                head2 = vad_heads[2][0, 0, :].float().cpu()  # shape [6]
                sm = F.softmax(head2, dim=0)  # softmax over 6 values
                
                word_str = word if word else ""
                if frame_count % 3 == 0 or word:
                    raw_str = " ".join(f"{v:7.3f}" for v in head2.tolist())
                    sm_str = f"{sm[0]:6.3f} {sm[1]:6.3f}"
                    print(f"{frame_count:4d} | {word_str:>10} | {raw_str} | {sm_str}")
                    
        except KeyboardInterrupt:
            print("\nDone!")
        finally:
            mic.stop()


asyncio.run(main())
