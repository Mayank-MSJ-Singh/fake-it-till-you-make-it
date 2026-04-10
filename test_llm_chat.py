"""Test: Chat with the LLM in streaming mode to verify it works."""
from llm.llm_engine import LLMEngine
from core.config import LLM_MODEL_ID, LLM_SYSTEM_PROMPT

print(f"Model: {LLM_MODEL_ID}")
print(f"System: {LLM_SYSTEM_PROMPT[:80]}...")
print()

engine = LLMEngine()
engine.load_model()

print("\n" + "=" * 50)
print("Chat with the model (streaming)! Type 'quit' to exit.")
print("=" * 50 + "\n")

history = []

while True:
    user_input = input("[YOU] ")
    if user_input.strip().lower() in ("quit", "exit", "q"):
        break

    history.append({"role": "user", "content": user_input})

    # Stream the response token by token
    print("[BOT] ", end="", flush=True)
    full_response = ""
    for chunk in engine.generate_stream(history):
        print(chunk, end="", flush=True)
        full_response += chunk
    print("\n")

    history.append({"role": "assistant", "content": full_response.strip()})

print("\n👋 Bye!")
