from huggingface_hub import InferenceClient
import yaml

with open("config.yaml") as f:
    config = yaml.safe_load(f)

token = config["llm"]["hf_token"]

# Test a few known free-tier chat models
candidates = [
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "HuggingFaceH4/zephyr-7b-beta",
    "google/gemma-2-2b-it",
    "google/gemma-7b-it",
    "microsoft/Phi-3-mini-4k-instruct",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
]

print("Testing models...\n")
for model in candidates:
    try:
        client = InferenceClient(model=model, token=token)
        resp = client.chat_completion(
            messages=[{"role": "user", "content": "say ok"}],
            max_tokens=5
        )
        print(f"✅ {model}")
    except Exception as e:
        print(f"❌ {model} — {str(e)[:80]}")

