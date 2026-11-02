"""Compare how efficiently tokenizers encode Sri Lankan Tamil (validation books).

Usage:  python tokenizer_check.py > ../experiments/2026-10-tokenizer-efficiency/tokenizer_results.txt
Needs HF_TOKEN in .env (Llama 3 is gated).
"""
import os

from datasets import load_dataset
from dotenv import load_dotenv
from transformers import AutoTokenizer

load_dotenv()
token = os.environ.get("HF_TOKEN")

ds = load_dataset("ctnlpr/sri-lankan-tamil-corpus", "documents", split="validation", token=token)
text = "\n".join(ds["text"])
n_chars, n_words = len(text), len(text.split())
print(f"Validation text: {n_chars:,} characters, {n_words:,} words\n")

TOKENIZERS = {
    "Llama 3 8B": "meta-llama/Meta-Llama-3-8B",
    "Qwen 2.5 7B": "Qwen/Qwen2.5-7B",
    "Qwen 3.5 4B Base": "Qwen/Qwen3.5-4B-Base",
    "Qwen 3.5 9B Base": "Qwen/Qwen3.5-9B-Base",
    "Agaram 2.5B": "murai-labs/agaram-2.5b",
}

print(f"{'Tokenizer':22} {'vocab':>8} {'tokens':>11} {'chars/token':>12} {'tokens/word':>12}")
for name, repo in TOKENIZERS.items():
    try:
        tok = AutoTokenizer.from_pretrained(repo, token=token)
        n_tok = len(tok(text, add_special_tokens=False)["input_ids"])
        print(f"{name:22} {len(tok):>8,} {n_tok:>11,} {n_chars / n_tok:>12.2f} {n_tok / n_words:>12.2f}")
    except Exception as e:  # keep going if one tokenizer fails
        print(f"{name:22} FAILED: {type(e).__name__}: {str(e)[:120]}")

sample = "யாழ்ப்பாணம் இலங்கையின் வடக்கே அமைந்துள்ள ஒரு நகரமாகும்."
print("\nExample:", sample)
for name, repo in TOKENIZERS.items():
    try:
        tok = AutoTokenizer.from_pretrained(repo, token=token)
        ids = tok(sample, add_special_tokens=False)["input_ids"]
        print(f"\n{name}: {len(ids)} tokens\n{[tok.decode([i]) for i in ids]}")
    except Exception:
        pass
