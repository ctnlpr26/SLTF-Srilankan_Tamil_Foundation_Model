# E2 · Base-model comparison on Sri Lankan Tamil

**Date:** 16 Nov 2026 · **Run:** Kaggle notebook "SLTFM · E2", Version 2 (Save & Run All, 2 h 53 min)
**Code:** `07_evaluation/sltfm_e2_base_model_comparison.ipynb` (commit aab87bd)

## Question
Which base model models Sri Lankan Tamil best **before** any training of ours? This informs the choice of base model for the final CPT run.

## Setup
- **Data:** validation split of `ctnlpr/sri-lankan-tamil-corpus` (documents): 13 books, 1,001,456 characters, never used in training
- **Models:** all loaded in 4-bit NF4 with fp16 compute, one model per T4 GPU at a time
  - `meta-llama/Meta-Llama-3-8B` (current base)
  - `ctnlpr/Llama-3-SLTFM-8B-CPT-LoRA`, checkpoint-1100 (our CPT model; reference only)
  - `Qwen/Qwen3.5-4B-Base`, `Qwen/Qwen3.5-9B-Base`
  - `murai-labs/agaram-2.5b`
- **Metric:** bits per character (BPC), lower is better
  - tokenizer-independent, so models with different tokenizers can be compared fairly
  - overlapping windows: window 2048, stride 1024, every token scored once
  - 507 segments of 2,000 characters; bootstrap 95% CI (2,000 resamples); paired difference vs Llama 3
- **Generation:** 10 Sri Lankan Tamil questions (few-shot) and 6 continuations, greedy decoding
- **Hardware / versions:** Kaggle 2×T4, transformers 5.23.0, peft 0.21.3, torch 2.11.0+cu130, bitsandbytes 0.50.2

## Results

| Model | BPC (95% CI) | Δ vs Llama 3 | Questions correct /10* |
|---|---|---|---|
| Llama 3 8B | 1.561 (1.531–1.593) | — | 1 |
| Our CPT (Llama 3 8B + LoRA) | 1.244 | −0.318 (−20%) | 3 |
| Qwen 3.5 4B | 1.418 | −0.144 | ~2 |
| Qwen 3.5 9B | 1.309 | −0.252 | ~4 |
| **Agaram 2.5B** | **1.097** | **−0.464** | **9** |

\*Read by one person; not a formal rating. Full numbers and CIs are in `e2_results.md` and `e2_summary.json`.

## What it means
- Agaram 2.5B has the lowest BPC, and its continuations are clearly Sri Lankan Tamil, despite being the smallest model.
- Our CPT reduced Llama 3's BPC by about 20%, but the result is still behind Agaram and Qwen 3.5 9B is close.
- Qwen 3.5 writes encyclopedia-style Tamil and invents facts. Llama 3 base often switches to English. The CPT model repeats itself.

## Caveats / open checks
- **Contamination:** Agaram may have seen some of our validation books in pretraining. A per-book BPC check (Agaram/Qwen 9B ratio) is pending.
- Question scoring was done by one reader. It needs a second rater before it goes in the paper.
- All models were run in 4-bit, so absolute BPC may differ slightly from full precision. The ranking is what matters.
- The CPT model number replaces the old model-card figure (1.248 → 1.024 on 4% of the text).

## Files
- `e2_results.md`: results table plus all generations
- `e2_summary.json`: BPC, CIs, paired differences
- `e2_full/*.json`: per-model raw outputs (per-segment bits, generations)
