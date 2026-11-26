# CPT · Agaram 2.5B on Sri Lankan Tamil (1 epoch, LoRA r=64, 16-bit)

**Date:** 25 Nov 2026 · **Run:** Kaggle notebook `sltfm_cpt_kaggle`, Version 2 (Save & Run All), 2×T4
**Code:** `06_pretraining/sltfm_cpt_kaggle.ipynb` (`MODEL = "agaram-2.5b"`, `RUN_MODE = "full"`)
**Adapter:** `ctnlpr/Agaram-2.5B-SLTFM-CPT-LoRA` (private), final adapter at the root, checkpoints every 25 steps (25 … 675, 695)
**W&B:** project `sltfm-cpt`, run `agaram-2.5b-sltfm-cpt-r64-16bit`

## Setup
| Setting | Value | Llama 3 run (for comparison) |
|---|---|---|
| Base model | `murai-labs/agaram-2.5b` (Apache-2.0) | `meta-llama/Meta-Llama-3-8B` |
| Precision | **16-bit** LoRA (fp16 frozen base, fp32 adapters) | 4-bit QLoRA |
| LoRA | r 64, alpha 128, dropout 0.05; q, k, v, o, gate, up, down | same |
| Trainable parameters | 94,961,664 (3.62%) | 167.8M (2.05%) |
| Tokenizer | Agaram's own: 4.721 characters/token, 2.016 tokens/word | 0.74 characters/token |
| Training text | 13,430,884 characters → 2,844,905 tokens → 1,389 blocks of 2,048 | same text, ~18.1M tokens |
| Document separator | `<|endoftext|>` (id 0), no BOS | `<|begin_of_text|>` … `<|end_of_text|>` |
| Batch | 1 × 1 accumulation × 2 GPUs = 4,096 tokens ≈ **19,337 characters per step** | ~24,300 characters per step |
| Steps | **695 (1 epoch)** | 553 per epoch, 2 epochs |
| Learning rate | 2e-4, cosine, 5% warm-up, 8-bit AdamW, fp16 | same |
| Versions | transformers 5.23.0, peft 0.21.3 | transformers 4.5x, peft 0.1x |

## Cost
- Training time: 3,676 s (**61 min**), plus ~20 min of evaluation (28 evals × ~44 s).
- Throughput: ~1,330–1,490 tokens/s ≈ **6,300–7,000 characters/s** on 2×T4.
- No skipped steps (`grad_norm` never `nan`).

## Eval loss (fixed validation subset: 100 of 102 blocks ≈ all held-out text)
| Step | 0 (base) | 25 | 100 | 200 | 350 | 500 | 695 (end) |
|---|---|---|---|---|---|---|---|
| Eval loss (nats/token) | 3.707 | 3.624 | 3.586 | 3.574 | 3.567 | 3.561 | **3.558** |

- **−4.0%** loss on held-out text. With a fixed text and tokenizer, that is also −4.0% bits per character.
- Approximate BPC from the eval loss: 3.707 / ln 2 / 4.787 ≈ **1.12 → 1.07**. This is block-based without overlapping windows, so it is **not** the E2 / Experiment A number. Those must be measured with the E2 method.
- Most of the gain came in the first ~200 steps (≈ 0.3 epoch); after that the curve is almost flat.
- Eval loss is in nats per **Agaram** token: never compare it with Llama or Qwen eval loss directly.

## Generation (greedy, repetition penalty 1.15, 120 new tokens; base vs + adapter)
- **Jaffna prompt:** the adapter writes Sri Lankan encyclopedic Tamil (யாழ்ப்பாணக் குடாநாட்டினதும் தீவுகளினதும் தலைநகரமாகும்; ஆரியச் சக்கரவர்த்திகள், நல்லூர்). Some details are wrong or vague (e.g. "north-west coast", "12th century").
- **Eelam Tamil literature prompt:** the base loops ("ஒரு நாள் கழிந்தது" repeated). The adapter writes a Noolaham-style biography (Velanai, Jaffna Hindu College), but the person and facts are invented.
- **Co-operative movement prompt:** both are fluent. The adapter has factual errors (e.g. "1972 open economy policy"; it was 1977).
- **Reading:** CPT moved style and vocabulary towards Sri Lankan Tamil and removed the repetition loop, but did **not** make the model more factual. This is expected from 13M characters; SFT and RAG address facts.

## Caveats
- 16-bit training, while Llama 3 and Qwen 3.5 (4B, 9B) were trained in 4-bit (Qwen 4B ran out of memory in 16-bit). Precision decision pending with Anna; a 4-bit rerun costs ~1.5 h.
- ~20% less text per step than the Llama run (minimum of 1 block per GPU).
- 1 epoch, versus 2 for the released Llama 3 adapter.

## Next
- Experiment A on this adapter (training vs held-out vs Wikipedia; checkpoints 0, 125, 350, 695 ≈ 0, 0.18, 0.5, 1.0 epoch), and E2-style full held-out BPC.
- The 10 questions and 6 continuations from E2.
