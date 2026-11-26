# CPT · Qwen 3.5 4B Base on Sri Lankan Tamil (1 epoch, QLoRA r=64, 4-bit)

**Date:** 25 Nov 2026 · **Run:** Kaggle notebook `sltfm_cpt_kaggle_qwen3.5-4b`, Version 2 (Save & Run All), 2×T4
**Code:** `06_pretraining/sltfm_cpt_kaggle.ipynb` (`MODEL = "qwen3.5-4b"`, `QUANT = "4bit"`, `RUN_MODE = "full"`)
**Adapter:** `ctnlpr/Qwen3.5-4B-SLTFM-CPT-LoRA` (private), final adapter at the root, checkpoints every 25 steps (25 … 525, 534)
**W&B:** project `sltfm-cpt`, run `qwen3.5-4b-sltfm-cpt-r64-4bit`

## Setup
| Setting | Value |
|---|---|
| Base model | `Qwen/Qwen3.5-4B-Base` (Apache-2.0), loaded as `Qwen3_5ForCausalLM` (text only) |
| Precision | **4-bit QLoRA** (NF4, fp16 compute). 16-bit was planned but ran out of memory on T4: the 248k vocabulary needs a 1.89 GiB logits tensor at 2,048 tokens, on top of an 8.7 GB model |
| LoRA | r 64, alpha 128, dropout 0.05, `all-linear`: q, k, v, o, gate, up, down **plus the linear-attention projections** in_proj_qkv, in_proj_z, in_proj_a, in_proj_b, out_proj |
| Trainable parameters | 129,859,584 (3.00%) |
| Tokenizer | Qwen's own: 2.047 characters/token, 4.651 tokens/word |
| Training text | 13,430,884 characters → 6,561,077 tokens → 3,203 blocks of 2,048 |
| Document separator | `<|endoftext|>` (id 248044), no BOS |
| Batch | 1 × 3 accumulation × 2 GPUs = 12,288 tokens ≈ **25,154 characters per step** |
| Steps | **534 (1 epoch)** |
| Learning rate | 2e-4, cosine, 5% warm-up, 8-bit AdamW, fp16 |
| Versions | transformers 5.23.0, peft 0.21.3 |

## Cost
- Training time: 17,100 s (**4.75 h**), including evaluation every 25 steps (~108 s each).
- Throughput: ~460–490 tokens/s ≈ **~1,000 characters/s** on 2×T4, about 7× slower than Agaram per character.
- Cause: the linear-attention layers run in slow PyTorch fallback code on T4 (`causal_conv1d` and `flash-linear-attention` kernels not installed).
- The first 5 steps and 1 later step were skipped by the fp16 loss scaler (`grad_norm = nan`, update not applied): ~1% of steps.

## Eval loss (fixed validation subset: 100 blocks of 2,048 tokens, ~43% of the held-out text)
| Step | 0 (base) | 25 | 100 | 200 | 275 | 375 | 450 | 534 (end) |
|---|---|---|---|---|---|---|---|---|
| Eval loss (nats/token) | 2.069 | 2.003 | 1.939 | 1.878 | 1.846 | 1.817 | 1.800 | **1.796** |

- **−13.2%** loss on held-out text (same text and tokenizer, so also ≈ −13% bits per character).
- Approximate BPC from the eval loss: 2.069 / ln 2 / 2.056 ≈ **1.45 → 1.26**. This is block-based and covers only part of the held-out text, so it is **not** the E2 / Experiment A number.
- Unlike Agaram (flat after ~0.3 epoch), the curve kept falling until ~0.9 epoch: Qwen was still learning when the data ran out.
- Eval loss is in nats per **Qwen** token: never compare it with other models' eval loss directly.

## Generation (greedy, repetition penalty 1.15, 120 new tokens)
- **Jaffna prompt:** base writes Indian-centred text ("சமஸ்தானம்", Indian migration). The adapter switches to Saivite / temple themes typical of the corpus, but the text drifts and is not factual.
- **Eelam Tamil literature prompt:** both invent; the adapter mixes Agathiyar with a made-up "Nallur-kurinji king".
- Style moved towards the corpus (Saivite, literary), but fluency and facts are clearly weaker than Agaram + CPT.

## Caveats
- 4-bit (as Llama 3 and Qwen 9B); Agaram was trained in 16-bit.
- Eval subset covers ~43% of the held-out text (100 of 237 blocks); E2 / Experiment A use all of it.
- 1 epoch; the curve suggests more data would still help.

## Next
- Experiment A and E2-style full held-out BPC on this adapter (checkpoints 0, 100, 275, 534 ≈ 0, 0.19, 0.5, 1.0 epoch).
- The 10 questions and 6 continuations from E2.
