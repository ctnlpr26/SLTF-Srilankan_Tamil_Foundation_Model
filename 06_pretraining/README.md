# 06 · Foundation model pretraining / continued pretraining

`sltfm_llama3_cpt_kaggle.ipynb` – QLoRA continual pretraining of Llama 3 8B (base) on the
`documents` config of the corpus. Instructions are in the first cell.

| Setting | Value |
|---|---|
| LoRA r / alpha / dropout | 64 / 128 / 0.05, all attention + MLP projections |
| Sequence length | 2,048 |
| Effective batch | 32,768 tokens per step (1 × 8 accumulation × 2 GPUs) |
| Learning rate | 2e-4, cosine, warmup 5% |
| Epochs / steps | 2 / 1,100 (released checkpoint) |
| Time | ~31 h over 3 Kaggle sessions |

Result: [`ctnlpr/Llama-3-SLTFM-8B-CPT-LoRA`](https://huggingface.co/ctnlpr/Llama-3-SLTFM-8B-CPT-LoRA),
`checkpoint-1100`. Run record: `experiments/2026-10-cpt-llama3-8b-r64/`.
