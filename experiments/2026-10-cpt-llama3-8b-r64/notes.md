# CPT · Llama 3 8B · QLoRA r64 · Oct 2026

- Data: `ctnlpr/sri-lankan-tamil-corpus` (documents), revision in `hf_revisions.txt`
- Notebook: `06_pretraining/sltfm_llama3_cpt_kaggle.ipynb`
- 3 Kaggle sessions (0→376→758→1100), ~31 h; W&B run `llama3-8b-sltfm-cpt-r64`
- Released: checkpoint-1100 (epoch 1.99)
- Eval (first ~4% of validation): BPC 1.248 (base) → 1.024 (CPT), −17.9%; 3/10 vs 1/10 sample questions
