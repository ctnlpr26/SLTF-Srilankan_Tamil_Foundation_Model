# 07 · Model evaluation & Sri Lankan Tamil benchmarking

`sltfm_eval_kaggle.ipynb` – base Llama 3 8B vs the CPT adapter:
bits per character on validation books, 10 Sri Lankan Tamil questions, text continuations.

Current result (first ~4% of validation only): BPC 1.248 → 1.024.

To do (see the paper plan):
- Full-validation BPC with confidence intervals; base-model comparison (Llama 3 vs Qwen 3.5)
- Frozen test sets: open-book Q&A on held-out books, Sri Lankan knowledge questions, dialect probe
- English forgetting (MMLU, HellaSwag, ARC, GSM8K)
- 13-token contamination check of every test set against training data
