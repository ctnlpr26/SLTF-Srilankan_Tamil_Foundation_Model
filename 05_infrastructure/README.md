# 05 · Training infrastructure & data pipeline

Current setup (implemented inside `06_pretraining/sltfm_llama3_cpt_kaggle.ipynb`):

- Kaggle, 2× T4 (16 GB each), DDP, free GPU quota (~30 h/week, 12 h per session)
- QLoRA: 4-bit NF4 base, fp16 compute, 8-bit AdamW
- Documents packed into 2,048-token blocks with `<|end_of_text|>` between documents
- Full checkpoint (adapter, optimizer, scheduler, RNG) pushed to the Hub every 25 steps;
  a new session resumes from the latest Hub checkpoint
- Secrets via Kaggle *Add-ons → Secrets* (`HF_TOKEN`, `WANDB_API_KEY`); W&B for logging

To do: move the checkpoint push/resume code into `common/` so SFT can reuse it.
