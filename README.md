# SLTFM – Sri Lankan Tamil Foundation Model

Adapting large language models to **Sri Lankan Tamil** using authentic text from the
[Noolaham](https://noolaham.org) digital library, on free compute (Kaggle 2× T4).

Data and model weights live on the Hugging Face Hub, not in this repository:

| Resource | Hub |
|---|---|
| Corpus | [`ctnlpr/sri-lankan-tamil-corpus`](https://huggingface.co/datasets/ctnlpr/sri-lankan-tamil-corpus) |
| CPT model (Llama 3 8B, LoRA) | [`ctnlpr/Llama-3-SLTFM-8B-CPT-LoRA`](https://huggingface.co/ctnlpr/Llama-3-SLTFM-8B-CPT-LoRA) |
| Instruction data | `ctnlpr/sri-lankan-tamil-instruct` (private, in progress) |

## Phases

| Phase | Folder | Status |
|---|---|---|
| 1. Scope, requirements & architecture | [`01_scope`](01_scope) | Decisions log, literature survey |
| 2. Corpus selection, cleaning & dataset | [`02_corpus`](02_corpus) | Done (pilot, 121 books) |
| 3. Normalization & preprocessing | [`03_preprocessing`](03_preprocessing) | Done |
| 4. Tokenizer development & optimization | [`04_tokenizer`](04_tokenizer) | Efficiency analysis done; extension paused |
| 5. Training infrastructure & data pipeline | [`05_infrastructure`](05_infrastructure) | Done (inside the CPT notebook) |
| 6. Pretraining / continued pretraining | [`06_pretraining`](06_pretraining) | Llama 3 8B CPT done (checkpoint-1100) |
| 7. Evaluation & Sri Lankan Tamil benchmarking | [`07_evaluation`](07_evaluation) | Partial |
| 8. Fine-tuning for downstream tasks | [`08_finetuning`](08_finetuning) | Q&A data generation done; SFT training next |
| 9. Inference deployment & Noolaham integration | [`09_deployment`](09_deployment) | Not started |
| 10. Documentation, model card & handover | [`10_documentation`](10_documentation) | Model card done |

Shared helpers go in [`common/`](common). Every run is recorded in [`experiments/`](experiments).

## Setup (laptop scripts)

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env          # then fill in your keys; never commit .env
```

## Rules

- Never commit `.env`, API keys, corpus text, Q&A data or model weights.
- Clear notebook outputs before committing.
- One commit per change, with a clear message (`03_preprocessing: remove repeated page headers`).
- Record every training or evaluation run in `experiments/` with its config, Hub revisions and results.
