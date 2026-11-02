---
base_model: meta-llama/Meta-Llama-3-8B
library_name: peft
license: llama3
language:
- ta
datasets:
- ctnlpr/sri-lankan-tamil-corpus
pipeline_tag: text-generation
tags:
- llama-3
- qlora
- lora
- continual-pretraining
- tamil
- sri-lankan-tamil
- low-resource
- noolaham
- sltfm
---

# Llama-3-SLTFM-8B-CPT-LoRA

**Sri Lankan Tamil Foundation Model (SLTFM) – continual-pretraining baseline.**

A LoRA adapter for **Meta Llama 3 8B (base)**, trained by continual pretraining (QLoRA) on
digitised Sri Lankan Tamil books and articles from the Noolaham digital library.
This is the **baseline run with the original Llama 3 tokenizer** (no Tamil vocabulary
expansion); it is the reference point for the SLTFM tokenizer study.

Built with Meta Llama 3.

> **Status:** experimental research checkpoint. **Released model: `checkpoint-1100`** (step 1100 of ~1,106, end of epoch 2). This is a *base* (completion) model,
> not an instruction-tuned or chat model.

## Model details

| | |
|---|---|
| Base model | `meta-llama/Meta-Llama-3-8B` |
| Method | QLoRA continual pretraining (4-bit NF4 base, double quantization, fp16 compute) |
| LoRA r / alpha / dropout | 64 / 128 / 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Trainable parameters | 167.8M (2.05% of 8.2B) |
| Tokenizer | original Llama 3 tokenizer (unchanged) |
| Language | Tamil (Sri Lankan) |
| Developed by | CTNLPR / ExentAI |

## Training data

[`ctnlpr/sri-lankan-tamil-corpus`](https://huggingface.co/datasets/ctnlpr/sri-lankan-tamil-corpus),
`documents` config: cleaned OCR text of Noolaham books and articles.

| | train | validation |
|---|---|---|
| Documents | 108 | 13 |
| Characters | 13.4M | 1.0M |
| Llama 3 tokens | 18.1M | 1.35M |

The split is made at document level, so no book appears in both train and validation.

**Tokenizer baseline on this corpus:** Llama 3 produces **0.74 characters per token** and
**12.8 tokens per word** for Sri Lankan Tamil (English is roughly 4 characters per token),
which is close to byte-level segmentation. This motivates the planned vocabulary-expansion
experiments.

## Training procedure

| | |
|---|---|
| Hardware | 2× NVIDIA T4 (16 GB), DDP, free Kaggle GPUs |
| Training time | ~31 hours, over 3 Kaggle sessions (resumed from Hub checkpoints) |
| Epochs / steps | 2 / 1,100 (released checkpoint) |
| Tokens seen | ~36.2M |
| Sequence length | 2,048 (packed documents, `<|end_of_text|>` between documents) |
| Batch | 1 per GPU × grad. accumulation 8 × 2 GPUs = 32,768 tokens per step |
| Optimizer | 8-bit AdamW (bitsandbytes), weight decay 0, max grad norm 1.0 |
| Learning rate | 2e-4, cosine schedule, warmup ratio 0.05 |
| Precision | fp16 mixed precision (T4 has no bf16) |
| Checkpointing | full checkpoint (adapter, optimizer, scheduler, RNG) every 25 steps |

Evaluation ran every 25 steps on a fixed subset of 100 validation blocks
(2,048 tokens each). Validation loss decreased steadily through training
(e.g. 0.683 → 0.661 during epoch 2).

## Evaluation (checkpoint-1100 vs. base Llama 3 8B)

**Held-out Sri Lankan Tamil text** (13 validation books, never seen in training; 44,395 characters):

| Model | Bits per character ↓ | Token perplexity ↓ |
|---|---|---|
| Llama 3 8B (base) | 1.248 | 1.90 |
| Llama 3 8B + SLTFM | **1.024** | **1.69** |

**17.9% fewer bits per character.** Bits per character is reported because Llama 3 splits
Tamil into near-byte tokens, which makes per-token perplexity misleadingly low.

**Example questions** (10 Sri Lankan Tamil questions, one-shot prompt, greedy decoding):
SLTFM answered **3/10** correctly vs. **1/10** for base Llama 3 (e.g. Jaffna's province →
வடமாகாணம்; Nallur Kandaswamy temple → யாழ்ப்பாணம், where the base model answered "Japan").

**Text continuation:** base Llama 3 switched to English on most Tamil prompts; SLTFM
continued in fluent, formal Sri Lankan Tamil, but still produces factual errors and some
repetition.

## How to use

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

base_id = "meta-llama/Meta-Llama-3-8B"           # gated: accept the Llama 3 license first
adapter_id = "ctnlpr/Llama-3-SLTFM-8B-CPT-LoRA"

tok = AutoTokenizer.from_pretrained(base_id)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(base_id, quantization_config=bnb, device_map="auto")

# the released model is checkpoint-1100 (step 1100 of ~1,106, i.e. end of epoch 2)
model = PeftModel.from_pretrained(model, adapter_id, subfolder="checkpoint-1100")
model.eval()

prompt = "யாழ்ப்பாணம் இலங்கையின் வடக்கே அமைந்துள்ள"
x = tok(prompt, return_tensors="pt").to(model.device)
y = model.generate(**x, max_new_tokens=120, do_sample=False, repetition_penalty=1.15)
print(tok.decode(y[0][x["input_ids"].shape[1]:], skip_special_tokens=True))
```

## Repository layout

- `checkpoint-1100/` – **the released adapter**.
- `checkpoint-N/` – every intermediate checkpoint (adapter + optimizer, scheduler and RNG
  state), saved every 25 steps, so training can be resumed or an earlier step selected.
- `sltfm_run_config.json` – exact training configuration.

## Intended use

Research on Sri Lankan Tamil language modelling: a baseline for comparison with
tokenizer-expanded models, and a starting point for further pretraining or
instruction tuning.

## Limitations

- **Small corpus** (~18M tokens). In epoch 2 the train loss fell faster than the validation
  loss, so the model partly memorises the corpus; intermediate checkpoints are kept so the
  best one can be selected.
- **OCR noise** in the source books remains in the training data.
- **Narrow domain:** mostly literary, historical, medical-magazine and formal registers;
  little conversational or modern colloquial Tamil.
- **Inefficient tokenization** (12.8 tokens per word) limits how much Tamil text fits in
  the context window.
- Not instruction-tuned, not safety-tuned; it can produce incorrect or biased text.

## License

Llama 3 Community License. Derivatives must include "Llama 3" in the model name and
display "Built with Meta Llama 3". Source texts come from the Noolaham digital library;
redistribution terms of the training data are being confirmed with the Noolaham Foundation.

## Contact

CTNLPR / ExentAI
