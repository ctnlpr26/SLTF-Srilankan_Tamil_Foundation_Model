# Decisions log

Newest decisions at the bottom. Add a row for every decision that changes the method.

| Date | Decision | Reason / evidence | Decided by | Phase |
|---|---|---|---|---|
| 2026-10-06 | Corpus split by document (90/10 per source); sentences + documents configs on HF | Prevents book-level leakage between train and validation | Vithyabavan | Corpus |
| 2026-10-07 | Run CPT with the original Llama 3 tokenizer first | Baseline before any vocabulary change | Anna | CPT |
| 2026-10-07 | QLoRA (4-bit NF4, r=64) on Kaggle 2x T4, checkpoints pushed to HF every 25 steps | Llama 3 8B does not fit a T4 in fp16; free compute; resume across 12 h sessions | Vithyabavan | CPT |
| 2026-10-09 | Release checkpoint-1100 (step 1100 of ~1,106) as the CPT model | End of epoch 2; approved | Anna | CPT |
| 2026-10-09 | Report bits per character, not only perplexity | Llama 3 tokenizes Tamil near byte level (0.74 chars/token); BPC is comparable across tokenizers | Vithyabavan | Evaluation |
| 2026-10-14 | Pause the tokenizer study; move to instruction tuning | Project priority | Anna | Tokenizer |
| 2026-10-14 | Method: SFT after CPT, then RAG as a separate phase | SLTFM progress plan, Phases 9-10 | Anna (plan) | SFT |
| 2026-10-14 | Use an API teacher model to generate Tamil Q&A; choose it by a quality test | Approved by Anna; terms of use must allow training | Anna | SFT |
| 2026-10-14 | Generate SFT Q&A only from CPT-train documents; use early stopping | Gekhman et al. 2024 (P01): new-fact SFT examples increase hallucination | Vithyabavan (proposed) | SFT |
| 2026-10-16 | Teacher model gpt-4.1, prompt v3 | 30-passage pilot: gpt-4o-mini wrong facts and broken Tamil; gpt-4.1 kept 92-96%, 87% acceptable in a 60-row rating | Vithyabavan / Anna | SFT |
| 2026-11-02 | Phase-based repository structure | Matches the project phase plan | Anna | Documentation |
