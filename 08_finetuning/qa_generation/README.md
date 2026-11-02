# Q&A generation for instruction tuning

`sltfm_generate_qa.py` generates Sri Lankan Tamil question-answer pairs from passages of the
CPT **train** books with an OpenAI teacher model, filters them, and builds SFT-ready data.

```bash
python sltfm_generate_qa.py generate --model gpt-4.1 --n-passages 30 --out pilot_run   # needs .env
python sltfm_generate_qa.py build --out pilot_run --review-n 60
python sltfm_generate_qa.py push  --out pilot_run      # private HF dataset
```

- Prompt version v3: passage quality and topic tags, `time_sensitive` flag, question-clarity check
- Filters: Tamil ratio, length, evidence must match the passage, answerability, no references to
  "the passage", near-duplicates
- Each pair is stored twice: `closed_book` (question only) and `open_book` (passage + question)
- Output (`sft_train.jsonl`, `sft_validation.jsonl`, `review_sample.csv`) is **not** committed

Pilot (30 passages, gpt-4.1, v3): 87% of 60 rated rows acceptable. Record:
`experiments/2026-10-qa-pilot/`.
