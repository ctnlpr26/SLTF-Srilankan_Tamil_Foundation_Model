# 04 · Tamil tokenizer development & optimization

| Script | What it does |
|---|---|
| `tokenizer_check.py` | Chars/token and tokens/word of several tokenizers on the validation books |
| `sltfm_tokenizer_study.py` | Llama 3 vocabulary extension study (configs A–E, Tamil-aware pre-tokenizer). **Paused.** |

**Results so far** (validation books, 1.0M characters; higher chars/token is better):

| Tokenizer | Chars/token | Tokens/word |
|---|---|---|
| Llama 3 8B | 0.74 | 13.07 |
| Qwen 2.5 7B | 0.91 | 10.58 |
| Qwen 3.5 (4B/9B Base) | 2.06 | 4.71 |

Finding: Llama 3's pre-tokenizer treats only letters (`\p{L}`) as word characters, so Tamil vowel
signs (combining marks, `\p{M}`) split every word before BPE runs.

Raw output: `experiments/2026-10-tokenizer-efficiency/tokenizer_results.txt`.
