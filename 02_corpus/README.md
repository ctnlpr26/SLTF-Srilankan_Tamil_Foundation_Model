# 02 · Corpus selection, cleaning & dataset

`build_sltfm_corpus.py` turns the preprocessed Noolaham books and articles into the Hugging Face
dataset [`ctnlpr/sri-lankan-tamil-corpus`](https://huggingface.co/datasets/ctnlpr/sri-lankan-tamil-corpus).

**Input** (output of `03_preprocessing`), one folder per document:
```
<root>/processed_tamil_books/0001_<title>/content_postprocessed.txt
<root>/article_tamil/<id>_<title>/content_postprocessed.txt
```

**Output**: two configs sharing one document-level train/validation split (90/10):
- `documents` – one cleaned document per row (used for continual pretraining)
- `sentences` – one sentence per row (tokenizer study)

**Run**
```bash
huggingface-cli login
python build_sltfm_corpus.py --root <path> --dry-run                    # build locally, no push
python build_sltfm_corpus.py --root <path> --repo ctnlpr/sri-lankan-tamil-corpus --private
python build_sltfm_corpus.py --root <path> --card-only                  # update the dataset card only
```

Pilot corpus: 121 documents (108 train / 13 validation), 13.4M train characters, chosen to be
medical-heavy.
