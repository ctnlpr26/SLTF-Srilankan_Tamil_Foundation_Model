#!/usr/bin/env python3
"""
Build the SLTFM Sri Lankan Tamil corpus (Noolaham books + articles) as a
Hugging Face dataset and push it to the Hub.

Expected input (one folder per document):
  <root>/processed_tamil_books/0001_<title>/content_postprocessed.txt  (preferred)
                                          /content.txt                  (fallback)
  <root>/article_tamil/<id>_<title>/...                                  (same layout)

Two configs are pushed to the same repo, sharing ONE document-level split:
  sentences  -> one cleaned sentence per row   (tokenizer study, like the Sinhala set)
  documents  -> one cleaned document per row   (continual pretraining)

Usage:
  huggingface-cli login
  python build_sltfm_corpus.py --root /path/to/processed_tamil_books --dry-run
  python build_sltfm_corpus.py --root /path/to/processed_tamil_books \
         --repo ctnlpr/sri-lankan-tamil-corpus --private
"""
import argparse
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path

from datasets import Dataset, DatasetDict

# ---------------------------------------------------------------- cleaning ---
ZERO_WIDTH = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff\u00ad]")
CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
HTML_TAG = re.compile(r"<[^>\n]{1,200}>")
URL = re.compile(r"(https?://|www\.)\S+")
# Noolaham scan footers / watermarks that repeat on every page
BOILERPLATE = re.compile(
    r"(noolaham\.org|aavanaham\.org|digitized by noolaham|நூலகம்\s*அறக்கட்டளை)",
    re.IGNORECASE,
)
LINE_END = re.compile(r"[.!?][\"'”’)\]]*$")
ISSUE_STAMP = re.compile(r"^\d{1,4}\s*\(\s*\d{1,3}\s*\)\s+")  # e.g. "31 (2 )" volume(issue)
PAGE_NUMBER = re.compile(r"^\s*[-–—]?\s*\d{1,4}\s*[-–—]?\s*$")
REPEAT_CHAR = re.compile(r"(.)\1{9,}")
# legacy-font conversion turns ':' into aytham; real aytham (எஃகு, அஃது) is never word-final
AYTHAM_COLON = re.compile(r"(?<=\S)ஃ(?=\s|$)")
# "இருக்கலாம்.செவிவழக்கிலும்" -> add the missing space (needs 4+ Tamil chars before,
# so dotted abbreviations like கி.மு. are untouched)
MISSING_SPACE = re.compile(r"(?<=[\u0B80-\u0BFF]{4})([.?!])(?=[\u0B80-\u0BFF])")
# legacy-font bullet that converted to a lone aytham
LONE_AYTHAM = re.compile(r"(^|(?<=\s))ஃ(?=\s|$)", re.MULTILINE)
# legacy-font conversion put pre-base vowel signs before the consonant: "ேகாட்டை" -> "கோட்டை"
PREBASE_FIRST = re.compile(r"(^|(?<=\s))([\u0BC6\u0BC7\u0BC8])([\u0B95-\u0BB9])", re.MULTILINE)
WORD_PUNCT = ".,;:!?\"'“”‘’()[]{}"


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = ZERO_WIDTH.sub("", text)
    text = CONTROL.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
    text = HTML_TAG.sub(" ", text)
    text = URL.sub(" ", text)
    text = LONE_AYTHAM.sub("", text)
    text = PREBASE_FIRST.sub(r"\3\2", text)
    text = unicodedata.normalize("NFC", text)  # recompose e.g. க + ே + ா -> கோ
    text = AYTHAM_COLON.sub(":", text)
    text = MISSING_SPACE.sub(r"\1 ", text)
    return text


def is_tamil(c: str) -> bool:
    return "\u0B80" <= c <= "\u0BFF"


def count_words(root: Path) -> Counter:
    """Corpus-wide word frequencies, used to repair words split by OCR line breaks."""
    counts = Counter()
    for _, _, f in find_documents(root):
        text = normalize(f.read_text(encoding="utf-8", errors="replace"))
        counts.update(w.strip(WORD_PUNCT) for w in text.split())
    return counts


def rejoin_split_words(text: str, counts: Counter, stats: Counter) -> str:
    """'அகத் தியர்' -> 'அகத்தியர்' when the joined form is attested in the corpus at
    least twice and is at least as frequent as the second piece on its own."""
    out_lines = []
    for line in text.split("\n"):
        out = []
        for tok in line.split(" "):
            if out and tok and out[-1]:
                a, b = out[-1], tok
                core = b.rstrip(WORD_PUNCT)
                if (core and a == a.rstrip(WORD_PUNCT) and b == b.lstrip(WORD_PUNCT)
                        and is_tamil(a[-1]) and is_tamil(core[0])):
                    joined = counts.get(a + core, 0)
                    if joined >= 2 and counts.get(core, 0) <= joined:
                        out[-1] = a + b
                        stats["rejoined_split_words"] += 1
                        continue
            out.append(tok)
        out_lines.append(" ".join(out))
    return "\n".join(out_lines)


EMPTY_BRACKETS = re.compile(r"[(\[{][\s,.;:'\"\-–]*[)\]}]")
SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?])")
REPEATED_PUNCT = re.compile(r"([,.;:])(\s*[,.;:])+")


def tidy(s: str) -> str:
    """Clean holes left by upstream removal of English text: '( )', ' ,', '. .'."""
    s = ISSUE_STAMP.sub("", s.strip())
    prev = None
    while prev != s:
        prev = s
        s = EMPTY_BRACKETS.sub(" ", s)
        s = SPACE_BEFORE_PUNCT.sub(r"\1", s)
        s = REPEATED_PUNCT.sub(r"\1", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.lstrip(",.;:-–) ").strip()


def tamil_ratio(s: str) -> float:
    """Share of letters/marks that are Tamil script (ignores digits, punctuation, spaces)."""
    letters = [c for c in s if unicodedata.category(c)[0] in "LM"]
    if not letters:
        return 0.0
    return sum("\u0B80" <= c <= "\u0BFF" for c in letters) / len(letters)


def digit_ratio(s: str) -> float:
    chars = [c for c in s if not c.isspace()]
    return sum(c.isdigit() for c in chars) / len(chars) if chars else 0.0


def to_paragraphs(text: str, line_mode: str) -> list[str]:
    lines = text.split("\n")
    # running headers/footers: short lines repeated >=3 times inside one document
    counts = Counter(l.strip() for l in lines if l.strip())
    repeated = {l for l, c in counts.items() if c >= 3 and len(l) < 80}

    # running headers printed into the text, e.g. "சித்த மருத்துவ இதழ் 2019 2020":
    # confirmed by a 3-8 word line-initial prefix containing a digit on >=10 lines,
    # trimmed to its last digit-bearing word, then removed EVERYWHERE in the document
    # (page breaks often fall mid-sentence, so the header appears mid-line too)
    headers = set()
    for k in range(8, 2, -1):
        c = Counter(tuple(l.split()[:k]) for l in lines if len(l.split()) > k)
        for words, n in c.items():
            digit_idx = [i for i, w in enumerate(words) if re.search(r"\d", w)]
            if n >= 10 and digit_idx:
                phrase = words[:digit_idx[-1] + 1]
                if len(phrase) >= 3 and any(is_tamil(w[0]) for w in phrase):
                    headers.add(" ".join(phrase))
    header_res = [re.compile(r"(?<!\S)" + re.escape(h) + r"(?!\S)")
                  for h in sorted(headers, key=len, reverse=True)]
    if header_res:
        lines = [l for l in lines]
        for i, l in enumerate(lines):
            for rx in header_res:
                l = rx.sub(" ", l)
            lines[i] = re.sub(r"[ ]{2,}", " ", l)

    kept = []
    for line in lines:
        s = ISSUE_STAMP.sub("", line.strip())
        if s and (s in repeated or PAGE_NUMBER.match(s) or BOILERPLATE.search(s)
                  or digit_ratio(s) > 0.5):  # tables / figure numbers
            continue
        kept.append(s)

    if line_mode == "auto":
        nonempty = [l for l in kept if l]
        ended = sum(bool(LINE_END.search(l)) for l in nonempty)
        line_mode = "keep" if nonempty and ended / len(nonempty) >= 0.8 else "join"

    if line_mode == "keep":  # each non-empty line is its own paragraph
        return [l for l in kept if l]

    # "join": blank lines separate paragraphs; wrapped OCR lines are joined
    paras, buf = [], []
    for s in kept + [""]:
        if s:
            buf.append(s)
        elif buf:
            p = " ".join(buf)
            p = re.sub(r"(\S)-\s+(?=\S)", r"\1", p)  # de-hyphenate line breaks
            paras.append(re.sub(r"\s+", " ", p).strip())
            buf = []
    return paras


# ------------------------------------------------------ sentence splitting ---
ABBREVIATIONS = {
    "திரு", "திருமதி", "செல்வி", "கலாநிதி", "டாக்டர்", "பேரா", "வண", "பக்",
    "Dr", "Mr", "Mrs", "Ms", "Prof", "Rev", "St", "No", "vol", "pp", "etc",
}
CANDIDATE_END = re.compile(r"[.!?]+[\"'”’)\]]*(?=\s)")
OPEN_PUNCT = "\"'“”‘’([{"


def split_sentences(p: str) -> list[str]:
    out, start = [], 0
    for m in CANDIDATE_END.finditer(p):
        before = p[start:m.start()].split()
        last = before[-1].strip(OPEN_PUNCT) if before else ""
        if m.group().startswith(".") and m.group().count(".") == 1:
            # initials (க. சி. ஜே.), short abbreviations, dotted forms (கி.மு.)
            if len(last) <= 3 or "." in last or last in ABBREVIATIONS:
                continue
            # list numbering / times like "11. 30" (years like 1947. still split)
            if last.isdigit() and len(last) <= 2:
                continue
        out.append(p[start:m.end()].strip())
        start = m.end()
    tail = p[start:].strip()
    if tail:
        out.append(tail)
    return out


def chunk_long(s: str, max_chars: int) -> list[str]:
    """Unpunctuated OCR runs can be huge; cut them at word boundaries."""
    if len(s) <= max_chars:
        return [s]
    pieces, cur = [], ""
    for w in s.split():
        if cur and len(cur) + 1 + len(w) > max_chars:
            pieces.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        pieces.append(cur)
    return pieces


def keep_sentence(s: str, args, stats: Counter) -> bool:
    if len(s) < args.min_chars:
        stats["drop_short"] += 1
        return False
    if tamil_ratio(s) < args.min_tamil_ratio:
        stats["drop_non_tamil"] += 1
        return False
    if digit_ratio(s) > 0.3:
        stats["drop_numeric"] += 1
        return False
    if REPEAT_CHAR.search(s):
        stats["drop_repeated_chars"] += 1
        return False
    return True


# ------------------------------------------------------------ discovery ---
def find_documents(root: Path):
    """Yield (source, doc_dir, text_file) for every folder holding content."""
    for doc_dir in sorted(p for p in root.rglob("*") if p.is_dir()):
        for name in ("content_postprocessed.txt", "content.txt"):
            f = doc_dir / name
            if f.is_file():
                rel = doc_dir.relative_to(root).parts
                source = "articles" if any("article" in x.lower() for x in rel) else "books"
                yield source, doc_dir, f
                break


def title_from_folder(name: str) -> str:
    return re.sub(r"^\d+_", "", name).strip()


# ----------------------------------------------------------------- main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--repo", default="ctnlpr/sri-lankan-tamil-corpus")
    ap.add_argument("--out", default="sltfm_corpus_out", type=Path)
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--card-only", action="store_true",
                    help="only rewrite the Hub README from <out>/build_stats.json (no rebuild/upload)")
    ap.add_argument("--dry-run", action="store_true", help="build + save locally, no push")
    ap.add_argument("--line-mode", choices=["auto", "join", "keep"], default="auto",
                    help="auto: keep lines if >=80%% end in . ! ? (already-segmented text), else join wrapped OCR lines")
    ap.add_argument("--no-rejoin", action="store_true",
                    help="skip repair of words split by OCR line breaks")
    ap.add_argument("--val-ratio", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-chars", type=int, default=10)
    ap.add_argument("--max-chars", type=int, default=1000)
    ap.add_argument("--min-tamil-ratio", type=float, default=0.6)
    args = ap.parse_args()

    if args.card_only:
        stats_file = args.out / "build_stats.json"
        push_card(args.repo, json.loads(stats_file.read_text(encoding="utf-8")))
        print(f"README updated: https://huggingface.co/datasets/{args.repo}")
        return

    stats = Counter()
    counts = None if args.no_rejoin else count_words(args.root)
    docs = []
    seen_doc_hashes = set()

    for source, doc_dir, f in find_documents(args.root):
        stats[f"files_{f.name}"] += 1
        text = normalize(f.read_text(encoding="utf-8", errors="replace"))
        if counts is not None:
            text = rejoin_split_words(text, counts, stats)
        paragraphs_out = []
        for para in to_paragraphs(text, args.line_mode):
            sents = []
            for s in split_sentences(para):
                s = tidy(s)
                for piece in chunk_long(s, args.max_chars):
                    if keep_sentence(piece, args, stats):
                        sents.append(piece)
            if sents:
                paragraphs_out.append(sents)

        if not paragraphs_out:
            stats["drop_empty_doc"] += 1
            continue
        doc_text = "\n\n".join(" ".join(p) for p in paragraphs_out)
        h = hashlib.sha1(doc_text.encode()).hexdigest()
        if h in seen_doc_hashes:
            stats["drop_duplicate_doc"] += 1
            continue
        seen_doc_hashes.add(h)
        docs.append({
            "doc_id": f"{source}/{doc_dir.name}",
            "source": source,
            "title": title_from_folder(doc_dir.name),
            "paragraphs": paragraphs_out,
            "text": doc_text,
        })

    # document-level split, per source, so no book leaks into validation
    rng = random.Random(args.seed)
    split_of = {}
    for src in sorted({d["source"] for d in docs}):
        ids = sorted(d["doc_id"] for d in docs if d["source"] == src)
        rng.shuffle(ids)
        n_val = max(1, round(len(ids) * args.val_ratio)) if len(ids) > 1 else 0
        for i, did in enumerate(ids):
            split_of[did] = "validation" if i < n_val else "train"

    sent_rows = {"train": [], "validation": []}
    doc_rows = {"train": [], "validation": []}
    seen_sents = set()
    # train first, so validation sentences that also occur in train are removed
    for split in ("train", "validation"):
        for d in (x for x in docs if split_of[x["doc_id"]] == split):
            n = 0
            for sents in d["paragraphs"]:
                for s in sents:
                    key = re.sub(r"\s+", " ", s)
                    if key in seen_sents:
                        stats["drop_duplicate_sentence"] += 1
                        continue
                    seen_sents.add(key)
                    sent_rows[split].append({
                        "text": s, "doc_id": d["doc_id"], "source": d["source"],
                        "title": d["title"], "sent_idx": n,
                    })
                    n += 1
            doc_rows[split].append({
                "text": d["text"], "doc_id": d["doc_id"], "source": d["source"],
                "title": d["title"], "n_chars": len(d["text"]),
            })

    sentences = DatasetDict({k: Dataset.from_list(v) for k, v in sent_rows.items()})
    documents = DatasetDict({k: Dataset.from_list(v) for k, v in doc_rows.items()})

    # local copies: plain-text files feed the tokenizer study directly
    args.out.mkdir(parents=True, exist_ok=True)
    for split, rows in sent_rows.items():
        with open(args.out / f"sentences_{split}.txt", "w", encoding="utf-8") as fh:
            fh.writelines(r["text"] + "\n" for r in rows)
    summary = {
        "documents": {k: len(v) for k, v in doc_rows.items()},
        "sentences": {k: len(v) for k, v in sent_rows.items()},
        "characters": {k: sum(len(r["text"]) for r in v) for k, v in sent_rows.items()},
        "by_source": dict(Counter(d["source"] for d in docs)),
        "filters": dict(stats),
    }
    (args.out / "build_stats.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nSample sentences:")
    for r in sent_rows["train"][:8]:
        print("  ", r["text"])

    if args.dry_run:
        print(f"\nDry run: nothing pushed. Inspect {args.out}/ first.")
        return

    sentences.push_to_hub(args.repo, config_name="sentences", set_default=True,
                          private=args.private)
    documents.push_to_hub(args.repo, config_name="documents", private=args.private)
    push_card(args.repo, summary)
    print(f"\nPushed: https://huggingface.co/datasets/{args.repo}")


def push_card(repo: str, s: dict):
    """Write README.md with an explicit `configs` block. Without it, `datasets` merges
    sentences/ and documents/ into one table and fails with a CastError."""
    import yaml
    from huggingface_hub import DatasetCard

    def files(cfg):
        return [{"split": sp, "path": f"{cfg}/{sp}-*"} for sp in ("train", "validation")]

    meta = {
        "language": ["ta"],
        "task_categories": ["text-generation"],
        "tags": ["tamil", "sri-lankan-tamil", "continual-pretraining",
                 "low-resource", "noolaham", "sltfm"],
        "configs": [
            {"config_name": "sentences", "default": True, "data_files": files("sentences")},
            {"config_name": "documents", "data_files": files("documents")},
        ],
    }
    body = f"""
# Sri Lankan Tamil Corpus (SLTFM)

Sri Lankan Tamil text from digitised Noolaham books and articles, prepared for the
Sri Lankan Tamil Foundation Model (SLTFM) project: tokenizer study and continual pretraining.

## Configs

| Config | Row = | Use |
|---|---|---|
| `sentences` | one cleaned sentence | tokenizer training / analysis |
| `documents` | one cleaned document (paragraphs joined by `\\n\\n`) | continual pretraining |

```python
from datasets import load_dataset
sents = load_dataset("{repo}")               # default = sentences
docs  = load_dataset("{repo}", "documents")
```

Fields: `text`, `doc_id`, `source` (books / articles), `title`, plus `sent_idx` or `n_chars`.

## Splits

The split is made at **document level** (90/10 per source), so no book appears in both
train and validation. Validation sentences that also occur in train are removed.

| | train | validation |
|---|---|---|
| documents | {s['documents']['train']:,} | {s['documents']['validation']:,} |
| sentences | {s['sentences']['train']:,} | {s['sentences']['validation']:,} |
| characters | {s['characters']['train']:,} | {s['characters']['validation']:,} |

## Processing

1. Text source: `content_postprocessed.txt`, falling back to `content.txt`; metadata files excluded.
2. Unicode NFC; zero-width and control characters, HTML tags and URLs removed;
   word-final `ஃ` (legacy-font colon) restored to `:`; missing space after `.`/`?` restored.
   Words split by OCR line breaks (`அகத் தியர்`) are rejoined when the joined form is
   attested in the corpus and is at least as frequent as the trailing fragment.
3. Noolaham footers, page numbers and per-document repeated running headers removed,
   including page headers printed mid-sentence (confirmed by a digit-bearing prefix
   repeated on 10+ line starts, then removed throughout that document);
   legacy-font artifacts repaired (lone `ஃ` bullets, misplaced pre-base vowel signs).
4. Already-segmented files keep one line per paragraph; wrapped OCR lines are joined
   and line-break hyphenation repaired.
5. Empty brackets and punctuation residue (left where upstream processing removed
   Latin-script text) cleaned up.
6. Sentence splitting on `. ! ?`, protecting Tamil initials (e.g. `க.`), dotted
   abbreviations (`கி.மு.`), common titles and short list numbers.
7. Filters: min 10 chars, Tamil-script share of letters >= 0.6, digit share <= 0.3,
   no 10+ repeated characters; unpunctuated runs cut at 1,000 chars.
8. Exact deduplication of documents and sentences.

## Limitations

- OCR errors remain; only coarse noise filtering has been applied. Near-duplicate
  (MinHash) deduplication has not been applied yet.
- Rule-based sentence splitting will occasionally split or merge incorrectly.
- Content skews toward literary, historical and formal registers of Sri Lankan Tamil.

## License / access

Source texts come from the Noolaham digital library. Redistribution terms must be
confirmed with the Noolaham Foundation before this dataset is made public.

Maintained by CTNLPR / ExentAI.
"""
    header = yaml.safe_dump(meta, sort_keys=False, allow_unicode=True)
    card = DatasetCard(f"---\n{header}---\n{body}")
    card.push_to_hub(repo, repo_type="dataset")


if __name__ == "__main__":
    main()
