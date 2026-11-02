#!/usr/bin/env python3
"""
SLTFM tokenizer study, Stages 1-4 (CPU only, no model weights needed).

Stage 1  Baseline metrics of the original Llama 3 tokenizer on Sri Lankan Tamil.
Stage 2  Tamil candidate vocabulary discovery: BPE merges learned *on top of* Llama 3's own
         segmentation, on the TRAIN split only. Each merge's frequency is exactly the number
         of tokens it removes, so the ranking implements "frequency x token reduction".
Stage 3  Llama 3 vocabulary extension: new tokens + merges appended to tokenizer.json, so the
         original vocabulary, ids and merge priorities are preserved.
Stage 4  Re-tokenization of the held-out VALIDATION split and comparison of all configurations.

Key design point (finding): Llama 3's pre-tokenizer regex uses \\p{L}+ for words. Tamil vowel
signs and the virama are combining marks (\\p{M}), so every Tamil word is cut into pieces
before BPE runs, e.g. யாழ்ப்பாணம் -> ய|ாழ|்ப|்ப|ாணம|். Merges can never cross these cuts, so
the number of pre-tokens per word is a hard floor on tokens/word. We therefore compare:

  A  original                          Llama 3 as released
  B  original pre-tokenizer + 8K       standard extension, limited by the pre-tokenizer
  C  Tamil-aware pre-tokenizer only    regex change, no new tokens
  D  Tamil-aware pre-tokenizer + 8K
  E  Tamil-aware pre-tokenizer + 16K

The Tamil-aware regex only adds the Tamil combining marks (U+0B82, U+0BBE-U+0BCD, U+0BD7) to
the letter class, so text without those characters (English, code, numbers, other scripts) is
tokenized exactly as before. The script verifies this.

Usage (PowerShell, in your .venv):
  pip install -U tokenizers huggingface_hub datasets regex
  $env:HF_TOKEN = "hf_..."            # needs access to meta-llama/Meta-Llama-3-8B
  python sltfm_tokenizer_study.py
  # optional: also compare other models' tokenizers (needs transformers + sentencepiece)
  python sltfm_tokenizer_study.py --compare-models

Outputs (in --out, default tokenizer_study_out/):
  report.md, metrics.json, per_doc_baseline.csv, top_fragmented_words.csv,
  candidates_tamil_8k.csv / _16k.csv, tokenizers/<config>/ (HF-loadable tokenizer folders)
"""
import argparse
import collections
import csv
import heapq
import json
import os
import shutil
import statistics
import time

from tokenizers import Tokenizer, decoders

TAMIL_MARKS = "ஂா-்ௗ"
SRI_LANKAN_TERMS = [  # place names and locally common words, used for a coverage check
    "யாழ்ப்பாணம்", "மட்டக்களப்பு", "திருகோணமலை", "வவுனியா", "கிளிநொச்சி", "முல்லைத்தீவு",
    "மன்னார்", "கொழும்பு", "கண்டி", "அம்பாறை", "நல்லூர்", "பருத்தித்துறை", "சாவகச்சேரி",
    "ஊர்காவற்றுறை", "வல்வெட்டித்துறை", "புங்குடுதீவு", "நெடுந்தீவு", "இலங்கை", "ஈழத்து",
    "சிங்களம்", "பனங்கட்டி", "கதைத்தல்", "நூலகம்", "சித்த", "கூட்டுறவு",
]
BUCKETS = [(1, 1, "1"), (2, 2, "2"), (3, 4, "3-4"), (5, 9, "5-9"), (10, 10 ** 9, "10+")]
NON_TAMIL_CHECK = [
    "The cooperative movement in Sri Lanka began in 1911.",
    "def f(x):\n    return x ** 2  # comment\n",
    "Prices rose 12.5% in 2024; see https://example.org/a?b=c.",
    "நூலகம் noolaham.org — Jaffna Public Library (1933)",  # mixed: only the Tamil part may change
    "हिन्दी पाठ और বাংলা লেখা, ಕನ್ನಡ ಪಠ್ಯ.",
]


def is_tamil(s):
    return any("஀" <= c <= "௿" for c in s)


# --------------------------------------------------------------------------- data
def load_splits(args):
    """Returns {"train": [(doc_id, text)], "validation": [...]}."""
    if args.local_train:
        out = {}
        for split, path in (("train", args.local_train), ("validation", args.local_validation)):
            with open(path, encoding="utf-8") as f:
                out[split] = [(f"{split}/all", f.read())]
        return out
    from datasets import load_dataset
    ds = load_dataset(args.dataset, "documents", token=os.environ.get("HF_TOKEN"))
    return {s: list(zip(ds[s]["doc_id"], ds[s]["text"])) for s in ("train", "validation")}


# --------------------------------------------------------------------------- tokenizer files
def get_base_files(args):
    if args.base_dir:
        return args.base_dir
    from huggingface_hub import hf_hub_download
    d = None
    for fn in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        try:
            p = hf_hub_download(args.base, fn, token=os.environ.get("HF_TOKEN"))
            d = os.path.dirname(p)
        except Exception as e:
            if fn == "tokenizer.json":
                raise SystemExit(f"Cannot download {args.base}/tokenizer.json: {e}\n"
                                 "Set HF_TOKEN to a token with Llama 3 access.")
    return d


def split_pattern(tj):
    """Find the regex Split inside Llama 3's pre_tokenizer (Sequence[Split, ByteLevel])."""
    pt = tj["pre_tokenizer"]
    steps = pt["pretokenizers"] if pt.get("type") == "Sequence" else [pt]
    for st in steps:
        if st.get("type") == "Split":
            return st
    raise SystemExit("No Split pre-tokenizer found: is this a Llama 3 tokenizer.json?")


def tamil_aware_pattern(p):
    tm = TAMIL_MARKS
    new = (p.replace(r"[^\r\n\p{L}\p{N}]?\p{L}+", rf"[^\r\n\p{{L}}\p{{N}}{tm}]?[\p{{L}}{tm}]+")
            .replace(r" ?[^\s\p{L}\p{N}]+", rf" ?[^\s\p{{L}}\p{{N}}{tm}]+"))
    if new == p:
        raise SystemExit("Unexpected Llama 3 regex; could not make it Tamil-aware:\n" + p)
    return new


def build_tokenizer(base_json, tamil_aware, merges, out_dir=None, base_dir=None):
    """Return a Tokenizer with (optionally) the Tamil-aware regex and appended merges."""
    tj = json.loads(json.dumps(base_json))
    st = split_pattern(tj)
    if tamil_aware:
        st["pattern"]["Regex"] = tamil_aware_pattern(st["pattern"]["Regex"])
    model = tj["model"]
    vocab = model["vocab"]
    # Pin special tokens to their original ids. On load, `tokenizers` gives added tokens that are
    # not in the model vocab the ids len(vocab), len(vocab)+1, ... ; once we append new tokens that
    # would move Llama 3's specials (e.g. <|begin_of_text|> 128000) and break the model.
    for t in tj.get("added_tokens", []):
        vocab.setdefault(t["content"], t["id"])
    next_id = max(vocab.values()) + 1
    pair_as_str = bool(model["merges"]) and isinstance(model["merges"][0], str)
    for a, b in merges:
        tok = a + b
        if tok not in vocab:
            vocab[tok] = next_id
            next_id += 1
        model["merges"].append(f"{a} {b}" if pair_as_str else [a, b])
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "tokenizer.json"), "w", encoding="utf-8") as f:
            json.dump(tj, f, ensure_ascii=False)
        for fn in ("tokenizer_config.json", "special_tokens_map.json"):
            src = os.path.join(base_dir or "", fn)
            if base_dir and os.path.exists(src):
                shutil.copy(src, os.path.join(out_dir, fn))
    return Tokenizer.from_str(json.dumps(tj)), next_id


# --------------------------------------------------------------------------- stage 2: BPE on top of Llama 3
def pretoken_counts(tok, docs):
    """Counts of Tamil-containing pre-tokens (byte-level strings) under tok's pre-tokenizer."""
    counts = collections.Counter()
    for _, text in docs:
        for piece, (s, e) in tok.pre_tokenizer.pre_tokenize_str(text):
            if is_tamil(text[s:e]):
                counts[piece] += 1
    return counts


def learn_merges(tok, counts, n_merges, min_freq=2, log_every=2000):
    """Greedy BPE starting from Llama 3's own segmentation of each pre-token."""
    words = []  # [symbols(list[str]), freq]
    for piece, f in counts.items():
        words.append([[t.value for t in tok.model.tokenize(piece)], f])
    pair_cnt = collections.Counter()
    where = collections.defaultdict(set)
    for wi, (sym, f) in enumerate(words):
        for p in zip(sym, sym[1:]):
            pair_cnt[p] += f
            where[p].add(wi)
    heap = [(-c, p) for p, c in pair_cnt.items()]
    heapq.heapify(heap)
    merges, t0 = [], time.time()
    while heap and len(merges) < n_merges:
        negc, pair = heapq.heappop(heap)
        c = pair_cnt.get(pair, 0)
        if -negc != c or c <= 0:
            continue  # stale heap entry
        if c < min_freq:
            break
        a, b = pair
        new = a + b
        merges.append((a, b, c))
        changed = set()
        for wi in list(where[pair]):
            sym, f = words[wi]
            if len(sym) < 2:
                continue
            for p in zip(sym, sym[1:]):
                pair_cnt[p] -= f
                changed.add(p)
            out, i = [], 0
            while i < len(sym):
                if i < len(sym) - 1 and sym[i] == a and sym[i + 1] == b:
                    out.append(new)
                    i += 2
                else:
                    out.append(sym[i])
                    i += 1
            words[wi][0] = out
            for p in zip(out, out[1:]):
                pair_cnt[p] += f
                where[p].add(wi)
                changed.add(p)
        pair_cnt.pop(pair, None)
        where.pop(pair, None)
        for p in changed:
            if pair_cnt.get(p, 0) > 0:
                heapq.heappush(heap, (-pair_cnt[p], p))
        if log_every and len(merges) % log_every == 0:
            print(f"    {len(merges):,} merges (last freq {c:,}) [{time.time() - t0:.0f}s]", flush=True)
    return merges


# --------------------------------------------------------------------------- metrics
def word_token_counts(tok, word_counts):
    return {w: len(tok.encode(" " + w, add_special_tokens=False).ids) for w in word_counts}


def metrics(tok, docs, label, orig_total=None, new_id_start=None, pretok_floor=True):
    n_tok = n_chars = 0
    used_new = set()
    per_doc = []
    word_counts = collections.Counter()
    for doc_id, text in docs:
        ids = tok.encode(text, add_special_tokens=False).ids
        n_tok += len(ids)
        n_chars += len(text)
        if new_id_start is not None:
            used_new.update(i for i in ids if i >= new_id_start)
        words = text.split()
        word_counts.update(words)
        per_doc.append((doc_id, len(text), len(words), len(ids)))
    wtc = word_token_counts(tok, word_counts)
    tam = {w: c for w, c in word_counts.items() if is_tamil(w)}
    n_words, n_tam = sum(word_counts.values()), sum(tam.values())
    tam_tokens = sum(wtc[w] * c for w, c in tam.items())
    bucket = collections.Counter()
    for w, c in tam.items():
        k = wtc[w]
        for lo, hi, name in BUCKETS:
            if lo <= k <= hi:
                bucket[name] += c
    m = {
        "config": label,
        "total_tokens": n_tok,
        "characters": n_chars,
        "chars_per_token": n_chars / n_tok,
        "tokens_per_word_all": n_tok / max(n_words, 1),
        "tokens_per_tamil_word": tam_tokens / max(n_tam, 1),
        "single_token_retention_pct": 100 * bucket["1"] / max(n_tam, 1),
        "buckets_pct": {name: 100 * bucket[name] / max(n_tam, 1) for _, _, name in BUCKETS},
        "max_fragmentation": max((wtc[w] for w in tam), default=0),
        "sri_lankan_terms_tokens": {t: len(tok.encode(" " + t, add_special_tokens=False).ids)
                                    for t in SRI_LANKAN_TERMS},
    }
    m["sri_lankan_terms_mean_tokens"] = statistics.mean(m["sri_lankan_terms_tokens"].values())
    if orig_total:
        m["token_reduction_pct"] = 100 * (orig_total - n_tok) / orig_total
    if new_id_start is not None:
        m["new_tokens_used_on_this_split"] = len(used_new)
    if pretok_floor:
        pieces = sum(len(tok.pre_tokenizer.pre_tokenize_str(" " + w)) * c for w, c in tam.items())
        m["pretokens_per_tamil_word_floor"] = pieces / max(n_tam, 1)
    return m, per_doc, wtc, tam


# --------------------------------------------------------------------------- checks
def check_preservation(orig, cand, label, base_json=None):
    ok = True
    for t in (base_json or {}).get("added_tokens", []):
        if cand.token_to_id(t["content"]) != t["id"]:
            ok = False
            print(f"  [{label}] special token {t['content']} moved to id {cand.token_to_id(t['content'])}")
            break
    for s in NON_TAMIL_CHECK:
        if is_tamil(s):
            continue
        if orig.encode(s, add_special_tokens=False).ids != cand.encode(s, add_special_tokens=False).ids:
            ok = False
            print(f"  [{label}] non-Tamil text tokenized differently: {s[:40]!r}")
    return ok


def check_roundtrip(tok, docs, label, n_chars=200_000):
    text = "".join(t for _, t in docs)[:n_chars]
    ok = tok.decode(tok.encode(text, add_special_tokens=False).ids) == text
    if not ok:
        print(f"  [{label}] WARNING: decode(encode(x)) != x")
    return ok


def readable(tokens):
    d = decoders.ByteLevel()
    return d.decode(tokens)


# --------------------------------------------------------------------------- optional comparison
def compare_models(models, docs):
    from transformers import AutoTokenizer
    rows = []
    text = "\n".join(t for _, t in docs)
    words = text.split()
    tam_words = [w for w in words if is_tamil(w)]
    for mid in models:
        try:
            t = AutoTokenizer.from_pretrained(mid, token=os.environ.get("HF_TOKEN"))
            n = len(t(text, add_special_tokens=False)["input_ids"])
            sample = tam_words[:20000]
            nt = sum(len(t(" " + w, add_special_tokens=False)["input_ids"]) for w in sample)
            rows.append({"model": mid, "vocab": len(t), "chars_per_token": len(text) / n,
                         "tokens_per_word_all": n / len(words),
                         "tokens_per_tamil_word": nt / len(sample)})
            print(f"  {mid}: {rows[-1]['chars_per_token']:.2f} chars/token")
        except Exception as e:
            print(f"  {mid}: skipped ({type(e).__name__}: {str(e)[:120]})")
    return rows


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="meta-llama/Meta-Llama-3-8B")
    ap.add_argument("--base-dir", help="local folder with tokenizer.json (skips download)")
    ap.add_argument("--dataset", default="ctnlpr/sri-lankan-tamil-corpus")
    ap.add_argument("--local-train", help="use a local text file instead of the HF dataset")
    ap.add_argument("--local-validation")
    ap.add_argument("--sizes", default="8000,16000")
    ap.add_argument("--min-freq", type=int, default=2)
    ap.add_argument("--out", default="tokenizer_study_out")
    ap.add_argument("--compare-models", action="store_true")
    args = ap.parse_args()
    sizes = [int(s) for s in args.sizes.split(",")]
    os.makedirs(args.out, exist_ok=True)
    t_start = time.time()

    print("Loading data ...")
    data = load_splits(args)
    print(f"  train docs={len(data['train'])}  validation docs={len(data['validation'])}")

    base_dir = get_base_files(args)
    base_json = json.load(open(os.path.join(base_dir, "tokenizer.json"), encoding="utf-8"))
    orig, base_next = build_tokenizer(base_json, False, [])
    tamil0, _ = build_tokenizer(base_json, True, [])
    print(f"  base tokenizer: {base_dir}  (model vocab {len(base_json['model']['vocab']):,}, "
          f"first free id {base_next:,})")

    # ---- Stage 1
    print("\nStage 1: baseline metrics (validation) ...")
    results = {}
    m, per_doc, wtc, tam = metrics(orig, data["validation"], "A original")
    results["A"] = m
    orig_total = m["total_tokens"]
    with open(os.path.join(args.out, "per_doc_baseline.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["doc_id", "chars", "words", "tokens", "chars_per_token", "tokens_per_word"])
        for d, c, n, t in per_doc:
            w.writerow([d, c, n, t, round(c / max(t, 1), 3), round(t / max(n, 1), 3)])
    with open(os.path.join(args.out, "top_fragmented_words.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["word", "tokens", "frequency"])
        for word in sorted(tam, key=lambda x: (-wtc[x], -tam[x]))[:300]:
            w.writerow([word, wtc[word], tam[word]])
    print(f"  {m['chars_per_token']:.3f} chars/token | {m['tokens_per_tamil_word']:.2f} tokens/Tamil word | "
          f"floor from pre-tokenizer {m['pretokens_per_tamil_word_floor']:.2f}")

    # ---- Stage 2 + 3
    print("\nStage 2: learning Tamil merges on TRAIN ...")
    plan = [("B", False, sizes[0]), ] + [(chr(ord("D") + i), True, s) for i, s in enumerate(sizes)]
    learned = {}
    for tamil_aware in (False, True):
        need = max(s for k, ta, s in plan if ta == tamil_aware)
        base_tok = tamil0 if tamil_aware else orig
        name = "Tamil-aware" if tamil_aware else "original"
        counts = pretoken_counts(base_tok, data["train"])
        print(f"  [{name} pre-tokenizer] {len(counts):,} unique Tamil pre-tokens; learning {need:,} merges")
        learned[tamil_aware] = learn_merges(base_tok, counts, need, args.min_freq)
        print(f"  [{name}] learned {len(learned[tamil_aware]):,} merges")

    configs = [("C", True, 0)] + plan
    tokenizers = {"A": orig}
    for key, ta, n in configs:
        merges = learned[ta][:n]
        label = {"B": f"B original pre-tok + {n // 1000}K", "C": "C Tamil-aware pre-tok only"}.get(
            key, f"{key} Tamil-aware pre-tok + {n // 1000}K")
        folder = os.path.join(args.out, "tokenizers", label.split(" ", 1)[1].replace(" ", "_").replace("+", "plus"))
        tok, _ = build_tokenizer(base_json, ta, [(a, b) for a, b, _ in merges], folder, base_dir)
        tokenizers[key] = tok
        if n:
            fn = f"candidates_{'tamil' if ta else 'origpretok'}_{n // 1000}k.csv"
            with open(os.path.join(args.out, fn), "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["rank", "token_text", "left", "right", "train_frequency_at_merge(=tokens_saved)"])
                for r, (a, b, c) in enumerate(merges, 1):
                    w.writerow([r, readable([a + b]), readable([a]), readable([b]), c])

        # ---- Stage 4
        print(f"\nStage 4: {label} on validation ...")
        m, *_ = metrics(tok, data["validation"], label, orig_total, base_next if n else None)
        m["preserves_non_tamil_tokenization"] = check_preservation(orig, tok, label, base_json)
        m["lossless_roundtrip"] = check_roundtrip(tok, data["validation"], label)
        m["added_tokens"] = len({a + b for a, b, _ in merges})
        m["tokenizer_folder"] = folder
        results[key] = m
        print(f"  {m['chars_per_token']:.3f} chars/token | {m['tokens_per_tamil_word']:.2f} tok/Tamil word | "
              f"reduction {m['token_reduction_pct']:.1f}% | preserves non-Tamil: {m['preserves_non_tamil_tokenization']}")

    results["A"]["preserves_non_tamil_tokenization"] = True
    results["A"]["lossless_roundtrip"] = check_roundtrip(orig, data["validation"], "A")
    results["A"]["added_tokens"] = 0
    results["A"]["token_reduction_pct"] = 0.0

    comparison = []
    if args.compare_models:
        print("\nComparing other tokenizers on validation ...")
        comparison = compare_models([
            "meta-llama/Llama-2-7b-hf", "meta-llama/Meta-Llama-3-8B", "google/gemma-2-9b",
            "Qwen/Qwen2.5-7B", "abhinand/tamil-llama-7b-base-v0.1", "google/mt5-base",
            "ai4bharat/IndicBERTv2-MLM-only",
        ], data["validation"])

    # ---- report
    order = ["A", "B", "C"] + [chr(ord("D") + i) for i in range(len(sizes))]
    rows = [
        ("Added tokens", "added_tokens", "{:,}"),
        ("Total tokens (validation)", "total_tokens", "{:,}"),
        ("Token reduction vs. original", "token_reduction_pct", "{:.1f}%"),
        ("Characters / token", "chars_per_token", "{:.3f}"),
        ("Tokens / word (all words)", "tokens_per_word_all", "{:.2f}"),
        ("Tokens / Tamil word", "tokens_per_tamil_word", "{:.2f}"),
        ("Pre-tokenizer floor (pieces / Tamil word)", "pretokens_per_tamil_word_floor", "{:.2f}"),
        ("Single-token retention", "single_token_retention_pct", "{:.1f}%"),
        ("Max fragmentation (tokens)", "max_fragmentation", "{}"),
        ("Sri Lankan terms: mean tokens", "sri_lankan_terms_mean_tokens", "{:.2f}"),
        ("New tokens used on validation", "new_tokens_used_on_this_split", "{:,}"),
        ("Non-Tamil text + special-token ids unchanged", "preserves_non_tamil_tokenization", "{}"),
        ("Lossless round-trip", "lossless_roundtrip", "{}"),
    ]
    L = ["# SLTFM tokenizer study: results (validation split, held out)", "",
         "| Metric | " + " | ".join(results[k]["config"] for k in order) + " |",
         "|---|" + "---|" * len(order)]
    for name, key, fmt in rows:
        cells = []
        for k in order:
            v = results[k].get(key)
            cells.append("-" if v is None else fmt.format(v))
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    L += ["", "**Fragmentation buckets (% of Tamil word occurrences, by tokens per word)**", "",
          "| Bucket | " + " | ".join(results[k]["config"] for k in order) + " |", "|---|" + "---|" * len(order)]
    for _, _, b in BUCKETS:
        L.append(f"| {b} | " + " | ".join(f"{results[k]['buckets_pct'][b]:.1f}%" for k in order) + " |")
    L += ["", "**Sri Lankan terms: tokens per word**", "",
          "| Term | " + " | ".join(k for k in order) + " |", "|---|" + "---|" * len(order)]
    for t in SRI_LANKAN_TERMS:
        L.append(f"| {t} | " + " | ".join(str(results[k]["sri_lankan_terms_tokens"][t]) for k in order) + " |")
    if comparison:
        L += ["", "**Other tokenizers (validation)**", "",
              "| Tokenizer | Vocab | Chars/token | Tokens/word | Tokens/Tamil word |", "|---|---|---|---|---|"]
        for r in comparison:
            L.append(f"| {r['model']} | {r['vocab']:,} | {r['chars_per_token']:.2f} | "
                     f"{r['tokens_per_word_all']:.2f} | {r['tokens_per_tamil_word']:.2f} |")
    L += ["", "Notes:",
          "- Merges were learned on the train split only; all numbers are on the held-out validation split.",
          "- Tokens/word uses whitespace words; OCR-merged words inflate it, so chars/token is the headline metric.",
          "- Pre-tokenizer floor = number of pre-tokens per Tamil word: no vocabulary can produce fewer tokens.",
          "- Tamil-aware configs only add Tamil combining marks to the regex letter class; non-Tamil text is unchanged.",
          f"- Runtime: {time.time() - t_start:.0f}s."]
    report = "\n".join(L)
    open(os.path.join(args.out, "report.md"), "w", encoding="utf-8").write(report)
    json.dump({"results": results, "comparison": comparison, "args": vars(args)},
              open(os.path.join(args.out, "metrics.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n" + report)
    print(f"\nSaved to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
