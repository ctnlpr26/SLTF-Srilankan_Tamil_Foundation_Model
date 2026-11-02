#!/usr/bin/env python3
"""
SLTFM: generate Sri Lankan Tamil instruction-tuning data (Q&A) from the Noolaham corpus
with an OpenAI teacher model, filter it, and push it to the Hugging Face Hub.

Design (see the SLTFM instruction-tuning study):
  * Passages come ONLY from the CPT *train* split, so every fact is one the CPT model has
    seen (Gekhman et al., 2024) and the CPT validation books stay clean for testing.
  * Each pair is grounded: the teacher must quote the supporting sentence; the quote is
    checked against the passage, and (optionally) a second call re-answers the question
    from the passage to confirm it is answerable.
  * Two formats per pair: closed-book (question only) and open-book (passage + question).
  * Everything is resumable and logged with provenance (model, prompt version, passage).

Commands (PowerShell, in your .venv):
  pip install -U openai datasets huggingface_hub pyyaml python-dotenv
  # put your keys in a .env file next to this script:
  #   OPENAI_API_KEY=sk-...
  #   HF_TOKEN=hf_...
  #   OPENAI_MODEL=<model name>
  #   PRICE_IN_PER_1M=...      (optional, USD, for the cost estimate)
  #   PRICE_OUT_PER_1M=...     (optional)
  #   HF_REPO=ctnlpr/sri-lankan-tamil-instruct   (optional)

  # 1) pilot: 30 passages, then review sft_out/review_sample.csv by hand
  python sltfm_generate_qa.py generate --model <openai-model> --n-passages 30
  python sltfm_generate_qa.py build

  # 2) scale up (resumes; already-done passages are skipped)
  python sltfm_generate_qa.py generate --model <openai-model> --n-passages 1500
  python sltfm_generate_qa.py build

  # 3) push (private) to the Hub
  python sltfm_generate_qa.py push --repo ctnlpr/sri-lankan-tamil-instruct

  # offline test of the whole pipeline without an API key:
  python sltfm_generate_qa.py generate --mock --n-passages 20 && python sltfm_generate_qa.py build
"""
import argparse
import collections
import concurrent.futures as cf
import csv
import hashlib
import json
import os
import random
import re
import threading
import time
import unicodedata
from datetime import datetime, timezone
from difflib import SequenceMatcher

PROMPT_VERSION = "v3"
TASK_TYPES = ["factual", "why_how", "term_meaning", "summary", "list"]
QUALITY = ["good", "noisy_but_usable", "unusable"]
TOPICS = ["history", "literature", "religion_philosophy", "medicine_health", "education", "society_culture",
          "politics_law", "science_agriculture", "biography", "other"]
SYSTEM_TA = "நீங்கள் இலங்கைத் தமிழில் தெளிவாகவும் சரியாகவும் பதிலளிக்கும் உதவியாளர்."
OPEN_BOOK_TEMPLATE = "கீழே உள்ள பகுதியை வாசித்து, கேள்விக்குப் பதிலளிக்கவும்.\n\nபகுதி:\n{passage}\n\nகேள்வி: {question}"

GEN_SYSTEM = """You create instruction-tuning data in SRI LANKAN TAMIL from a passage taken from a Sri Lankan Tamil book (Noolaham digital library).

Step 1 - judge the passage first:
- "passage_quality": "good"; "noisy_but_usable" (OCR errors, but the meaning of the sentences you use is clear); "unusable" (broken words, mixed-up lines or columns, meaning unclear). If "unusable", return an EMPTY items list.
- "topic": the main subject of the passage.
- "time_sensitive": true if the passage states facts that may be outdated today (medical or health claims, treatments, statistics, prices, laws, "current" office holders, technology); otherwise false.

Step 2 - write the items. Rules:
1. Write questions and answers ONLY in Tamil script, in standard Sri Lankan Tamil (Eelam Tamil) usage and spelling. Do not switch to English except for names that the passage itself writes in English.
2. Every answer must be fully supported by the passage. Do NOT add facts that are not in the passage, even if you know them. Do not change the meaning (e.g. "stop when the child is full" must not become "give enough food").
3. Use only sentences whose meaning is clear. Skip garbled parts of the passage completely.
4. Questions must be grammatical, natural Tamil that a Sri Lankan Tamil speaker would actually ask. Read each question again before returning it; if any word is not a real Tamil word, rewrite it.
5. Questions must make sense WITHOUT the passage (a reader who has not seen it must understand what is asked). Never write "இந்தப் பகுதியில்", "மேலே உள்ள", "ஆசிரியர் கூறுவது போல" or similar references to the passage. Name the topic explicitly instead.
6. Make questions specific: include the names, places, works or terms they are about. Use the passage's own terms (keep Sri Lankan Tamil terms such as வைத்தியசாலை, பாடசாலை; do not replace them with Indian Tamil words).
7. Answers: complete, natural Tamil sentences; 1-4 sentences (summary answers may be up to 6). Write answers in clean Tamil: fix obvious OCR spelling errors and joined/split words; never copy broken words into the answer.
8. For each item give "evidence": one sentence copied EXACTLY (character for character, including its OCR errors) from the passage that supports the answer.
9. Task types - label each item correctly:
   factual = who/what/when/where; why_how = reasons or processes; term_meaning = meaning of a term the passage explains;
   summary = the main points of a topic, combining SEVERAL sentences of the passage; list = items the passage enumerates - give ALL the listed items and nothing extra.
10. Medicine and health: ask only descriptive questions about what the text explains (causes, terms, history, the methods it describes). Do NOT write questions that ask for personal medical advice ("ஒரு நோயாளி என்ன செய்ய வேண்டும்?").
11. Fewer good items are better than many weak ones. If the passage has no clear factual content, return an empty list.
Return JSON only."""

GEN_USER = """Passage (from: {title}):
\"\"\"
{passage}
\"\"\"

Create up to {k} diverse question-answer items following the rules."""

GEN_SCHEMA = {
    "name": "qa_items",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "passage_quality": {"type": "string", "enum": QUALITY},
            "topic": {"type": "string", "enum": TOPICS},
            "time_sensitive": {"type": "boolean"},
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "task_type": {"type": "string", "enum": TASK_TYPES},
                        "question": {"type": "string"},
                        "answer": {"type": "string"},
                        "evidence": {"type": "string"},
                    },
                    "required": ["task_type", "question", "answer", "evidence"],
                },
            }
        },
        "required": ["passage_quality", "topic", "time_sensitive", "items"],
    },
}

VERIFY_SYSTEM = """You check a question written from a passage.
1. answerable: can the question be answered using only the passage? If not, false. Give the answer in Tamil, using only the passage.
2. question_clear: is the question grammatical Tamil with only real Tamil words (no typos), AND would a reader who has NOT seen the passage understand exactly what is asked (it names its topic; it does not treat a heading, section number or fragment as if it were a person or thing)? If not, false."""
VERIFY_USER = """Passage:
\"\"\"
{passage}
\"\"\"

Question: {question}"""
VERIFY_SCHEMA = {
    "name": "verify",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {"answerable": {"type": "boolean"}, "question_clear": {"type": "boolean"},
                       "answer": {"type": "string"}},
        "required": ["answerable", "question_clear", "answer"],
    },
}

PASSAGE_REF = re.compile(r"(இந்தப்\s*பகுதி|இப்பகுதி|மேலே\s*உள்ள|மேற்கண்ட|இந்தப்\s*பத்தி|கொடுக்கப்பட்ட\s*பகுதி|ஆசிரியர்\s*கூறு|"
                         r"இந்தக்\s*கட்டுரை|இக்கட்டுரை|இந்த\s*நூல்|இந்நூல்|இந்தப்\s*புத்தக|இப்புத்தக|இந்த\s*அத்தியாய|இவ்வத்தியாய)")


# --------------------------------------------------------------------------- text utils
def tamil_ratio(s):
    letters = [c for c in s if unicodedata.category(c)[0] in "LM"]
    return sum("஀" <= c <= "௿" for c in letters) / len(letters) if letters else 0.0


def norm(s):
    s = unicodedata.normalize("NFC", s)
    return re.sub(r"[\s\"'“”‘’.,;:!?()\[\]-]+", " ", s).strip()


def char_ngrams(s, n=3):
    s = norm(s).replace(" ", "")
    return {s[i:i + n] for i in range(max(len(s) - n + 1, 1))}


def jaccard(a, b):
    return len(a & b) / max(len(a | b), 1)


def evidence_score(evidence, passage):
    """1.0 = exact substring; otherwise best fuzzy match of the evidence inside the passage."""
    e, p = norm(evidence), norm(passage)
    if not e:
        return 0.0
    if e in p:
        return 1.0
    m = SequenceMatcher(None, p, e, autojunk=False)
    return sum(b.size for b in m.get_matching_blocks()) / len(e)


# --------------------------------------------------------------------------- passages
def load_train_docs(dataset, local_docs=None):
    if local_docs:  # jsonl with doc_id/title/text (offline testing)
        with open(local_docs, encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]
    from datasets import load_dataset
    ds = load_dataset(dataset, "documents", split="train", token=os.environ.get("HF_TOKEN"))
    return [{"doc_id": r["doc_id"], "title": r["title"], "text": r["text"]} for r in ds]


def make_passages(docs, min_chars=500, max_chars=1500, min_tamil=0.75):
    out = []
    for d in docs:
        paras = [p.strip() for p in d["text"].split("\n\n") if p.strip()]
        buf = ""
        chunks = []
        for p in paras:
            if len(p) > max_chars:  # long OCR paragraph: cut at sentence ends
                sents = re.split(r"(?<=[.!?])\s+", p)
            else:
                sents = [p]
            for s in sents:
                if buf and len(buf) + len(s) + 1 > max_chars:
                    chunks.append(buf)
                    buf = ""
                buf = f"{buf}\n{s}".strip() if buf else s
        if buf:
            chunks.append(buf)
        for i, c in enumerate(chunks):
            if len(c) >= min_chars and tamil_ratio(c) >= min_tamil:
                pid = f"{d['doc_id']}#p{i:04d}"
                out.append({"passage_id": pid, "doc_id": d["doc_id"], "title": d["title"], "passage": c[:max_chars]})
    return out


def sample_passages(passages, n, max_per_doc, seed):
    rng = random.Random(seed)
    by_doc = collections.defaultdict(list)
    for p in passages:
        by_doc[p["doc_id"]].append(p)
    for v in by_doc.values():
        rng.shuffle(v)
    # round-robin across books so no single book dominates
    picked, docs = [], sorted(by_doc)
    rng.shuffle(docs)
    level = 0
    while len(picked) < n and level < max_per_doc:
        added = False
        for d in docs:
            if level < len(by_doc[d]) and len(picked) < n:
                picked.append(by_doc[d][level])
                added = True
        if not added:
            break
        level += 1
    return picked


# --------------------------------------------------------------------------- teacher
class Teacher:
    def __init__(self, model, temperature=None, mock=False):
        self.model, self.temperature, self.mock = model, temperature, mock
        self.lock = threading.Lock()
        self.usage = collections.Counter()
        if not mock:
            from openai import OpenAI
            self.client = OpenAI()

    def _call(self, system, user, schema, max_tokens):
        if self.mock:
            return None
        import openai
        kw = dict(model=self.model, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                  response_format={"type": "json_schema", "json_schema": schema}, max_completion_tokens=max_tokens)
        if self.temperature is not None:
            kw["temperature"] = self.temperature
        for attempt in range(6):
            try:
                r = self.client.chat.completions.create(**kw)
                with self.lock:
                    self.usage["calls"] += 1
                    self.usage["input_tokens"] += r.usage.prompt_tokens
                    self.usage["output_tokens"] += r.usage.completion_tokens
                msg = r.choices[0].message
                if getattr(msg, "refusal", None):
                    return {"_refusal": msg.refusal}
                return json.loads(msg.content)
            except (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError) as e:
                wait = min(60, 2 ** attempt * 2)
                print(f"  retry in {wait}s ({type(e).__name__})", flush=True)
                time.sleep(wait)
            except json.JSONDecodeError:
                return {"_error": "bad_json"}
        return {"_error": "gave_up"}

    def generate(self, p, k):
        if self.mock:  # offline: build items from passage sentences, incl. some bad ones on purpose
            sents = [s for s in re.split(r"(?<=[.!?])\s+", p["passage"]) if len(s) > 30][:k]
            items = []
            for i, s in enumerate(sents):
                q = f"{p['title']} பற்றிய கேள்வி {i}: {s[:40]} என்ன?" if i != 1 else f"இந்தப் பகுதியில் {s[:30]}?"
                ev = s if i != 2 else "இது பகுதியில் இல்லாத ஒரு வாக்கியம்."
                items.append({"task_type": TASK_TYPES[i % 5], "question": q, "answer": s, "evidence": ev})
            h = int(hashlib.sha1(p["passage_id"].encode()).hexdigest(), 16)
            quality = QUALITY[h % 3]
            return {"passage_quality": quality, "topic": TOPICS[h % len(TOPICS)], "time_sensitive": h % 4 == 0,
                    "items": [] if quality == "unusable" else items}
        return self._call(GEN_SYSTEM, GEN_USER.format(title=p["title"], passage=p["passage"], k=k), GEN_SCHEMA, 4000)

    def verify(self, passage, question):
        if self.mock:
            return {"answerable": True, "question_clear": "இந்தப்" not in question, "answer": passage[:200]}
        return self._call(VERIFY_SYSTEM, VERIFY_USER.format(passage=passage, question=question), VERIFY_SCHEMA, 1500)


# --------------------------------------------------------------------------- commands
def cmd_generate(a):
    os.makedirs(a.out, exist_ok=True)
    raw_path = os.path.join(a.out, "raw_generations.jsonl")
    done = set()
    if os.path.exists(raw_path):
        with open(raw_path, encoding="utf-8") as f:
            done = {json.loads(l)["passage_id"] for l in f if l.strip()}
    if a.mock:
        a.model = "mock"
    elif not a.model:
        raise SystemExit("No model set: add OPENAI_MODEL=... to .env or pass --model")

    print("Loading CPT train documents ...")
    docs = load_train_docs(a.dataset, a.local_docs)
    passages = make_passages(docs, a.min_chars, a.max_chars)
    picked = sample_passages(passages, a.n_passages, a.max_per_doc, a.seed)
    todo = [p for p in picked if p["passage_id"] not in done]
    print(f"  {len(docs)} train docs -> {len(passages)} usable passages; target {len(picked)}, "
          f"already done {len(picked) - len(todo)}, to generate {len(todo)}")

    teacher = Teacher(a.model, a.temperature, a.mock)
    lock = threading.Lock()

    def work(p):
        res = teacher.generate(p, a.k)
        items = (res or {}).get("items", []) if isinstance(res, dict) else []
        if a.verify and items:
            for it in items:
                v = teacher.verify(p["passage"], it["question"]) or {}
                it["verify_answerable"] = v.get("answerable")
                it["verify_question_clear"] = v.get("question_clear")
                it["verify_answer"] = v.get("answer", "")
        meta = res if isinstance(res, dict) else {}
        rec = {**p, "passage_quality": meta.get("passage_quality"), "topic": meta.get("topic"),
               "time_sensitive": meta.get("time_sensitive"),
               "items": items, "error": (res or {}).get("_error") or (res or {}).get("_refusal"),
               "teacher": a.model, "prompt_version": PROMPT_VERSION, "k": a.k, "verify": a.verify,
               "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        with lock, open(raw_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return len(items)

    t0, n_items = time.time(), 0
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, n in enumerate(ex.map(work, todo), 1):
            n_items += n
            if i % 10 == 0 or i == len(todo):
                u = teacher.usage
                print(f"  {i}/{len(todo)} passages, {n_items} raw items, {u['calls']} calls, "
                      f"{u['input_tokens']:,} in / {u['output_tokens']:,} out tokens [{time.time() - t0:.0f}s]", flush=True)

    u = teacher.usage
    log = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "model": a.model,
           "prompt_version": PROMPT_VERSION, "passages": len(todo), "raw_items": n_items, **u}
    if a.price_in is not None and a.price_out is not None:
        log["est_cost_usd"] = round(u["input_tokens"] / 1e6 * a.price_in + u["output_tokens"] / 1e6 * a.price_out, 4)
    with open(os.path.join(a.out, "generation_log.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(log) + "\n")
    with open(os.path.join(a.out, f"prompt_{PROMPT_VERSION}.txt"), "w", encoding="utf-8") as f:
        f.write("=== GENERATION SYSTEM ===\n" + GEN_SYSTEM + "\n\n=== GENERATION USER ===\n" + GEN_USER +
                "\n\n=== VERIFY SYSTEM ===\n" + VERIFY_SYSTEM + "\n\n=== VERIFY USER ===\n" + VERIFY_USER + "\n")
    print(json.dumps(log, indent=2))


def filter_item(it, passage, a):
    """Return the list of reasons the item fails (empty = keep)."""
    q, ans, ev = it.get("question", "").strip(), it.get("answer", "").strip(), it.get("evidence", "").strip()
    bad = []
    if not q or not ans:
        return ["empty"]
    if len(q) < 10 or len(q) > 300:
        bad.append("q_length")
    if len(ans) < 5 or len(ans) > 1500:
        bad.append("a_length")
    if tamil_ratio(q) < a.min_tamil or tamil_ratio(ans) < a.min_tamil:
        bad.append("not_tamil")
    if PASSAGE_REF.search(q):
        bad.append("refers_to_passage")
    es = evidence_score(ev, passage)
    it["evidence_score"] = round(es, 3)
    if es < a.min_evidence:
        bad.append("evidence_not_in_passage")
    if it.get("verify_answerable") is False:
        bad.append("not_answerable")
    if it.get("verify_question_clear") is False:
        bad.append("unclear_question")
    if it.get("verify_answer"):
        it["verify_similarity"] = round(jaccard(char_ngrams(ans), char_ngrams(it["verify_answer"])), 3)
    return bad


def cmd_build(a):
    raw_path = os.path.join(a.out, "raw_generations.jsonl")
    with open(raw_path, encoding="utf-8") as f:
        recs = [json.loads(l) for l in f if l.strip()]
    # keep only the latest record per passage
    recs = list({r["passage_id"]: r for r in recs}.values())

    stats = collections.Counter()
    quality, topics = collections.Counter(), collections.Counter()
    exclude = {t.strip() for t in (a.exclude_topics or "").split(",") if t.strip()}
    kept, seen_q = [], []
    for r in recs:
        stats["passages"] += 1
        quality[r.get("passage_quality") or "not_rated"] += 1
        topics[r.get("topic") or "not_rated"] += 1
        if r.get("error"):
            stats["passage_errors"] += 1
        for it in r["items"]:
            stats["raw_items"] += 1
            reasons = filter_item(it, r["passage"], a)
            if r.get("passage_quality") == "unusable":
                reasons.append("passage_unusable")
            if r.get("topic") in exclude:
                reasons.append("excluded_topic")
            if a.drop_time_sensitive and r.get("time_sensitive"):
                reasons.append("time_sensitive")
            qg = char_ngrams(it.get("question", ""))
            if not reasons and any(jaccard(qg, s) >= a.dedup for s in seen_q):
                reasons = ["near_duplicate"]
            if reasons:
                for x in reasons:
                    stats[f"drop_{x}"] += 1
                continue
            seen_q.append(qg)
            h = hashlib.sha1((r["passage_id"] + it["question"]).encode()).hexdigest()[:12]
            kept.append({**it, "id": f"slt-{h}", "doc_id": r["doc_id"], "title": r["title"],
                         "passage_id": r["passage_id"], "passage": r["passage"], "teacher": r["teacher"],
                         "passage_quality": r.get("passage_quality"), "topic": r.get("topic"),
                         "time_sensitive": r.get("time_sensitive"),
                         "prompt_version": r["prompt_version"], "created_at": r["created_at"]})
    stats["kept_items"] = len(kept)
    stats["by_task_type"] = dict(collections.Counter(k["task_type"] for k in kept))
    stats["passages_by_quality"] = dict(quality)
    stats["passages_by_topic"] = dict(topics)
    stats["kept_by_topic"] = dict(collections.Counter(k["topic"] or "not_rated" for k in kept))
    stats["kept_time_sensitive"] = sum(1 for k in kept if k["time_sensitive"])

    # SFT train/validation split BY DOCUMENT (validation = early-stopping set)
    docs = sorted({k["doc_id"] for k in kept})
    rng = random.Random(a.seed)
    rng.shuffle(docs)
    n_val = max(1, round(len(docs) * a.val_ratio)) if len(docs) > 1 else 0
    val_docs = set(docs[:n_val])

    def to_row(k, fmt):
        user = k["question"] if fmt == "closed_book" else OPEN_BOOK_TEMPLATE.format(passage=k["passage"], question=k["question"])
        return {"id": f"{k['id']}-{fmt[0]}", "messages": [{"role": "system", "content": SYSTEM_TA},
                                                          {"role": "user", "content": user},
                                                          {"role": "assistant", "content": k["answer"]}],
                "format": fmt, "task_type": k["task_type"], "question": k["question"], "answer": k["answer"],
                "evidence": k["evidence"], "evidence_score": k["evidence_score"], "doc_id": k["doc_id"],
                "title": k["title"], "passage_id": k["passage_id"], "passage": k["passage"],
                "passage_quality": k["passage_quality"], "topic": k["topic"], "time_sensitive": k["time_sensitive"],
                "source": "noolaham_synthetic", "teacher": k["teacher"], "prompt_version": k["prompt_version"],
                "created_at": k["created_at"]}

    splits = {"train": [], "validation": []}
    for k in kept:
        s = "validation" if k["doc_id"] in val_docs else "train"
        for fmt in ("closed_book", "open_book"):
            splits[s].append(to_row(k, fmt))
    for s, rows in splits.items():
        with open(os.path.join(a.out, f"sft_{s}.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats["rows_train"], stats["rows_validation"] = len(splits["train"]), len(splits["validation"])
    stats["docs_train"], stats["docs_validation"] = len(docs) - n_val, n_val

    # review sheet for the human check (random sample)
    sample = rng.sample(kept, min(a.review_n, len(kept)))
    with open(os.path.join(a.out, "review_sample.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "task_type", "topic", "passage_quality", "time_sensitive", "question", "answer", "evidence",
                    "passage", "correct (1-3)", "fluent (1-3)", "sri_lankan_natural (1-3)", "keep? (y/n)", "comment"])
        for k in sample:
            w.writerow([k["id"], k["task_type"], k["topic"], k["passage_quality"], k["time_sensitive"], k["question"],
                        k["answer"], k["evidence"], k["passage"], "", "", "", "", ""])
    with open(os.path.join(a.out, "build_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    keep_rate = 100 * stats["kept_items"] / max(stats["raw_items"], 1)
    print(f"\nKept {stats['kept_items']} of {stats['raw_items']} items ({keep_rate:.0f}%). "
          f"Review {os.path.join(a.out, 'review_sample.csv')} before scaling up.")


def cmd_push(a):
    from datasets import Dataset, DatasetDict
    from huggingface_hub import DatasetCard
    import yaml
    splits = {}
    for s in ("train", "validation"):
        with open(os.path.join(a.out, f"sft_{s}.jsonl"), encoding="utf-8") as f:
            splits[s] = Dataset.from_list([json.loads(l) for l in f if l.strip()])
    stats = json.load(open(os.path.join(a.out, "build_stats.json"), encoding="utf-8"))
    logs = [json.loads(l) for l in open(os.path.join(a.out, "generation_log.jsonl"), encoding="utf-8") if l.strip()]
    teachers = sorted({l["model"] for l in logs})
    pversions = ", ".join(sorted({l.get("prompt_version", "?") for l in logs}))
    DatasetDict(splits).push_to_hub(a.repo, private=True, token=os.environ.get("HF_TOKEN"))
    meta = {"language": ["ta"], "task_categories": ["question-answering", "text-generation"],
            "tags": ["tamil", "sri-lankan-tamil", "instruction-tuning", "sft", "synthetic", "noolaham", "sltfm"],
            "configs": [{"config_name": "default", "data_files": [
                {"split": "train", "path": "data/train-*"}, {"split": "validation", "path": "data/validation-*"}]}]}
    drops = "\n".join(f"| {k[5:]} | {v:,} |" for k, v in stats.items() if k.startswith("drop_"))
    tt = "\n".join(f"| {k} | {v:,} |" for k, v in stats.get("by_task_type", {}).items())
    body = f"""
# Sri Lankan Tamil Instruction Data (SLTFM, synthetic, Noolaham-grounded)

Question-answer pairs in Sri Lankan Tamil for supervised fine-tuning (SFT) of the SLTFM model.
Each pair was generated by a teacher model from a passage of a Noolaham book in the **train** split of
[`ctnlpr/sri-lankan-tamil-corpus`](https://huggingface.co/datasets/ctnlpr/sri-lankan-tamil-corpus);
the corpus validation books were never used, so they remain available for testing.

**Status: pilot / work in progress. Not yet human-validated at scale.**

## Format
Each row has `messages` (system / user / assistant) ready for SFT, plus provenance fields.
Every Q&A pair appears in two formats:
- `closed_book`: the user turn is the question only (teaches knowledge + answering).
- `open_book`: the user turn is the passage + question (reading comprehension; later RAG use).

Fields: `id, messages, format, task_type, question, answer, evidence, evidence_score, doc_id, title,
passage_id, passage, passage_quality, topic, time_sensitive, source, teacher, prompt_version, created_at`.

`time_sensitive = true` marks pairs from passages whose facts may be outdated (e.g. old medical claims);
filter them out if the model should not learn them as current facts.

## Splits (by document; validation = early-stopping set)
| | train | validation |
|---|---|---|
| rows | {stats['rows_train']:,} | {stats['rows_validation']:,} |
| source documents | {stats['docs_train']} | {stats['docs_validation']} |

Task types (unique Q&A pairs):

| task_type | pairs |
|---|---|
{tt}

## Generation
- Teacher model(s): {', '.join(teachers)}; prompt version(s) {pversions} (full prompts kept with the generation scripts).
- Passages of 500-1,500 characters from train documents, sampled round-robin across books.
- The teacher must answer only from the passage and quote a supporting sentence (`evidence`).

## Filtering
Kept {stats['kept_items']:,} of {stats['raw_items']:,} generated pairs.

| Filter (pair removed because) | count |
|---|---|
{drops}

Filters: Tamil-script share, length limits, no references to "the passage" in questions, evidence must
match the passage, optional answerability re-check by a second call, near-duplicate questions removed.

## Limitations
- Synthetic: may contain errors or Indian-Tamil phrasing; human review of a sample is in progress.
- Source books contain OCR noise.
- Use of teacher-model outputs is subject to the teacher provider's terms; redistribution of Noolaham-derived
  text must be confirmed with the Noolaham Foundation before this dataset is made public.

Maintained by CTNLPR / ExentAI.
"""
    card = DatasetCard("---\n" + yaml.safe_dump(meta, sort_keys=False, allow_unicode=True) + "---\n" + body)
    card.push_to_hub(a.repo, repo_type="dataset", token=os.environ.get("HF_TOKEN"))
    print(f"Pushed: https://huggingface.co/datasets/{a.repo}")


def load_env():
    """Read OPENAI_API_KEY / HF_TOKEN from a .env file next to the script (or the current folder)."""
    import sys
    for stream in (sys.stdout, sys.stderr):  # Tamil output on Windows consoles without PYTHONUTF8
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    here = os.path.dirname(os.path.abspath(__file__))
    for path in (os.path.join(here, ".env"), os.path.join(os.getcwd(), ".env")):
        if os.path.exists(path):
            load_dotenv(path, override=False)


def env_float(name):
    """Number from .env, or None if missing or not a number (e.g. a '...' placeholder)."""
    v = os.environ.get(name, "").strip()
    if not v:
        return None
    try:
        return float(v)
    except ValueError:
        print(f"Warning: {name}={v!r} in .env is not a number - ignored (no cost estimate).")
        return None


def main():
    load_env()
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--model", default=os.environ.get("OPENAI_MODEL"), help="or set OPENAI_MODEL in .env")
    g.add_argument("--n-passages", type=int, default=30)
    g.add_argument("--k", type=int, default=4, help="Q&A items requested per passage")
    g.add_argument("--max-per-doc", type=int, default=40)
    g.add_argument("--min-chars", type=int, default=500)
    g.add_argument("--max-chars", type=int, default=1500)
    g.add_argument("--temperature", type=float, default=None, help="omit for models that do not accept it")
    g.add_argument("--no-verify", dest="verify", action="store_false", help="skip the answerability re-check")
    g.add_argument("--workers", type=int, default=4)
    g.add_argument("--price-in", type=float, default=env_float("PRICE_IN_PER_1M"), help="USD per 1M input tokens (or PRICE_IN_PER_1M in .env)")
    g.add_argument("--price-out", type=float, default=env_float("PRICE_OUT_PER_1M"), help="USD per 1M output tokens (or PRICE_OUT_PER_1M in .env)")
    g.add_argument("--mock", action="store_true", help="offline test without the API")
    g.add_argument("--local-docs", help="jsonl of {doc_id,title,text} instead of the HF dataset")
    b = sub.add_parser("build")
    b.add_argument("--min-tamil", type=float, default=0.8)
    b.add_argument("--min-evidence", type=float, default=0.85)
    b.add_argument("--dedup", type=float, default=0.8, help="char-3gram Jaccard threshold for near-duplicate questions")
    b.add_argument("--val-ratio", type=float, default=0.1)
    b.add_argument("--review-n", type=int, default=60)
    b.add_argument("--exclude-topics", default="", help="comma list, e.g. medicine_health")
    b.add_argument("--drop-time-sensitive", action="store_true", help="drop pairs from passages flagged time_sensitive")
    p = sub.add_parser("push")
    p.add_argument("--repo", default=os.environ.get("HF_REPO", "ctnlpr/sri-lankan-tamil-instruct"))
    for s in (g, b, p):
        s.add_argument("--out", default="sft_out")
        s.add_argument("--seed", type=int, default=42)
        s.add_argument("--dataset", default="ctnlpr/sri-lankan-tamil-corpus")
    a = ap.parse_args()
    {"generate": cmd_generate, "build": cmd_build, "push": cmd_push}[a.cmd](a)


if __name__ == "__main__":
    main()
