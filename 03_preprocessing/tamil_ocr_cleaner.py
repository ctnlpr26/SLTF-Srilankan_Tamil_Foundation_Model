"""
Tamil Text Cleaning Pipeline
==================================
Production pipeline for large-scale Tamil OCR cleaning (KG building).

Active fixes:
  Fix  0–16 : Structural cleanup, garbage lines, citations, unicode, punct
  Fix 17    : Protect honorific abbreviations (திரு., பண்டிதர்., etc.)
  Fix 17b   : Protect 1–2 char Tamil initials mid-text (வ., ச., கோ., மு.)
  Fix 18    : Merge initial line fragments after sentence splitting
  Fix 19    : OCR pulli-split correction (corpus frequency gated)
  Fix 21    : Mid-word ஃ (aytham) OCR correction
  Fix 22    : Intra-word OCR space correction (corpus frequency gated)
  Fix 23b   : Unicode combining char fix (ு + ே → ோ)
  Fix 28    : Historical pre-reform Tamil ligature OCR correction
              (master dict gated — requires ta_master_dict.pkl)
  Fix 29    : Grammar-based participial compound merging
              (no corpus dependency — grammar rules only)
              e.g. இருந்த வர்கள் → இருந்தவர்கள்
                   வெளியிடு வதற்கு → வெளியிடுவதற்கு
  IndicNLP  : Sentence tokenization

Fix 20 (fused-word segmentation via WordNet + Wiktionary) was removed:
  - Very low recall on Tamil proper nouns / place names
  - KG building benefits from intact named entities
  - High startup cost (loading 270K+ external lexicon entries)
"""

import argparse
import os
import pickle
import re
import sys
import unicodedata
from pathlib import Path
from typing import List, Tuple

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Tamil vowel signs that cannot stand alone ──────────────────────────────
VOWEL_SIGNS = "\u0BBE\u0BBF\u0BC0\u0BC1\u0BC2\u0BC6\u0BC7\u0BC8\u0BCA\u0BCB\u0BCC\u0BCD\u0BD7"

# ── Fix 17: Abbreviation placeholder ──────────────────────────────────────
_PLACEHOLDER = "§"

TAMIL_HONORIFICS: List[str] = [
    "திருமதி",
    "டாக்டர்",
    "பேராசிரியர்",
    "பண்டிதர்",
    "ஸ்ரீல்ஸ்ரீ",
    "ஸ்ரீமதி",
    "செல்வி",
    "திரு",
    "ஸ்ரீ",
    "பண்டிதர்திரு",
    "இல",          # இல. = இலக்கம் (number) — added in v5
]

# ── Fix 19: Pulli-split correction ─────────────────────────────────────────
_PULLI_SPLIT_RE = re.compile(
    r"(?<![,.\u0B80-\u0BFF])([\u0B80-\u0BFF]{2,5}்)\s+([\u0B80-\u0BFF]{1,3})(?![\u0B80-\u0BFF])"
)
_BLOCKLIST_RIGHT = {
    "என", "என்", "ஒரு", "பல", "சில", "கி", "சம்", "சிவ",
    "இரு", "உடை", "சடை", "சமய", "வாழ",
}

# ── Indic NLP: lazy-loaded cache (avoid re-reading resources every page) ──
_INDIC_NLP_CACHE: dict = {}   # keys: "normalizer", "sentence_tokenize"

def _get_indic_nlp():
    if not _INDIC_NLP_CACHE:
        from indicnlp import common, loader
        from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
        from indicnlp.tokenize import sentence_tokenize as _st
        # Set INDIC_RESOURCES_PATH, or place indic_nlp_resources/ next to this file
        RESOURCES = os.environ.get(
            "INDIC_RESOURCES_PATH",
            str(Path(__file__).resolve().parent / "indic_nlp_resources"),
        )
        common.set_resources_path(RESOURCES)
        loader.load()
        _INDIC_NLP_CACHE["normalizer"] = IndicNormalizerFactory().get_normalizer("ta")
        _INDIC_NLP_CACHE["sentence_tokenize"] = _st
    return _INDIC_NLP_CACHE["normalizer"], _INDIC_NLP_CACHE["sentence_tokenize"]


# ── Fix 22: Intra-word space correction ────────────────────────────────────

# ── Fix 21: Mid-word aytham OCR errors ─────────────────────────────────────
_AYTHAM_MID_PATTERNS = [
    (re.compile(r"([\u0B95-\u0BB9])ஃ\1யா"), r"\1்\1ையா"),
    (re.compile(r"துஃண"), "துண"),
]

# ── Fix 28: Historical pre-reform Tamil ligature confusion matrix ───────────
_OLD_FONT_CONFUSIONS = [
    # --- AI (ை) Series (OCR misreads left loop as Kombu ே) ---
    ("ணே", "ணை"),
    ("னே", "னை"),
    ("ளே", "ளை"),
    ("லே", "லை"),

    # --- AA (ா) Series (OCR misreads downward loop as ஞ or ளு) ---
    ("ணஞ", "ணா"), ("ணளு", "ணா"),
    ("னஞ", "னா"), ("னளு", "னா"),
    ("றஞ", "றா"), ("றளு", "றா"),

    # --- O (ொ) Series (OCR misreads as ெ + base + ஞ/ளு) ---
    ("ெணஞ", "ணொ"), ("ெணளு", "ணொ"),
    ("ெனஞ", "னொ"), ("ெனளு", "னொ"),
    ("ெறஞ", "றொ"), ("ெறளு", "றொ"),

    # --- OO (ோ) Series (OCR misreads as ே + base + ஞ/ளு) ---
    ("ேணஞ", "ணோ"), ("ேணளு", "ணோ"),
    ("ேனஞ", "னோ"), ("ேனளு", "னோ"),
    ("ேறஞ", "றோ"), ("ேறளு", "றோ"),

    # --- EE→OO (ோ) Series (OCR misreads ோ as ே — Kombu vs Kombu+u) ---
    # e.g. அறிகின்றோம் → அறிகின்றேம், கொள்ளுகின்றோம் → கொள்ளுகின்றேம்
    ("ணேம்", "ணோம்"),
    ("னேம்", "னோம்"),
    ("ளேம்", "ளோம்"),
    ("லேம்", "லோம்"),
    ("றேம்", "றோம்"),

    # --- Other Known Visual Confusions ---
    ("ஞா", "னா"),
]
# Longest pattern first: ensures OO/O series tried before AA series
_OLD_FONT_CONFUSIONS = sorted(_OLD_FONT_CONFUSIONS, key=lambda x: len(x[0]), reverse=True)

# Fix 28 guard: these patterns appear word-finally as the Tamil emphatic particle ஏ
# (e.g. அவர்களே = "they themselves", புலவர்களே = "the scholars themselves").
# If bad_str ends the word AND dropping it + adding pulli gives a dict word,
# the ே is emphatic — NOT an OCR ை confusion — so skip the substitution.
# All four are guarded: on modern-script books the emphatic reading
# (உடனே, தானே, அவனே) is far more common than an OCR ை error, so a
# rare missed அதனே → அதனை is cheaper than corrupting -உடனே everywhere.
_EMPHATIC_FINALS = frozenset(["ளே", "னே", "லே", "ணே"])

# Fix 28 guard: a fused-compound window shorter than this is too generic to
# trust (டனை, னைக, லையு match almost anything in a 2M-word dict).
_MIN_LIGATURE_WINDOW = 6

# Two combining marks in a row (e.g. ா் from வண்ணஞ் → வண்ணா்) is never valid.
_DOUBLE_SIGN_RE = re.compile(r"[\u0BBE-\u0BCD]{2}")

# Fix 28 guard: sequences where bad_str is part of a genuine Tamil morpheme,
# NOT an OCR confusion.  If every occurrence of bad_str in the token falls
# inside one of these longer sequences, skip the substitution entirely.
# e.g. முன்னேற்றம் contains னேற் — the ஏற் morpheme (rise/advance), not ை.
# மேற்கு, மேற்படி, ஏற்படும் are similarly protected.
_PROTECTED_SEQUENCES = frozenset([
    "னேற்",   # ஏற்றம் / முன்னேற்றம் / ஏற்படு
    "லேற்",   # மேற்கு / மேற்படி
])

# ── Fix 29: Participial compound suffix list ────────────────────────────────
# Right-side tokens that are Tamil verbal/participial suffixes.
# Sorted longest-first so startswith matching picks the most specific suffix.
# e.g. "வதிலிருந்து" is tried before "வதில்" to avoid a short match swallowing
# a longer valid suffix.
_PARTICIPIAL_SUFFIXES: List[str] = sorted([
    "வர்கள்",       # plural relative participle: those who
    "வர்",          # singular relative participle: the one who
    "வன்",          # masculine relative participle: the man who
    "வள்",          # feminine relative participle: the woman who
    "வது",          # neuter verbal noun: the act of
    "வதை",          # acc. of வது
    "வதில்",        # loc. of வது: in the act of
    "வதற்கு",       # dat. of வது: for the purpose of
    "வதாக",         # as-if form of வது
    "வதும்",        # additive of வது: also does
    "வதற்காக",      # for the sake of doing
    "வதிலிருந்து",  # ablative: from doing
    "வதனால்",       # causal: because of doing
    "வதோடு",        # associative: along with doing
    "வதுண்டு",      # existential: there is the act of
    "வதில்லை",      # negative: there is no act of
], key=len, reverse=True)


# ══════════════════════════════════════════════════════════════════════════════
# Fix 0–16  (unchanged from v4)
# ══════════════════════════════════════════════════════════════════════════════

def strip_structural_markup(text: str, verbose: bool = True) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    removed_sep = removed_page = removed_latex = removed_bullet = removed_img = stripped_tags = 0
    out_lines = []
    for line in text.split("\n"):
        if re.fullmatch(r"=+\s*", line):
            removed_sep += 1; continue
        if re.fullmatch(r"PAGE\s+\d+\s*", line):
            removed_page += 1; continue
        new = re.sub(r"^\s*\[(?:paragraph|heading|image_ref)\]\s*", "", line)
        if new != line:
            stripped_tags += 1
            if re.match(r"^\s*\[heading\]", line):
                new = new + "_HEADING_"
            line = new
        new = re.sub(r"\s*\[(?:paragraph|heading|image_ref)\]\s*", " ", line)
        if new != line: stripped_tags += 1; line = new
        if "\\" in line:
            new = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", "", line)
            if new != line: removed_latex += 1; line = new
        if re.match(r"^\s*[·•]", line):
            line = re.sub(r"^\s*[·•]\s*", "", line); removed_bullet += 1
        if "page_" in line and "_image_" in line:
            new = re.sub(r"page_\d+_image_\d+\.\w+", "", line)
            if new != line: removed_img += 1; line = new
        line = re.sub(r"  +", " ", line).strip()
        out_lines.append(line)
    text = "\n".join(out_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    if verbose:
        print(f"[Fix 0] Structural markup — tags: {stripped_tags}, sep lines: {removed_sep}, "
              f"PAGE headers: {removed_page}, LaTeX: {removed_latex}, "
              f"bullets: {removed_bullet}, img refs: {removed_img}")
    return text


def fix_lone_vowels(text: str, verbose: bool = True) -> str:
    before = len(re.findall(r"(?<=\s)[" + VOWEL_SIGNS + r"]", text))
    text = re.sub(r"(\s)[" + VOWEL_SIGNS + r"]+", r"\1", text)
    after = len(re.findall(r"(?<=\s)[" + VOWEL_SIGNS + r"]", text))
    if verbose: print(f"[Fix 1] Lone vowel signs  — before: {before}, after: {after}")
    return text


def fix_aytham(text: str, verbose: bool = True) -> str:
    before = len(re.findall(r"([\u0B80-\u0BFF])ஃ(\s)", text))
    text = re.sub(r"([\u0B80-\u0BFF])ஃ(\s)", r"\1.\2", text)
    after = len(re.findall(r"([\u0B80-\u0BFF])ஃ(\s)", text))
    if verbose: print(f"[Fix 2] Aytham as period  — before: {before}, after: {after}")
    return text


def fix_garbage_lines(text: str, min_tamil_ratio: float = 0.5, verbose: bool = True) -> str:
    def is_garbage(line):
        stripped = line.strip()
        if not stripped: return False
        if "_HEADING_" in stripped:
            heading_text = stripped.replace("_HEADING_", "").strip()
            return not bool(re.search(r"[\u0B80-\u0BFF]", heading_text))
        if not re.search(r"[\u0B80-\u0BFF]", stripped) and re.search(r"[a-zA-Z]", stripped):
            return True
        if len(stripped) < 10:
            tamil_chars = re.findall(r"[\u0B80-\u0BFF]", stripped)
            if not tamil_chars: return True
            if len(tamil_chars) <= 2 and re.search(r"[^\u0B80-\u0BFF\s]", stripped): return True
            return False
        tamil_chars = len(re.findall(r"[\u0B80-\u0BFF]", stripped))
        total_chars = len(re.sub(r"\s", "", stripped))
        if total_chars == 0: return False
        if tamil_chars / total_chars < min_tamil_ratio:
            # Keep Tamil lines diluted by English glosses or numbers, e.g.
            # "உள்ளம்சார் உடல்நோய் (Psychosomatic Disease)" — useful for KG.
            tamil_words = re.findall(r"[\u0B80-\u0BFF]{3,}", stripped)
            if not (len(tamil_words) >= 2 and tamil_chars >= 12): return True
        words = stripped.split()
        if len(words) > 5:
            if sum(1 for w in words if len(w) == 1) / len(words) > 0.3: return True
            if sum(1 for w in words if len(w) <= 2) / len(words) > 0.4: return True
        return False
    lines = text.split("\n")
    cleaned = [l for l in lines if not is_garbage(l)]
    if verbose: print(f"[Fix 3] Garbage lines     — removed: {len(lines) - len(cleaned)}")
    return "\n".join(cleaned)


def fix_duplicates(text: str, threshold: float = 0.7, verbose: bool = True) -> str:
    def ngrams(t, n=3): return set(t[i:i+n] for i in range(len(t) - n + 1))
    def jaccard(t1, t2, n=3):
        a, b = ngrams(t1, n), ngrams(t2, n)
        if not a and not b: return 1.0
        if not a or not b: return 0.0
        return len(a & b) / len(a | b)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    joined = []; i = 0; merged_count = 0
    while i < len(paragraphs):
        current = paragraphs[i]
        if i + 1 < len(paragraphs) and jaccard(current, paragraphs[i + 1]) > threshold:
            joined.append(current + " " + paragraphs[i + 1]); merged_count += 1; i += 2; continue
        joined.append(current); i += 1
    if verbose: print(f"[Fix 4] Broken paragraphs — merged: {merged_count}")
    return "\n\n".join(joined)


def fix_fused_punct(text: str, verbose: bool = True) -> str:
    pattern = r"([\u0B80-\u0BFF])-(?=[\u0B80-\u0BFF])"
    before = len(re.findall(pattern, text))
    text = re.sub(pattern, r"\1 - ", text)
    after = len(re.findall(pattern, text))
    if verbose: print(f"[Fix 5] Fused hyphens     — before: {before}, after: {after}")
    return text


def fix_citations(text: str, verbose: bool = True) -> str:
    before_yp = len(re.findall(r"\(\s*\d{4}\s*:\s*\d+\s*\)", text))
    text = re.sub(r"\(\s*\d{4}\s*:\s*\d+\s*\)", "", text)
    before_yo = len(re.findall(r"(?<=[\u0B80-\u0BFF\s])\(\s*\d{4}\s*\)(?=[\s\u0B80-\u0BFF])", text))
    text = re.sub(r"(?<=[\u0B80-\u0BFF\s])\(\s*\d{4}\s*\)(?=[\s\u0B80-\u0BFF])", "", text)
    if verbose: print(f"[Fix 6] Citations         — year:page: {before_yp}, year-only: {before_yo}")
    return text


def fix_whitespace(text: str, verbose: bool = True) -> str:
    before_ms = len(re.findall(r" {2,}", text))
    before_mb = len(re.findall(r"\n{3,}", text))
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r" +\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    if verbose: print(f"[Fix 7] Whitespace        — multi-spaces: {before_ms}, multi-blanks: {before_mb}")
    return text


def fix_more_citations(text: str, verbose: bool = True) -> str:
    before = len(text)
    text = re.sub(r"\(\.\s*\d{4}[\s:,\-\d]*\)", "", text)
    text = re.sub(r"\(மேலது\)", "", text)
    text = re.sub(r"\(மேலது[ஃ\d:]+\)", "", text)
    text = re.sub(r"\([;,.]\s*", "(", text)
    text = re.sub(r"  +", " ", text)
    text = re.sub(r" ([,.\)])", r"\1", text)
    if verbose: print(f"[Fix 8] Citation artefacts— chars removed: {before - len(text)}")
    return text


def audit_after_fix8(text: str) -> None:
    p1 = re.findall(r"[\u0B80-\u0BFF]ை(?<![யலனந]ை)", text)
    p2 = re.findall(r"([க-ன])\1(?!்)", text)
    p3 = re.findall(r"[\u0B80-\u0BFF];", text)
    print(f"[Audit] ை without ய/ல/ன/ந : {len(p1)} → {p1[:10]}")
    print(f"[Audit] Double consonant   : {len(p2)} → {p2[:10]}")
    print(f"[Audit] Semicolon in Tamil : {len(p3)} → {p3[:10]}")


def fix_semicolons(text: str, verbose: bool = True) -> str:
    before = len(re.findall(r"[\u0B80-\u0BFF];", text))
    text = re.sub(r"([\u0B80-\u0BFF]);", r"\1", text)
    if verbose: print(f"[Fix 9] Stray semicolons  — removed: {before}")
    return text


def fix_stray_punct(text: str, verbose: bool = True) -> str:
    before = len(text)
    text = re.sub(r"([\u0B80-\u0BFF\-\d]+)\s*:\s*\.", r"\1", text)
    text = re.sub(r",{2,}", ",", text)
    text = re.sub(r"\s+([,.;:?!])", r"\1", text)
    text = re.sub(r"([,.;:?!])(?:\s*[,.;:?!]){1,}", r"\1", text)
    text = re.sub(r"  +", " ", text)
    if verbose: print(f"[Fix 10] Stray punct      — chars removed: {before - len(text)}")
    return text


def normalize_unicode(text: str, verbose: bool = True) -> str:
    chars_before = len(text); tokens_before = len(re.findall(r"[\u0B80-\u0BFF]+", text))
    text = unicodedata.normalize("NFC", text)
    # Fix 23b: some OCR engines emit ு (U+0BC1) + ே (U+0BC7) as two codepoints
    # instead of the precomposed ோ (U+0BCB). NFC does not collapse this pair.
    combining_fixed = text.count("\u0BC1\u0BC7")
    if combining_fixed:
        text = text.replace("\u0BC1\u0BC7", "\u0BCB")
    chars_after = len(text); tokens_after = len(re.findall(r"[\u0B80-\u0BFF]+", text))
    if verbose:
        print(f"[Fix 11] NFC normalize    — chars: {chars_before:,}→{chars_after:,}, "
              f"tokens: {tokens_before:,}→{tokens_after:,}"
              + (f", ுே→ோ: {combining_fixed}" if combining_fixed else ""))
    return text


def fix_empty_brackets(text: str, verbose: bool = True) -> str:
    count_round = text.count("()")
    count_square = len(re.findall(r"\[\s*\]", text))
    count_curly = len(re.findall(r"\{\s*\}", text))
    count_solo = len(re.findall(r"^\s*\)", text, re.MULTILINE))
    text = re.sub(r"\s*\(\)\s*", " ", text)
    text = re.sub(r"\[\s*\]", "", text)
    text = re.sub(r"\{\s*\}", "", text)
    text = re.sub(r"  +", " ", text)
    if verbose:
        print(f"[Fix 12] Empty brackets   — (): {count_round}, []: {count_square}, "
              + "{}:" + f" {count_curly}, solo ): {count_solo}")
    return text


def fix_bracket_artifacts(text: str, verbose: bool = True) -> str:
    before = len(text)
    count_dash = len(re.findall(r"\(\s*-\s*\)", text))
    count_dot = len(re.findall(r"\(\s*\.\s*\)", text))
    count_solo = len(re.findall(r"^\s*\)\s+", text, re.MULTILINE))
    text = re.sub(r"\(\s*[-.,;:]*\s*\)", "", text)
    text = re.sub(r"^\s*\)\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\(\s*\.(?=\s)", "", text)
    text = re.sub(r"  +", " ", text)
    if verbose:
        print(f"[Fix 13] Bracket artifacts— ( -): {count_dash}, (.): {count_dot}, "
              f"solo ): {count_solo}, chars removed: {before - len(text)}")
    return text


def fix_number_punct_garbage(text: str, verbose: bool = True) -> str:
    before = len(text)
    text = re.sub(r"\b\d+\.\s*\d+\.\s*[,.\-\s\d]+\d+", "", text)
    text = re.sub(r"^\s*[,.\-\s]+$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s+[,.\-\s]+\s+", " ", text)
    text = re.sub(r"\s+\.\s*\d+\s*[,.\-]*\s*\d*\s*\.?\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s+\.\s*\d+\s*\.\s*\d*\s*\.?\s+", " ", text)
    text = re.sub(r"^\s*(?:\d+\s*[.)-]\s*){2,}(?=\S)", "", text, flags=re.MULTILINE)
    text = re.sub(r"  +", " ", text)
    if verbose: print(f"[Fix 14] Num-punct garbage— chars removed: {before - len(text)}")
    return text


def fix_footnote_numbers(text: str, verbose: bool = True) -> str:
    pattern = r"([\u0B80-\u0BFF]{2,})(\d{1,2})(?=\s|[,.]|\)|$)"
    before = len(re.findall(pattern, text))
    text = re.sub(pattern, r"\1", text)
    if verbose: print(f"[Fix 15] Footnote numbers — removed: {before}")
    return text


def fix_trailing_fragments(text: str, verbose: bool = True) -> str:
    before = len(text)
    text = re.sub(r"\s+[,.]\s*\.$", ".", text, flags=re.MULTILINE)
    text = re.sub(r"\s+[,.]\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\.{2,}$", ".", text, flags=re.MULTILINE)
    text = re.sub(r"([\u0B80-\u0BFF])\s+\.$", r"\1.", text, flags=re.MULTILINE)
    if verbose: print(f"[Fix 16] Trailing fragments— chars removed: {before - len(text)}")
    return text


# ══════════════════════════════════════════════════════════════════════════════
# Fix 17: Protect honorific abbreviations
# ══════════════════════════════════════════════════════════════════════════════

def protect_abbreviations(text: str, verbose: bool = True) -> str:
    count = 0
    for h in TAMIL_HONORIFICS:
        token = h + "."
        if token in text:
            count += text.count(token)
            text = text.replace(token, h + _PLACEHOLDER)
    # Fix 17b: protect 1–2 char Tamil initials that appear mid-text.
    text, n = re.subn(
        r"(\s)([\u0B80-\u0BFF]{1,2})\.(?=\s+[\u0B80-\u0BFF])",
        lambda m: m.group(1) + m.group(2) + _PLACEHOLDER,
        text,
    )
    count += n
    if verbose: print(f"[Fix 17] Protect abbrevs  — placeholders inserted: {count}")
    return text


def restore_abbreviations(text: str, verbose: bool = True) -> str:
    count = text.count(_PLACEHOLDER)
    text = text.replace(_PLACEHOLDER, ".")
    if verbose: print(f"[Fix 17] Restore abbrevs  — placeholders restored: {count}")
    return text


# ══════════════════════════════════════════════════════════════════════════════
# Fix 18: Merge residual initial-fragment lines
# ══════════════════════════════════════════════════════════════════════════════

_INITIAL_FRAG_RE = re.compile(r"^[\u0B80-\u0BFF]{1,3}\.\s")


def _ends_with_abbrev(line: str) -> bool:
    s = line.rstrip()
    if any(s.endswith(h + ".") for h in TAMIL_HONORIFICS):
        return True
    if re.search(r"(?:^|[\s,])[\u0B80-\u0BFF]{1,2}\.$", s):
        return True
    return False


def fix_initial_line_fragments(lines: List[str]) -> Tuple[List[str], int]:
    merged_count = 0
    out: List[str] = []
    i = 0
    while i < len(lines):
        cur = lines[i]
        if i + 1 < len(lines) and _ends_with_abbrev(cur):
            nxt = lines[i + 1].strip()
            if re.match(r"^[\u0B80-\u0BFF0-9\"'(]", nxt):
                out.append((cur.rstrip() + " " + nxt).strip())
                merged_count += 1; i += 2; continue
        out.append(cur); i += 1
    final: List[str] = []
    for line in out:
        if final and _INITIAL_FRAG_RE.match(line.strip()):
            final[-1] = (final[-1].rstrip() + " " + line.strip()).strip()
            merged_count += 1
        else:
            final.append(line)
    return final, merged_count


# ══════════════════════════════════════════════════════════════════════════════
# Fix 19: OCR pulli-split correction
# ══════════════════════════════════════════════════════════════════════════════

def fix_ocr_pulli_splits(text: str, master_dict: set, verbose: bool = True) -> str:
    candidates = _PULLI_SPLIT_RE.findall(text)
    accepted = rejected = 0
    def _decide(m: re.Match) -> str:
        nonlocal accepted, rejected
        left, right = m.group(1), m.group(2)
        if right in _BLOCKLIST_RIGHT: rejected += 1; return m.group(0)
        joined = left + right
        if joined in master_dict: accepted += 1; return joined
        rejected += 1; return m.group(0)
    text = _PULLI_SPLIT_RE.sub(_decide, text)
    if verbose:
        print(f"[Fix 19] Pulli splits     — candidates: {len(candidates)}, "
              f"merged: {accepted}, kept split: {rejected}")
    return text


# ══════════════════════════════════════════════════════════════════════════════
# Fix 21: Mid-word ஃ (aytham) OCR correction
# ══════════════════════════════════════════════════════════════════════════════

def fix_mid_aytham(text: str, verbose: bool = True) -> str:
    before = text.count("ஃ")
    fixed = 0

    def _fix_doubled(m: re.Match) -> str:
        nonlocal fixed
        c = m.group(1)
        fixed += 1
        return c + "்" + c + "ையா"

    text = re.sub(r"([\u0B95-\u0BB9])்ஃ\1யா", _fix_doubled, text)

    count_p2 = text.count("துஃணயா")
    text = text.replace("துஃணயா", "துணையா")
    fixed += count_p2

    after = text.count("ஃ")
    if verbose:
        print(f"[Fix 21] Mid-word ஃ       — ஃ before: {before}, after: {after}, fixed: {fixed}")
    return text


# ══════════════════════════════════════════════════════════════════════════════
# Fix 22: Intra-word OCR space correction
# ══════════════════════════════════════════════════════════════════════════════

_TAMIL_TOKEN_RE = re.compile(r"^[\u0B80-\u0BFF]+$")
_MAX_COMBINED_LEN = 20


def fix_intra_word_spaces(
    text: str,
    master_dict: set,
    verbose: bool = True,
) -> str:
    merged = 0

    def merge_line(line: str) -> str:
        nonlocal merged
        parts = re.split(r"(\s+)", line)
        i = 0
        out: List[str] = []
        while i < len(parts):
            tok = parts[i]
            if (
                _TAMIL_TOKEN_RE.match(tok)
                and i + 2 < len(parts)
                and _TAMIL_TOKEN_RE.match(parts[i + 2])
            ):
                right = parts[i + 2]
                combined = tok + right
                # Only merge when the split is actually broken: if both halves
                # are valid words on their own (தமிழ் மக்கள், ஆம் ஆண்டு), the
                # space is real — the web-crawled dict also holds space-less
                # spellings of many such phrases.
                if (
                    combined in master_dict
                    and len(combined) <= _MAX_COMBINED_LEN
                    and not (tok in master_dict and right in master_dict)
                ):
                    out.append(combined)
                    merged += 1
                    i += 3
                    continue
            out.append(tok)
            i += 1
        return "".join(out)

    lines = text.split("\n")
    new_lines = [merge_line(ln) for ln in lines]
    if verbose:
        print(f"[Fix 22] Intra-word spaces — merged: {merged}")
    return "\n".join(new_lines)


def fix_historical_ligatures(
    text: str,
    master_dict: set,
    verbose: bool = True,
) -> str:
    if not master_dict:
        if verbose: print("[Fix 28] Historical lig.  — no master_dict, skipped")
        return text

    fixed_count = 0
    replacements_made = []

    def _attempt_fix(match: re.Match) -> str:
        nonlocal fixed_count
        original_word = match.group(0)

        if original_word in master_dict:
            return original_word

        # Multi-pass: keep applying corrections until no bad pattern remains.
        # A fused token may contain multiple independent bad patterns
        # (e.g. X-னே-Y-லே-Z).  Each pass fixes one and loops back so the
        # next pattern is tried on the already-corrected candidate.
        current = original_word
        changed = True
        while changed:
            changed = False
            for bad_str, good_str in _OLD_FONT_CONFUSIONS:
                if bad_str not in current:
                    continue
                # Guard: word-final emphatic ஏ  (e.g. அவர்களே, புலவர்களே)
                # If the pattern ends the word and stripping it + adding pulli
                # gives a dict word, the ே is emphatic — not an OCR ை error.
                if (bad_str in _EMPHATIC_FINALS
                        and current.endswith(bad_str)
                        and (current[:-len(bad_str)] + bad_str[0] + "்") in master_dict):
                    continue
                # Guard: bad_str occurs only inside a protected morpheme sequence
                # (e.g. னேற்  in முன்னேற்றம் — genuine ஏற் morpheme, not OCR ை).
                blen = len(bad_str)
                positions = [i for i in range(len(current) - blen + 1)
                             if current[i:i + blen] == bad_str]
                if positions and all(
                    current[i:i + blen + 2] in _PROTECTED_SEQUENCES
                    for i in positions
                ):
                    continue
                candidate = current.replace(bad_str, good_str)
                if _DOUBLE_SIGN_RE.search(candidate):
                    continue
                if candidate in master_dict:
                    if verbose:
                        replacements_made.append(f"{original_word} -> {candidate}")
                    current = candidate
                    changed = True
                    break  # restart loop with updated current
                # Fused-compound fallback: the corrected sub-word may be buried
                # inside a larger fused token.  Scan substrings of the candidate;
                # if a window is in the dict while the same window of current is
                # not, the correction is valid.
                pos = current.find(bad_str)
                found = False
                for scan_start in range(pos + 1):
                    # Window must cover the whole corrected span and be long
                    # enough to be evidence of a real word.
                    min_end = max(pos + len(good_str),
                                  scan_start + _MIN_LIGATURE_WINDOW)
                    for end in range(len(candidate), min_end - 1, -1):
                        win_cand = candidate[scan_start:end]
                        win_cur  = current[scan_start:end]
                        if win_cand in master_dict and win_cur not in master_dict:
                            # Emphatic-ஏ guard: reject window if the bad_str
                            # ends the window and the base form (with pulli) is
                            # a valid word — that ே is emphatic, not OCR ை.
                            if (bad_str in _EMPHATIC_FINALS
                                    and win_cur.endswith(bad_str)
                                    and (win_cur[:-len(bad_str)] + bad_str[0] + "்")
                                    in master_dict):
                                continue
                            if verbose:
                                replacements_made.append(
                                    f"{original_word} -> {candidate} (window:{win_cand})"
                                )
                            current = candidate
                            changed = True
                            found = True
                            break
                    if found:
                        break
                if changed:
                    break

        if current != original_word:
            fixed_count += 1

        return current

    cleaned_text = re.sub(r"[\u0B80-\u0BFF]+", _attempt_fix, text)

    if verbose:
        print(f"[Fix 28] Historical lig.  — fixed: {fixed_count}"
              + (f": {', '.join(replacements_made)}" if replacements_made else ""))

    return cleaned_text


# ══════════════════════════════════════════════════════════════════════════════
# Fix 29: Grammar-based participial compound merging
# ══════════════════════════════════════════════════════════════════════════════

def fix_participial_compounds(text: str, verbose: bool = True) -> str:
    """Fix 29: Merge OCR-split Tamil verbal participle compounds.

    Targets cases where OCR inserts a space between a verbal stem and a
    participial/verbal-noun suffix, e.g.:
      இருந்த வர்கள்  →  இருந்தவர்கள்   (those who were)
      வெளியிடு வதற்கு →  வெளியிடுவதற்கு (for the purpose of publishing)
      செய்த வர்       →  செய்தவர்        (the one who did)

    Strategy: no corpus dependency.
      - Right token must equal OR start with a suffix in _PARTICIPIAL_SUFFIXES
        (longest match first).  Handles cases where OCR fused the suffix with
        the following word: "வர்கள்பண்டிதர்" → suffix "வர்கள்" extracted,
        remainder "பண்டிதர்" kept as a separate token.
      - Left token must be all Tamil characters (any length).
      - Merge is unconditional once conditions are met.

    Runs BEFORE IndicNLP so the sentence splitter sees correct word forms.
    """
    merged = 0
    examples: List[str] = []

    def merge_line(line: str) -> str:
        nonlocal merged
        parts = re.split(r"(\s+)", line)
        i = 0
        out: List[str] = []
        while i < len(parts):
            tok = parts[i]
            if _TAMIL_TOKEN_RE.match(tok) and i + 2 < len(parts):
                right = parts[i + 2]
                matched_suffix = None
                for sfx in _PARTICIPIAL_SUFFIXES:
                    if right == sfx or right.startswith(sfx):
                        matched_suffix = sfx
                        break
                if matched_suffix:
                    remainder = right[len(matched_suffix):]
                    combined = tok + matched_suffix
                    if verbose and len(examples) < 10:
                        orig = f"{tok} {right}"
                        fixed = combined + (" " + remainder if remainder else "")
                        examples.append(f"{orig} → {fixed}")
                    out.append(combined)
                    merged += 1
                    if remainder:
                        out.append(parts[i + 1])  # restore whitespace before remainder
                        out.append(remainder)
                    i += 3   # skip: left-token, whitespace, right-token
                    continue
            out.append(tok)
            i += 1
        return "".join(out)

    lines = text.split("\n")
    new_lines = [merge_line(ln) for ln in lines]
    if verbose:
        print(f"[Fix 29] Participial merge — merged: {merged}"
              + (f": {', '.join(examples)}" if examples else ""))
    return "\n".join(new_lines)


# ══════════════════════════════════════════════════════════════════════════════
# Indic NLP sentence splitting
# ══════════════════════════════════════════════════════════════════════════════

def run_indic_nlp(text: str, verbose: bool = True) -> str:
    try:
        normalizer, sentence_tokenize = _get_indic_nlp()

        lines = text.split("\n")
        heading_map: dict = {}
        protected: List[str] = []
        for ln in lines:
            if ln.endswith("_HEADING_"):
                ph = f"\u229bH{len(heading_map)}\u229b"
                heading_map[ph] = ln
                protected.append(ph)
            else:
                protected.append(ln)
        text = "\n".join(protected)

        text       = protect_abbreviations(text, verbose=verbose)
        normalized = normalizer.normalize(text)
        sentences  = sentence_tokenize.sentence_split(normalized, lang="ta")
        sentences  = [s.strip() for s in sentences if s.strip()]
        result     = "\n".join(sentences)
        result     = restore_abbreviations(result, verbose=verbose)

        for ph, original_line in heading_map.items():
            result = result.replace(ph, original_line)

        if verbose: print(f"[Final] Indic NLP         — sentences: {len(sentences)}")
        return result

    except ImportError:
        if verbose: print("[Final] Indic NLP not installed — skipping")
        return text


# ══════════════════════════════════════════════════════════════════════════════
# Post-sentence cleanup (Fix 18 integrated)
# ══════════════════════════════════════════════════════════════════════════════

def post_sentence_cleanup(text: str, verbose: bool = True) -> str:
    aytham_colon = len(re.findall(r"[\u0B80-\u0BFF]ஃ\s", text))
    if aytham_colon:
        text = re.sub(r"([\u0B80-\u0BFF])ஃ(\s)", r"\1:\2", text)
        if verbose: print(f"[Post] ஃ→: restoration    — fixed: {aytham_colon}")

    def balance_round_brackets(line: str) -> str:
        opens, closes = line.count("("), line.count(")")
        if closes > opens:
            extra = closes - opens
            while extra > 0:
                idx = line.rfind(")")
                if idx == -1: break
                line = line[:idx] + line[idx + 1:]; extra -= 1
        return line

    lines = text.split("\n")
    cleaned = []
    dropped_punct_only = changed_lines = 0

    for line in lines:
        original = line; line = line.strip()
        is_heading = line.endswith("_HEADING_")
        if is_heading:
            line = line[:-len("_HEADING_")].strip()

        if not line or re.fullmatch(r"[-,.;:?!()\{\}\[\]·•]+", line):
            if line: dropped_punct_only += 1; continue
        if re.fullmatch(r"[\d\s,.;:-]+", line):
            dropped_punct_only += 1; continue
        if re.fullmatch(r"[\d\s\"'.,;:-]+", line):
            dropped_punct_only += 1; continue

        line = re.sub(r"^(?:\d+\s*[.)-]\s*){2,}(?=\S)", "", line)
        line = re.sub(r"^(?:\d+\s+){1,3}(?=[\u0B80-\u0BFF])", "", line)
        line = re.sub(r"\s+\d+\s*[-.,]\s*\d+(?:\s*[-.,]\s*\d+)*\.?\s*$", "", line)
        line = re.sub(r"\(\s*'\s*\)", "", line)
        line = re.sub(r"\({2,}", "(", line); line = re.sub(r"\){2,}", ")", line)
        line = re.sub(r"\(\s+", "(", line); line = re.sub(r"\s+\)", ")", line)
        line = re.sub(r"(?<!\S)[()](?!\S)", "", line)
        if line.startswith("(") and line.count("(") > line.count(")"):
            line = line[1:].lstrip()
        line = balance_round_brackets(line)
        line = re.sub(r"\s+([,.;:?!])", r"\1", line)
        line = re.sub(r"([,.;:?!])(?:\s*[,.;:?!]){1,}", r"\1", line)
        line = re.sub(r"  +", " ", line).strip()

        if line and line != original.strip(): changed_lines += 1
        if line:
            if is_heading:
                cleaned.append(line + "_HEADING_")
            else:
                cleaned.append(line)

    cleaned, merged_initial = fix_initial_line_fragments(cleaned)
    if verbose: print(f"[Fix 18] Initial fragments — merged: {merged_initial}")

    merged_short = 0; final: List[str] = []; i = 0
    while i < len(cleaned):
        line = cleaned[i]
        is_heading = line.endswith("_HEADING_")
        clean_line = line.replace("_HEADING_", "").strip()

        has_end = bool(re.search(r"[.?!:;]\"?$", clean_line))

        if (not is_heading
                and i + 1 < len(cleaned)
                and len(clean_line.split()) <= 6
                and not has_end):
            next_line = cleaned[i + 1].replace("_HEADING_", "").strip()
            final.append(f"{clean_line} {next_line}".strip())
            merged_short += 1; i += 2; continue

        final.append(clean_line)
        i += 1

    if verbose:
        print(f"[Post] Sentence cleanup    — dropped lines: {dropped_punct_only}, "
              f"changed lines: {changed_lines}, merged short lines: {merged_short}")
    return "\n".join(final)


# ══════════════════════════════════════════════════════════════════════════════
# Page splitter
# ══════════════════════════════════════════════════════════════════════════════

def extract_pages(raw_text: str) -> List[Tuple[int, str]]:
    raw_text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    pattern = re.compile(
        r"={10,}[^\n]*\n"
        r"PAGE\s+(\d+)[^\n]*\n"
        r"={10,}[^\n]*\n"
        r"(.*?)"
        r"(?=={10,}|\Z)",
        re.DOTALL,
    )
    pages = [(int(m.group(1)), m.group(2)) for m in pattern.finditer(raw_text)]
    if not pages:
        pages = [(1, raw_text)]
    return pages


# ══════════════════════════════════════════════════════════════════════════════
# Single-page cleaning
# ══════════════════════════════════════════════════════════════════════════════

def _clean_page(
    text: str,
    master_dict: set,
    verbose: bool = False,
) -> str:
    """Run the full v5j cleaning pipeline on a single page's text."""
    text = strip_structural_markup(text, verbose=verbose)
    text = fix_lone_vowels(text, verbose=verbose)
    text = fix_mid_aytham(text, verbose=verbose)          # Fix 21 — before Fix 2
    text = fix_aytham(text, verbose=verbose)
    text = fix_garbage_lines(text, verbose=verbose)
    text = fix_duplicates(text, verbose=verbose)
    text = fix_fused_punct(text, verbose=verbose)
    text = fix_citations(text, verbose=verbose)
    text = fix_whitespace(text, verbose=verbose)
    text = fix_more_citations(text, verbose=verbose)
    text = fix_semicolons(text, verbose=verbose)
    text = fix_stray_punct(text, verbose=verbose)
    text = normalize_unicode(text, verbose=verbose)       # Fix 11 / 23b — NFC first
    text = fix_empty_brackets(text, verbose=verbose)
    text = fix_bracket_artifacts(text, verbose=verbose)
    text = fix_number_punct_garbage(text, verbose=verbose)
    text = fix_footnote_numbers(text, verbose=verbose)
    text = fix_trailing_fragments(text, verbose=verbose)
    text = fix_ocr_pulli_splits(text, master_dict, verbose=verbose)
    text = fix_intra_word_spaces(text, master_dict, verbose=verbose)      # Fix 22
    text = fix_participial_compounds(text, verbose=verbose)               # Fix 29 — before IndicNLP
    text = run_indic_nlp(text, verbose=verbose)
    text = post_sentence_cleanup(text, verbose=verbose)
    text = fix_historical_ligatures(                      # Fix 28 — last, after sentences split
        text, master_dict, verbose=verbose
    )
    return text


# ══════════════════════════════════════════════════════════════════════════════
# Master pipeline (v5j)
# ══════════════════════════════════════════════════════════════════════════════

def clean_tamil_text(
    input_file: str,
    output_file: str,
    master_dict_pkl: str = "ta_master_dict.pkl",
) -> None:
    print(f"\n{'='*52}")
    print(f"  Tamil Text Cleaning Pipeline v5j")
    print(f"  Corpus + IndicNLP + OCR fixes + Fix 28 + Fix 29")
    print(f"  Input : {input_file}")
    print(f"  Output: {output_file}")
    print(f"{'='*52}\n")

    raw = Path(input_file).read_text(encoding="utf-8")
    print(f"  Starting chars : {len(raw):,}")

    pages = extract_pages(raw)
    print(f"  Pages detected : {len(pages)}")

    # Load master dictionary for Fix 28
    pkl_path = Path(master_dict_pkl)
    if not pkl_path.is_absolute():
        pkl_path = Path(input_file).parent / master_dict_pkl
    if pkl_path.exists():
        master_dict: set = pickle.load(open(pkl_path, "rb"))
        print(f"  Master dict    : {len(master_dict):,} words (Fix 28 active)")
    else:
        master_dict = set()
        print(f"  Master dict    : not found at {pkl_path} — Fix 28 will be skipped")
    print()

    cleaned_pages: List[str] = []

    for page_num, content in pages:
        print(f"--- PAGE {page_num} ---")
        cleaned = _clean_page(content, master_dict, verbose=True)
        if re.search(r"[\u0B80-\u0BFF]", cleaned):
            cleaned_pages.append(cleaned)
            print(f"  → {len(cleaned.splitlines())} lines kept")
        else:
            print(f"  → (no Tamil content, page skipped)")
        print()

    result = "\n\n".join(cleaned_pages)

    Path(output_file).write_text(result, encoding="utf-8")

    total_lines = len(result.splitlines())
    print(f"\n  Total pages with content : {len(cleaned_pages)}")
    print(f"  Total output lines       : {total_lines}")
    print(f"  Final chars              : {len(result):,}")
    print(f"✅ Saved to                : {output_file}\n")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tamil OCR text cleaning pipeline v5j")
    parser.add_argument("input_file", help="raw OCR text file (UTF-8)")
    parser.add_argument("output_file", help="where to write the cleaned text")
    parser.add_argument(
        "--dict",
        default=str(Path(__file__).resolve().parent / "ta_master_dict.pkl"),
        help="master dictionary pickle for Fix 28 (default: ta_master_dict.pkl next to this script)",
    )
    args = parser.parse_args()
    clean_tamil_text(args.input_file, args.output_file, master_dict_pkl=args.dict)
