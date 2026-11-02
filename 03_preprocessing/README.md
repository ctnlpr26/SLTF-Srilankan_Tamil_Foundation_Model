# 03 · Tamil text normalization & preprocessing: Tamil OCR Text Cleaner

Cleans raw OCR text from scanned Tamil books and articles (noolaham.org). In SLTFM it is the
step before [`02_corpus`](../02_corpus): its output files (`content_postprocessed.txt`, one per
document) are what `build_sltfm_corpus.py` turns into the Hugging Face corpus. It was first built
for knowledge-graph (KG) work, so it puts extra care into keeping named entities intact.

It removes OCR noise (stray symbols, page numbers, English-only fragments), repairs
broken Tamil words, joins lines that were split by the page layout, and writes clean
text, one sentence per line where possible.

```
python tamil_ocr_cleaner.py input.txt output_cleaned.txt
```

---

## Before and after

From *யாழ்ப்பாண பல்கலைக்கழகம் உருவான வரலாறு* (raw OCR, then cleaned output):

**Before**

```
சிந்தனை - 4
யாழ்ப்பாண பல்கலைக்கழகம்
உருவான
வரலாறு
-
ச.சுப்பிரமணியம்
யாழ்ப்பாண பல்கலைக்கழகம் தனது இருபத்தைந்து வருடப்
பூர்த்தியை வெள்ளி விழாவை அண்மையில் கொண்டாடியது.
விழாவை முன்னிட்டு மலர் ஒன்று (ஆங்கிலத்தில் மட்டும்)
வெளியிடப்பட்டதுடன், பல நிகழ்ச்சிகளும் ஏற்பாடு செய்யப்
பட்டிருந்தன.பல்கலைக் கழகத்தின் கடந்தகால பன்முக வளர்ச்சி
```

**After**

```
சிந்தனை 4 யாழ்ப்பாண பல்கலைக்கழகம்
உருவான வரலாறு ச. சுப்பிரமணியம்
யாழ்ப்பாண பல்கலைக்கழகம் தனது இருபத்தைந்து வருடப் பூர்த்தியை வெள்ளி விழாவை அண்மையில் கொண்டாடியது.
விழாவை முன்னிட்டு மலர் ஒன்று (ஆங்கிலத்தில் மட்டும்) வெளியிடப்பட்டதுடன், பல நிகழ்ச்சிகளும் ஏற்பாடு செய்யப்
பட்டிருந்தன.
பல்கலைக் கழகத்தின் கடந்தகால பன்முக வளர்ச்சி பற்றியும், ...
```

Word-level repairs, from [examples/](examples/) and the test books below:

| Before | After | Fix |
|---|---|---|
| இருந்த வர்கள் | இருந்தவர்கள் | 29: participial compound |
| வெளியிடு வதற்கு | வெளியிடுவதற்கு | 29: participial compound |
| வெள்ள ரிக்காய் | வெள்ளரிக்காய் | 22: space inside a word |
| எல்லோரை யும் | எல்லோரையும் | 22: space inside a word |
| என் பனவாகும் | என்பனவாகும் | 22: space inside a word |

---

## Tested on real books

Six documents from the noolaham.org collection were run through the cleaner (modern-script
OCR, 14K–145K characters each). Each took about 2 seconds, most of it loading the dictionary.

| Document | Type | Chars in → out | Lines in → out | Garbage lines removed | Words repaired* |
|---|---|---|---|---|---|
| 1947 இல் உதயமான இந்திராணி ஆஸ்பத்திரி | Hospital history | 14,697 → 14,413 | 403 → 269 | 23 | 1 |
| பரராசசேகரம் – பித்தரோக நிதானமும் சிகிச்சையும் | Classical medical verse | 57,248 → 55,818 | 2,244 → 938 | 236 | 40 |
| மருத்துவத் தமிழ் முன்னோடி டாக்டர் கிறீன் | Biography | 144,562 → 134,924 | 3,864 → 2,299 | 392 | 14 |
| யாழ்ப்பாண பல்கலைக்கழகம் உருவான வரலாறு | History | 39,430 → 38,527 | 832 → 520 | 7 | 5 |
| நாடி 1994 | Medical magazine | 91,531 → 88,124 | 3,307 → 1,872 | 238 | 46 |
| ஐக்கிய சங்க ஆசுபத்திரித் தொண்டின் ஆரம்ப முயற்சிகள் | Article | 16,220 → 15,952 | 559 → 331 | 7 | 19 |

\* Fix 19 + Fix 22 + Fix 28 + Fix 29 combined.

What was removed is mostly page numbers, running headers, English-only lines and OCR
debris (`07.`, `Gmai`, `DSTUEGO GO`). Lines that mix Tamil with English glosses, such as
`உள்ளம்சார் உடல்நோய் (Psychosomatic Disease)`, are kept.

---

## Setup

Requires Python 3.8+ (tested on 3.10).

```bash
git clone https://github.com/ctnlpr26/SLTF-Srilankan_Tamil_Foundation_Model.git
cd SLTF-Srilankan_Tamil_Foundation_Model/03_preprocessing
pip install -r requirements.txt
git clone https://github.com/anoopkunchukuttan/indic_nlp_resources.git
```

Keep `indic_nlp_resources/` next to the script, or point to it with an environment variable:

```bash
export INDIC_RESOURCES_PATH=/path/to/indic_nlp_resources      # Linux/macOS
$env:INDIC_RESOURCES_PATH = "C:\path\to\indic_nlp_resources"   # PowerShell
```

**Dictionary.** `ta_master_dict.pkl` (about 2.1M Tamil words, 78 MB) is **not stored in this
repository** because of its size. Get a copy from the maintainer (it will move to a private
Hugging Face Hub repo) and put it in this folder, or pass its path with `--dict`. If it is
missing, the cleaner still runs but skips every dictionary-checked fix (19, 22 and 28).

Note: the dictionary is a Python pickle. Only load a copy from a trusted source; loading a
pickle can run code.

## Usage

```bash
python tamil_ocr_cleaner.py input.txt output_cleaned.txt
python tamil_ocr_cleaner.py input.txt output_cleaned.txt --dict path/to/ta_master_dict.pkl
```

The input must be UTF-8. It can be split into pages with headers like this:

```
==========
PAGE 1
==========
...text...
```

A file without page headers is treated as a single page. The cleaner prints a log line
for each fix with what it changed, so you can check its work on a new book.

### As a library

```python
from tamil_ocr_cleaner import clean_tamil_text, _clean_page

clean_tamil_text("input.txt", "output.txt")
```

### Cleaning many books

```bash
for f in books/*/content.txt; do
  python tamil_ocr_cleaner.py "$f" "${f%.txt}_cleaned.txt" > "${f%.txt}_clean.log"
done
```

---

## What each fix does

| Fix | What it does |
|---|---|
| 0 | Strips structural markup: HTML tags, separator lines, PAGE headers, LaTeX, bullets, image references |
| 1 | Removes vowel signs left on their own with no consonant |
| 2 | Treats `ஃ` before a space as a misread full stop |
| 3 | Removes garbage lines: English-only lines, short symbol fragments, mostly non-Tamil lines |
| 4 | Re-joins paragraphs broken by the page layout |
| 5–16 | Hyphens, citations, whitespace, stray `;` and punctuation, Unicode NFC, empty brackets, footnote numbers, trailing fragments |
| 17 / 17b | Protects honorifics (`திரு.`, `டாக்டர்.`) and initials (`ச.`, `வ.`) so sentence splitting does not break on them |
| 18 | Re-joins initials that sentence splitting separated from the name |
| 19 | Joins words split after a pulli (`்`), only when the joined word is in the dictionary |
| 21 | Fixes `ஃ` misread in the middle of a word |
| 22 | Removes spaces inside a word, only when one half is not a real word and the joined form is |
| 23b | Fixes the combining-character sequence `ு + ே` → `ோ` |
| 28 | Corrects pre-reform Tamil script letter shapes misread by OCR (old `ை` / `ா` / `ொ` / `ோ` glyphs), checked against the dictionary |
| 29 | Joins participial compounds split by OCR (`இருந்த வர்கள்` → `இருந்தவர்கள்`) using grammar rules |
| — | Sentence tokenization with IndicNLP |

The order of the steps is in `_clean_page()` in [tamil_ocr_cleaner.py](tamil_ocr_cleaner.py).

---

## Recent changes (found during the book test)

- **Fix 22 joined real two-word phrases.** The dictionary is built from web text, so it
  also holds spellings with the space missing, and Fix 22 joined `ஆம் ஆண்டு`,
  `தமிழ் மக்கள்`, `எந்த ஒரு` and similar pairs. That is about 1,000 wrong joins across the
  six books, and it hurts entity extraction. Fix 22 now joins two halves only when at
  least one of them is not a valid word on its own.
- **Fix 28 corrupted correct modern Tamil.** It changed the emphatic `-ஏ` ending
  (`உடனே` → `உடனை`, `புரட்டினேன்` → `புரட்டினைன்`) and produced invalid Unicode
  (`வண்ணஞ்` → `வண்ணா்`). It made 69 changes, nearly all wrong. Now:
  - the emphatic-`ஏ` guard covers `னே`, `லே`, `ணே` as well as `ளே`;
  - the fused-word fallback needs a dictionary match of at least 6 characters
    (it used to accept 3-character matches such as `டனை`);
  - a correction that would put two vowel signs or a vowel sign and `்` together is rejected.
- **Fix 3 dropped useful mixed-language lines.** Lines with at least two Tamil words
  (12+ Tamil characters) are now kept even when English or numbers make up most of the line.

## Known limitations

- **Fix 28 still makes a few mistakes in classical verse.** For example, it changes the
  vocative `மின்னே` to `மின்னை` (8 times in பரராசசேகரம்). It is designed for books printed
  in the pre-reform script. For modern-script books you can run without it by passing a
  path that doesn't exist to `--dict`, but that also turns off Fixes 19 and 22.
- **Words split across a line break are not re-joined.** For example, `செய்யப்` /
  `பட்டிருந்தன` stays as two words.
- **Lines with many 1-character tokens are dropped.** Fix 3 treats them as garbage, which
  also catches some real lines with number ranges, e.g. `ஆனால் 20 - 30 வயதுடைய`.
- **Lines don't always hold exactly one sentence.** When the source has no full stops (verse,
  tables), lines follow the layout.

---

## Folder layout

```
tamil_ocr_cleaner.py     the cleaner (CLI + library)
requirements.txt         Python dependencies
examples/                small input and its expected output
ta_master_dict.pkl       Tamil word list for the dictionary-checked fixes (not committed; see Setup)
indic_nlp_resources/     IndicNLP resources (clone separately; not committed)
```

To check that the cleaner still works after a change:

```bash
python tamil_ocr_cleaner.py examples/sample_input.txt out.txt
diff out.txt examples/sample_output.txt        # Linux/macOS: no output means it matches
fc.exe out.txt examples\sample_output.txt      # Windows PowerShell: "no differences encountered"
```

---

## Version history

| Version | Change |
|---|---|
| v5j | First version (`clean_pipeline_v5j.py`) |
| current | Renamed to `tamil_ocr_cleaner.py`; fixes to 22, 28 and 3 (see Recent changes) |

## To do

- Check Fix 29 on words that start with a suffix-like part (e.g. `திரு வள்ளுவர்`) so it does
  not split names.
- Make relative `--dict` paths resolve from the working directory, not the input file's folder.
- Remove old wording ("v5j" in the banner and help text).
- Add a switch to turn Fix 28 off on its own for modern-script books.
- Move the dictionary to the Hugging Face Hub (or a plain-text word list) and add a download step.