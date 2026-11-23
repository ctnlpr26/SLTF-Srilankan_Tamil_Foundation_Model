# E2 · Contamination check for Agaram 2.5B

**Date:** 23 Nov 2026 · **Input:** the per-segment results of E2 (`e2_full/*.json`, Kaggle v2, 16 Nov 2026)

## Why
Agaram 2.5B had by far the lowest BPC in E2 (1.097 vs 1.309 for Qwen 3.5 9B). That could be genuine skill, or Agaram may have seen some of our validation books during its own pretraining (contamination).

## Method
- BPC per validation book for every model, from the same E2 run (no new scoring).
- Ratio **Agaram BPC / Qwen 3.5 9B BPC** per book. Qwen 9B is the closest-scoring model with no known link to our data.
- Reading: a roughly constant ratio means a general advantage. A few books with a much lower ratio (≈0.4–0.6) would mean those books were memorised.

## Results

| # | Chars | Llama 3 8B | Our CPT | Qwen 3.5 4B | Qwen 3.5 9B | Agaram 2.5B | Agaram/Qwen9B | Book |
|---|---|---|---|---|---|---|---|---|
| 0 | 7,193 | 1.163 | 0.983 | 1.139 | 1.005 | 0.840 | 0.84 | noolaham_65873 Dr. Ida Scudder |
| 1 | 6,851 | 1.199 | 0.964 | 1.133 | 1.015 | 0.795 | 0.78 | noolaham_66569 Dr. Green First Surgery |
| 2 | 118,491 | 1.148 | 0.964 | 1.043 | 0.949 | 0.777 | 0.82 | இலங்கை மனித உரிமைகள் நிலை 2000 |
| 3 | 159,155 | 1.684 | 1.352 | 1.548 | 1.450 | 1.286 | 0.89 | சித்த மருத்துவம் 1988-89 |
| 4 | 120,204 | 1.589 | 1.251 | 1.445 | 1.341 | 1.194 | 0.89 | சித்த மருத்துவம் 1992-1993 |
| 5 | 27,850 | 1.305 | 1.032 | 1.168 | 1.056 | 0.890 | 0.84 | நீரிழிவு வியாதியும் அதுபற்றிய சில அனுபவக... |
| 6 | 122,630 | 2.262 | 1.751 | 2.005 | 1.898 | 1.397 | 0.74 | பாலவைத்தியபோதினி |
| 7 | 6,245 | 1.362 | 0.896 | 1.274 | 1.128 | 0.979 | 0.87 | புற்றுநோய் ஓர் அறிமுகம் |
| 8 | 54,784 | 1.412 | 1.112 | 1.199 | 1.086 | 1.066 | 0.98 | மலேரியாவும் அதனைத் தடுத்தலும் |
| 9 | 51,993 | 1.538 | 1.195 | 1.388 | 1.280 | 1.120 | 0.87 | யாழ் மாநகரசபை சுகாதார விழா சிறப்பு மலர் |
| 10 | 47,056 | 1.643 | 1.362 | 1.519 | 1.416 | 1.206 | 0.85 | யாழ்ஜெய்ப்பூர் கால்திட்டம் ஏழாவது ஆண்டு... |
| 11 | 163,757 | 1.524 | 1.258 | 1.440 | 1.310 | 1.049 | 0.80 | வல்வெட்டித்துறை வரலாற்றுச் சுவடுகள் |
| 12 | 115,247 | 1.262 | 0.953 | 1.116 | 1.006 | 0.865 | 0.86 | வல்வெட்டித்துறை வைத்தியசாலை வைரவிழா மலர் |

## Findings
- **No outlier books.** The ratio is between 0.74 and 0.98 for all 13 books, mostly 0.80–0.89. Agaram is consistently ~15% better than Qwen 9B.
- The lowest ratio (0.74) is on பாலவைத்தியபோதினி, an older medical text and the hardest book for every model. Agaram's advantage is largest on older Tamil, but 0.74 is far from the memorisation range.
- Books 0, 1 and 7 are short (6–7k characters), so their ratios are noisier.
- Our CPT model beats Qwen 9B on 10 of 13 books. Agaram beats our CPT on every book.

## What Agaram's paper says (arXiv 2610.18420, Sec. 2, 3.2, 5.1, 7)
- **Sources:** Tamil web crawls (FineWeb-2, HPLT, CulturaX, OSCAR, MADLAD-400, Sangraha), Wikipedia, Project Madurai, TamilBooksPDF, Archive.org, OCR of the "Tamil Digital Library", and two further OCR passes (976 PDFs; 358 books) whose sources are not named.
- **Noolaham is not named.** The training corpus is not released, so a direct 13-token overlap check against it is not possible.
- **Timeline:** data collected Dec 2025–Sep 2026; pretraining ended Oct 2026.
- **Sanity check:** the paper reports its own held-out BPC as 1.034 (modern formal) and 1.173 (literary, OCR). Our 1.097 lies between them, so our measurement is consistent with theirs.

## Conclusion
No evidence that specific validation books were memorised. A **uniform** advantage from Noolaham-like data (e.g. via web crawls or Archive.org mirrors) cannot be ruled out with this test.

## Next check
A post-cutoff test set: Sri Lankan Tamil news published in November 2026, after Agaram's training ended. If Agaram is still clearly best there, the advantage is genuine skill. Optional: count `noolaham.org` URLs in FineWeb-2 (Tamil).

## Code
The check is the last cell of `07_evaluation/sltfm_e2_base_model_comparison.ipynb` (per-book BPC from `e2_full/*.json`).
