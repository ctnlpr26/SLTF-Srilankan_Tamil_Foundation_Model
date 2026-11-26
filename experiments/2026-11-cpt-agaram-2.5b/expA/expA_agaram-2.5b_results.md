*Source: `/kaggle/working/expA_agaram-2.5b_full`*

# Experiment A · agaram-2.5b (murai-labs/agaram-2.5b + ctnlpr/Agaram-2.5B-SLTFM-CPT-LoRA) · Training vs held-out next-token prediction

Bits per character (BPC, lower is better), same windowing as E2 (window 2048, stride 1024). Gain = % fewer bits per character than the base model on the same text. 95% CIs: bootstrap over documents, paired across checkpoints.

| Set | Documents | Characters |
|---|---|---|
| Training docs (seen) | 13 | 265,712 |
| Held-out docs | 13 | 260,108 |
| Tamil Wikipedia (control) | 33 | 263,774 |

| Step | Epoch | Tokens seen | Training docs (seen) BPC | gain % | Held-out docs BPC | gain % | Tamil Wikipedia (control) BPC | gain % | Train − held-out gain (pp) |
|---|---|---|---|---|---|---|---|---|---|
| base | 0 | 0 | 1.164 (1.054–1.294) | — | 1.100 (0.985–1.209) | — | 0.678 (0.626–0.728) | — | — |
| 125 | 0.18 | 0.5M | 1.111 (1.008–1.235) | 4.5 (3.9–5.1) | 1.057 (0.942–1.169) | 3.9 (2.9–5.0) | — | — | +0.7 (-0.7 to +1.8) |
| 350 | 0.50 | 1.4M | 1.073 (0.982–1.184) | 7.8 (6.4–9.1) | 1.053 (0.937–1.165) | 4.3 (3.2–5.6) | — | — | +3.5 (+1.5 to +5.3) |
| 695 | 1.00 | 2.8M | 1.057 (0.971–1.162) | 9.2 (7.4–10.8) | 1.049 (0.934–1.162) | 4.6 (3.4–6.0) | 0.754 (0.711–0.798) | -11.2 (-14.4 to -8.8) | +4.6 (+2.3 to +6.7) |

## What the numbers say

- At step 695: 9.2% gain on training text vs 4.6% on held-out text (gap +4.6 pp, 95% CI +2.3 to +6.7).
- Held-out Noolaham text improved 4.6%, Tamil Wikipedia -11.2% (95% CI -14.4 to -8.8).

## Held-out documents: base vs step 695

| Document | Base BPC | SLTFM BPC | Gain % |
|---|---|---|---|
| புற்றுநோய் ஓர் அறிமுகம் | 0.979 | 0.891 | 9.1 |
| வல்வெட்டித்துறை வைத்தியசாலை வைரவிழா மருத்துவ மலர் 2010 | 0.909 | 0.829 | 8.8 |
| மலேரியாவும் அதனைத் தடுத்தலும் | 1.146 | 1.056 | 7.8 |
| நீரிழிவு வியாதியும் அதுபற்றிய சில அனுபவக் குறிப்புக்ளும் | 0.927 | 0.864 | 6.7 |
| யாழ் மாநகரசபை சுகாதார விழா சிறப்பு மலர் 1980 | 1.108 | 1.057 | 4.6 |
| சித்த மருத்துவம் 1992-1993 | 1.286 | 1.228 | 4.5 |
| noolaham_65873_4. Dr.Ida Scudder  03.07.2022 | 0.840 | 0.804 | 4.3 |
| வல்வெட்டித்துறை வரலாற்றுச் சுவடுகள் | 1.052 | 1.008 | 4.2 |
| சித்த மருத்துவம் 1988-89 | 1.279 | 1.227 | 4.1 |
| இலங்கை மனித உரிமைகள் நிலை 2000 | 0.805 | 0.784 | 2.7 |
| யாழ்ஜெய்ப்பூர் கால்திட்டம் ஏழாவது ஆண்டு நிறைவு - விளையாட்டு விழா மலர்  | 1.242 | 1.214 | 2.2 |
| பாலவைத்தியபோதினி | 1.443 | 1.418 | 1.7 |
| noolaham_66569_ Dr. Green First Surgery 11.09.2022 (4) | 0.795 | 0.784 | 1.4 |