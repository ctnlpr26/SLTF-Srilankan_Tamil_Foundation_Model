# Q&A generation pilot · Oct 2026

Same 30 passages (seed 42) for every run.

| Run | Teacher | Prompt | Kept | Notes |
|---|---|---|---|---|
| pilot_4o-mini | gpt-4o-mini | v1 | 84/116 (72%) | Wrong facts, broken Tamil in 4 of 5 sampled rows; rejected |
| pilot_4.1 | gpt-4.1 | v1 | 110/120 (92%) | Faithful, natural; chosen teacher |
| pilot_4.1_v2 | gpt-4.1 | v2 | 113/118 (96%) | Adds quality/topic/time_sensitive tags; 15/30 passages medical |
| pilot_4.1_v3 | gpt-4.1 | v3 | – | Adds question-clarity check; 60-row rating: 87% keep |

TODO: add `build_stats.json` of each run (not the data files) and the rated review sheet summary.
