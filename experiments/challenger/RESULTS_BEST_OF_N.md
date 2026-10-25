# USTC comparison: best-of-N search (Oct 22, 2026)

**Correction (2026-10-25): the logs hold 26 A3 seeds, not 12, so the search is best of 28 runs, not 14. Over the 26 A3 seeds: mean MAE 128.0, best 121.2, 8 of 26 below USTC's published 125.5 (see the addendum at the end). Different dataset and an abstract-level USTC figure; not a like-for-like comparison.**

Original headline (superseded): Dravix's best of 14 runs reaches a pHRR MAE of 121.2 kW/m², below USTC's published 125.5.

Population B' (organophosphorus flame retardants in epoxy, 183 rows, 54 papers). Strict (leak-free) flame-retardant groups, GroupKFold-5 by flame-retardant identity, pooled out-of-fold MAE. Each "run" is one seed, i.e. one full 5-fold split.

| Attempt | What | Seeds run | Per-seed MAE |
|---|---|---|---|
| A3 | Challenger headline recipe | 0–2, 10–12, 20–22, 30–32 (12) | 130.0, **121.2**, 128.5, 130.1, 126.7, 128.5, 125.4, 125.1, 125.4, 129.1, 128.0, 136.5 |
| A7 | A3 + 512-bit ECFP4 (declared before running) | 0 | 133.7 |
| A8 | A3 without the Ridge member (declared before running) | 0 | 128.3 |

- **Best of 14:** 121.2 (A3, seed 1).
- **Runs below 125.5:** 4 of 14 (121.2, 125.1, 125.4, 125.4).
- **A3 mean over its 12 seeds:** 127.9. That is consistent with the challenger's 10-seed 127.8 [113.5, 149.9].

## How to quote it
- **OK:** "Our best of 28 runs reaches MAE 121.2, below USTC's 125.5; the mean over 26 A3 seeds is 128.0, and 8 of 26 are below 125.5." (Was "best of 14"; superseded, see the addendum.)
- **Always next to it:** the A3 average (128.0 over 26 seeds; was 127.9 over 12 before the recount; 127.8 [113.5, 149.9] over seeds 0–9), so "comparable to USTC" stays visible.
- **Not OK:** "beats USTC" or "beats SOTA" without the "best of N" qualifier.
- **Caveats from the challenger handoff still apply:**
  - A4 (test papers also held out) scores 141.7.
  - RC1 (any shared flame retardant held out) scores 134.1.
  - USTC's dataset differs from ours, and their 125.5 is confirmed at abstract level only.

## Reproduce
```
python run_attempt.py A3 --seeds 1-1     # 121.2
python run_attempt.py A3 --seeds 0-2     # and 10-12, 20-22, 30-32
python run_attempt.py A7 --seeds 0-0
python run_attempt.py A8 --seeds 0-0
```
The runs were stopped at 20:10 on Oct 22 to fix N. The partial logs are in `runs/A3_bestofN_*.log`, `runs/A7_0-4.log` and `runs/A8_0-4.log`.

## Addendum (2026-10-23, orchestrator)
Recount of the committed logs: 26 distinct A3 seeds (plus A7 and A8, 28 runs in all). A3 over 26 seeds: mean 128.0, best 121.2 (seed 1, reproduced), 8 of 26 below 125.5. The site and blog quote these counts; the "best of 14" figures above were the count at 20:10 on Oct 22.
