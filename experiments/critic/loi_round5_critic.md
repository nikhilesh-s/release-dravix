# Critic: LOI round 5, candidate c2 (F6_fixed + LOI test-standard one-hot + log density) vs exp089
Verdict: **SOUND WITH CAVEATS**. c2 meets the declared rule on every seed and I found no leakage, so it can replace exp089. But the round-5 gain on its own is not established: +0.020, with a paired paper-bootstrap CI of [-0.012, +0.059]. Report it as directional.
## Pre-declaration and forking paths
- loi_round5.json was committed at 14:58:34 (ffd726e). That is after the round-4 critic (14:45) and before the run (log 15:05, results e531feb 15:06). The script mtime is 15:03, so the order is OK. exp089 reproduces exactly (0.3542).
- The F6 rules follow the critic's list: bio-first precedence, polyether recall, PLA, heteroaromatic, and the R-ratio iso_index. They go beyond it in three places: the largest-mass component rule, a new other_polyester level, and the 'a:1' ratio parse. These are small design choices.
- c2's extras are outside the critic's list: the loi_standard one-hot and log density. The config's "nothing else is changed" covers c1 only. These were features chosen after seeing round-4 results. They were pre-declared, but there were 2 candidates and the max was taken.
- The fixes are incomplete and add some new errors:
  - The glycolysed-PET strings ("glycolyzed waste PET", "from PET glycolysis") are still unknown (6 benchmark rows). So is "polypropylene glycol (MCNS)" (9).
  - New errors: "MCOP (castor) + PEG-200 (20 g)" becomes polyether (6; only PEG has an amount); "liquefied pine sawdust biopolyol" becomes polyether (the PEG check runs first); "ESBO-DYM"-majority rows (2) become unknown.
## (1) Leakage
- Neither new feature uses LOI. loi_standard is the test-method text. It is constant within every paper (0/86 papers have more than one value), so it is paper-level metadata like F6. GroupKFold by paper prevents memorising a test paper, so it is not leakage.
- Caveat: it acts as a lab/region and missingness proxy. other_unknown is a missing standard (158 rows / 30 papers). Under exp089, other_unknown and GB/T rows have mean residuals of +0.88 and +0.94; c2 shrinks these to +0.65 and +0.77. The feature corrects a reporting-group offset, not chemistry, and only makes sense when predicting a measured-under-X LOI. PN-EN ISO 4589-3 (the elevated-temperature method, 14 rows) is binned with ISO 4589.
- log_density is a monotone transform of density_kgm3, which is already in the feature set, so it is redundant for an RF.
## (2) Paired gain vs exp089 and ablation (seed 0, 2000 paper resamples of the repeat-averaged dB)
| comparison | dB | wins | 95% CI | P(dB<=0) |
|---|---|---|---|---|
| c2 vs exp089 | +0.020 | 9/10 | [-0.012, +0.059] | 0.12 |
| c1 (F6 fixes only) vs exp089 | +0.007 | 7/10 | [-0.012, +0.028] | 0.24 |
| c1 + std only vs exp089 | +0.018 | 9/10 | [-0.015, +0.058] | 0.14 |
| c1 + logdens only vs exp089 | +0.005 | 7/10 | [-0.013, +0.026] | 0.30 |
- About 2/3 of the gain comes from the off-list loi_standard feature. The critic-listed typing fixes add about +0.007, indistinguishable from 0. The top 3 papers (N077, M079, M110) give 82% of the SSE reduction. Dropping any one of them leaves dB at +0.013 to +0.016.
## (3) Cumulative vs exp087 (0.311)
- c2 vs exp087: +0.063, 10/10 wins, CI [+0.017, +0.133], P = 0.002. Across seeds the mean is about +0.060.
- This comes from two rounds of max-of-candidates selection (3 candidates, then 2) after seeing results, so the CI is optimistic. Round 4's part (+0.04) had a CI that just included 0, and round 5's part (+0.02) has a CI that includes 0. A fair statement is that about +0.04 to +0.06 over exp087 is probably real, with roughly the last +0.02 the least secure.
## (4) RF seed stability (all models re-run per seed)
| seed | exp087 | exp089 | c2 | dB vs 089 | wins | c2 rep0 CI | paired CI |
|---|---|---|---|---|---|---|---|
| 0 | 0.311 | 0.354 | 0.374 | +0.020 | 9/10 | [0.201, 0.470] | [-0.012, +0.059] |
| 1 | 0.313 | 0.352 | 0.372 | +0.020 | 9/10 | [0.205, 0.474] | [-0.012, +0.059] |
| 2 | 0.311 | 0.351 | 0.372 | +0.021 | 8/10 | [0.207, 0.479] | [-0.012, +0.061] |
| 3 | 0.314 | 0.351 | 0.373 | +0.022 | 9/10 | [0.207, 0.478] | [-0.010, +0.061] |
The rule passes on all seeds and seed 0 is not cherry-picked. A is 0.700 -> 0.723; C is unchanged (0.257 -> 0.256).
## Caveats to carry with the number
1. Report B 0.374 [0.201, 0.470] with the paired gain over exp089 of +0.02, CI [-0.012, +0.059]. That gain is not significant, and it is mostly test-standard metadata, not formulation chemistry.
2. Polyol typing is still noisy (see above). B remains far below the Sun/Fu reference of 0.84, and the result should never be described as reaching it.
