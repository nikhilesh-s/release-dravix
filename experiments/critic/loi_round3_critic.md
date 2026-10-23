# Critic: LOI round 3 (exp084/085), selected c2 = exp085 (round-1 RF + 6 RDKit FR-structure features + resolved_fr_wt_share)

**Verdict: SOUND WITH CAVEATS.** You can replace exp064 with exp085 as the LOI number of record (0.363 [0.167, 0.517]), because the declared rule passes and the gain survives every check I ran.
But the gain is small: +0.0115 R2, and its paired paper-bootstrap CI touches 0. Two papers carry most of it. The rise in ci_lo from 0.130 to 0.167 is mostly noise from which repeat is used (see 4).
Reviewer: Opus 5.5, fresh context. Scratch: `$SCRATCHPAD/critic_loi3/{analyze,boot,placebo}.py` + logs. I ran the script's own data prep; exp064 and c2 reproduce to 4 dp.

## 1. Leakage / paper identifiers
- Only 33/333 rows have structure features. They come from **11 of 56 papers** (M032 8, M088 4, M071 4, M077 3, M060 3, M160 3, M064/M141/M004 2, M092/M089 1). These rows have just 8 distinct structure vectors.
  4 of the vectors appear in only one paper. 4 are shared across 2–3 papers (DMMP, TCPP, TEP, melamine-type).
- No leakage mechanism: the features come from FR names, never from LOI. Folds are GroupKFold by paper, so a vector unique to one paper is never seen in training when that paper is the test fold. It can only add noise there.
  Only vectors shared across papers can carry information, and that is legitimate transfer.
- Placebo: I permuted the structure vectors among the 33 rows (mask and everything else unchanged, 3 permutations). dB came out -0.0019 / -0.0006 / +0.0006 with 3/3/6 wins out of 10. So the gain depends on the correct chemistry assignment, not on adding columns.
- Ablation: structure features alone give +0.0118 (10/10 wins). resolved_fr_wt_share alone gives +0.0021 (7/10). The 6 RDKit features carry the gain.

## 2. Paired gain vs exp064 (paper bootstrap of the repeat-averaged gain, 2000 draws, seed 0)
- dB = +0.0115, **95% CI [-0.0019, +0.0299]**, P(gain <= 0) = 0.051. It does not clear 0, though only just. The declared rule does not require it to (CLAUDE.md promotion rule: >= 8/10 wins + ci_lo not lower than the incumbent's).
- Where the gain comes from: +0.0080 is on the 33 structure rows and +0.0035 on the other 300 (RF trees shift when columns are added). By paper, M032 gives +0.0050 and M077 +0.0041, so **2 papers give about 80%**.

## 3. RF randomness (diagnostic, RF random_state 0..4; folds unchanged)
- dB: +0.0115 (9/10), +0.0103 (10/10), +0.0109 (9/10), +0.0131 (10/10), +0.0115 (10/10). The gain is not an RF-seed artefact: it is positive with >= 9/10 wins on every seed.
- Paired-gain CIs for seeds 1–4: [-0.0035, +0.030], [-0.0024, +0.030], [-0.0006, +0.032], [-0.0005, +0.029]. They all touch 0. c2 rep-0 ci_lo ranges over 0.163–0.169.

## 4. CIs
- Rep-0 CI, the protocol headline: exp064 [0.130, 0.520] vs c2 [0.167, 0.517].
- Per-repeat ci_lo for exp064: 0.130 0.207 0.195 0.011 0.245 0.189 0.129 0.073 0.255 0.173 (median 0.181). For c2: 0.167 0.234 0.221 0.022 0.260 0.195 0.178 0.101 0.249 0.189 (median 0.192). In 8/10 repeats c2's ci_lo is >= 0.130.
- Pooled (repeat-averaged R2, paper bootstrap): exp064 [0.182, 0.458], c2 [0.202, 0.465]. The underlying ci_lo gain is about +0.01–0.02, not the +0.037 the rep-0 headline suggests. Repeat 3 drops ci_lo to about 0.02 for both models.

## 5. Undeclared rejection rules (forking-path check)
- Timeline: config committed 39bcb00 at 08:57:21. Resolver mtime 09:00:51, fr_structures.csv and its log 09:01:55, train script 09:02:34, run log 09:04:33, all committed in 48fdb1a at 09:04:57.
  The resolver was finalised before the train script existed in its final form. Both were untracked until the commit, so git cannot rule out an earlier unlogged iteration.
- What the extra rules actually did:
  - FR-class rule: 5 names (EG1–3 expandable graphite, Al and Fe phytate). These are already excluded by the config's own text ("graphite", "bio-materials", minerals/salts).
  - Single-atom rule: 1 name (BH resolved to bohrium, CID 56951713). It would have been rejected anyway as an inorganic non-C hit.
  - Keyword-element rule: **0** names. All the rules are subtractive, and none of them fired outside the config's stated exclusions. I see no sign they were tuned to results.
  Cost: BH is a real small-molecule phosphonate. It went unresolved (a missed hit, not a wrong one).

## Caveats to carry with the number
1. Report the result as "B 0.363 [0.167, 0.517]; +0.0115 vs exp064, paired CI [-0.002, +0.030]". Do not claim c2 is significantly better than exp064.
2. The improvement rests on 33 rows / 11 papers, and mostly on M032 and M077.
3. The LOI bar of 0.84 is far away. This is a bookkeeping upgrade, not a change in standing.
