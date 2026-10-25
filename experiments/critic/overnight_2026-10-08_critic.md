# Critic: overnight rounds of Oct 7-8, 2026 (round 9 TabPFN + smearing, CR round 2, TTI round 3, round 10 physics) and the pending TTI S675 record change

Fresh-context critic. I did not run these experiments. Everything in this file comes from the committed configs (69ffda8, 284ed43), the code, the logs, the npz/result/gate files (committed in ce34bb1 at 01:43 while this review ran, unchanged from what I read), and my own re-runs in `.venv312` (4 threads max). I changed no file except this one.

| Item | Verdict |
|---|---|
| Round 9 (round9_tabpfn_smearing: pHRR, FIGRA, LOI, CR members, TTI arm) | **SOUND WITH CAVEATS** |
| CR round 2 (cr_round2, joint CR family with round 9) | **SOUND WITH CAVEATS** |
| TTI round 3 (tti_round3 + round-9 tti_arm) | **SOUND WITH CAVEATS** |
| Round 10 (round10_physics) | **SOUND WITH CAVEATS** |
| Pending record change: exp046 re-scored on S675 becomes the TTI number of record | **SOUND WITH CAVEATS**: go ahead, with the note text in section 9 |

Bottom line: no candidate passes adoption gate v2 in any round. Every incumbent stays. I reproduced all nine gate JSONs bit for bit. I found no leakage and no implementation bug that changes a number. The caveats are about process (environment pins, timestamps, one family-composition deviation), about wording, and about the "near misses", whose gains all sit in 5 or fewer groups.

## 1. Pre-declaration, authority and order (git and file times, PDT)
- 22:14:55 `.venv312` created (tabpfn files 22:15:00). 22:16:27 **69ffda8**: D1-D5 amendments recorded. 22:19 `adoption_gate.py`. 22:24-22:25 V2 weights fetched, round-9 step 0 (synthetic). 22:27 TTI-arm step 0.
- First label-touching scores: about 22:28 (pHRR and FIGRA round 9), about 22:30 (round-9 LOI, TTI step 1), about 22:31 (cr_round2). **Every config was committed before its first score.**
- Round-9 gates: FIGRA 22:38:24, pHRR 22:41:25. **284ed43** at 22:45:34 commits the round-10 config, the round-9 environment erratum and the `rescore_loi_snapshot.py` .venv skip.
- Round 10: pHRR from 22:54, FIGRA 23:00, CR 23:02:46 (round-9 CR joint gate was 23:01:13), LOI 23:11-23:23. TTI gate 23:06:38. Round-9 LOI driver ran from about 22:30 to 01:04; its gate ran at 01:42:43.
- Authority. Every "Arnav approves" gate (D1 win-rule change, D2, D3 install, D4 un-park TTI and choose S675, the round-10 budget overrun) was answered by the orchestrator itself, quoting an owner (Nikhilesh S) delegation. The delegation text is quoted in the amendments; this repo has no independent record of it. The amendments say so openly. The record notes must say so too (section 10).
- **Amendment timestamps contradict git.** The D1-D4 amendments say "Authority (2026-10-07 22:45 PDT)" but were committed at 22:16:27. The round-9 erratum says "~23:20 PDT" and round10 `declared` says "~23:20 PDT", but both were committed at 22:45:34. The in-text clock runs about 30-35 minutes late. Git order is what counts, and it is clean, but the text should be corrected (section 11).
- Round 10 was declared after the orchestrator had read the round-9 pHRR and FIGRA gate files (disclosed in `prior_looks`). Its candidates come from `methods_menu.md`, which the config calls "written blind to round 9". That claim cannot be checked: the file's last mtime is 22:44:01, after the round-9 FIGRA and pHRR gate files existed (22:38 and 22:41), and it was committed at 22:48:57, after the config that cites it. It is unverifiable, not refuted. Nothing was adopted, so nothing depends on it.

## 2. Environment
- `.venv312`: Python 3.12.12 with torch 2.8.0, scikit-learn 1.7.1, numpy 2.2.6, scipy 1.16.1, pandas 2.3.3. These are the record pins exactly, plus tabpfn 9.1.0 and the eight pinned no-deps packages (`round9_env.txt`). I verified the checkpoint myself: `~/Library/Caches/tabpfn/tabpfn-v2-regressor.ckpt`, 44,390,977 bytes, sha256 `2ab5a07d…0736`, the only .ckpt in the cache. The cache path is `~/...`, not the config's `~/...`; that home directory does not exist here, so this is a necessary adaptation. A `clone()` of `create_default_for_version(V2, device='cpu', random_state=0)` keeps the V2 `model_path` (checked).
- **Round 9:** the committed D3 amendment named `.venv` (Python 3.13.13). The run used `.venv312`. The erratum naming `.venv312` was committed at 22:45, after pHRR, FIGRA and cr_round2 had finished. tabpfn was also installed at 22:15, about 1.5 minutes before D3 was committed. Both are label-free environment matters. Numerically they do not matter: every incumbent reproduced to about 1e-6 (section 3). It is still a retroactive environment record and should be described that way.
- **cr_round2.json** pins "python3 3.12.4 … If any version differs, stop BLOCKED". `train_cr_round2.py` checks only the four library versions and prints Python 3.12.12. Read literally, CR round 2 should have stopped BLOCKED. During this review, 954ad20 (01:52:54) added an erratum to cr_round2.json and tti_round3.json naming `.venv312` and saying the orchestrator "treats this patch-level interpreter difference as not triggering the version-mismatch BLOCKED clause". Its in-text time "~02:00" again postdates its own commit. My judgment: it **does not matter for any number**. The patch-level Python change does not touch numpy or sklearn numerics. exp117 reproduced exactly (0.650403 [0.438897, 0.789283]). The round-9 B2 CR incumbent OOF equals cr_round2's to 0.0. No CR record changes. It **does matter procedurally**: a declared stop rule was overridden by the executor at run time and ruled on afterwards. The erratum should say plainly that it was written after the run (the run was about 22:31-22:38).
- Threads: round 9, CR and TTI ran with OMP/torch/LOKY = 1. Round 10 ran with 3 (allowed: "<= 4"). The round-9 and round-10 incumbent OOFs differ by at most 7e-14. Immaterial.

## 3. Step-1 reproductions and fold identity (item 1)
Each record was reproduced in the same process as its candidates, within tolerance:
- exp074: 0.6113 [0.5155, 0.8376] (round 9). Round 10: 0.611289 [0.515501, 0.837643], within 0.001.
- exp117: 0.6504 [0.4389, 0.7893] in cr_round2, round 9 and round 10 (0.650403 [0.438897, 0.789283]).
- exp119: 0.4975 [0.3464, 0.6475] in rounds 9 and 10 (0.497507 [0.346372, 0.647530]).
- exp115: B 0.3875 [0.3124, 0.5289], A 0.7175, C 0.2475 (round 9). Round 10: 0.387494 [0.312381, 0.528881], C 0.2475.
- exp046: S681 0.3133 [0.0715, 0.4958]. My independent re-run gives 0.313336 [0.071457, 0.495782]. (The brief's "0.4962" is a typo; the value is 0.4958.)

Folds:
- Every candidate scored through `eval_protocol.repeated_group_kfold` uses the incumbent's groups and seeds; GroupKFold splits depend only on n and the groups.
- The custom loops carry checks that passed:
  - s1: per-fold `(oof+1)/s == incumbent oof + 1`, relative 1e-9.
  - n1: harness self-test, max |dR2| = 0.
  - TTI: `identity_check`, np.array_equal on S681 and S675. The TTI p2 OOF equals mean(exp046, p1) exactly (max diff 0).
- Each npz `oof_incumbent` is the same-run reproduction. The joint CR npz asserted y/groups equality and an incumbent OOF difference of 0.0.

## 4. Implementation vs config (item 2, spot checks)
- **TabPFN:**
  - Constructor `TabPFNRegressor.create_default_for_version(ModelVersion.V2, device='cpu', random_state=0)`, nothing else passed.
  - Env vars are set before import; HF_HUB_OFFLINE=1 in every scoring run; torch threads 1.
  - Fitted attributes 8 / cpu / False / None. Determinism bitwise, in-process and in a fresh process.
  - Input is the incumbent's float64 matrix with NaN kept. Scale: log1p for pHRR/FIGRA (fit on log1p, expm1 of the mean), raw otherwise.
  - p2 is `Avg([clone(incumbent), TabPFN])`: nested 1/2 weights, as declared.
- **Duan smearing (B3):** the code matches the config verbatim: outer fold model, inner `GroupKFold(5, shuffle, random_state=0)` on the training fold only, `s = mean(exp(z - inner))` unclipped, prediction `exp(m(x))·s − 1`. Fold factors: pHRR mean 1.142 (1.014-1.406), FIGRA 1.188 (1.104-1.291), none below 1.
- **CR o1:** base = `np.nanmean(X[:, [3,4,5]])`. Columns 3-5 are asserted to be sta_res_3K/10K/30K; coefficient 1; offset on HGB and ET only; Ridge on raw y; one final clip to [0, 100]; members not clipped. o2 is report-only and not selectable. n1 trains on the 136 extra rows (17 materials, asserted), drops the chair assembly when a linked layer is tested (24/50 folds), and never scores extra rows.
- **TTI:**
  - **S675** = S681 minus exactly the 6 config ids (details in section 9).
  - a0/a1 objective, gradient and start: Olsen form, alpha 0.01 on slopes, divisor n_ign, th ≥ 1e-8.
  - a2: QCAP 13.5, start q_cr 6.75, exponent −2, unbounded L-BFGS-B.
  - o1 offset uses `round(exp(log_q)/1000, 6)`, with `(q/50)^2` reweighting.
  - cap_fold = max training TTI on a0/a1/a2 only.
  - Self-tests T1-T4 passed. All 150 AFT fits converged.
  - a1 (−1.97) and a2 (−1.67) are catastrophic, but not because of a bug. sigma inflates from 0.88 (a0) to 1.32 (a1), so the lognormal mean `exp(mu+sigma²/2)` blows up and hits the cap (108 and 77 capped predictions). The config's honesty notes predicted this mechanism.
- **Round 10:**
  - pHRR o1: `ln(y) − ln(mcc_peak_hrr_Wg)` with exp074 HGB, coefficient 1, ln not log1p; peak asserted finite and > 0.
  - FGC: `Q/(T2−T1)·(T2−T0)/(T1−T0)`, T0 = 298 K, Q = max(mean total_hr_bc, 0)·1000. The PC/ABS self-test gives 280.1. T1 is asserted equal to exp074's `mcc_t_onset_C`.
  - GP: `ConstantKernel(1.0)*RBF(np.ones(d)) + WhiteKernel(1.0)`, normalize_y, n_restarts 0, random_state 0, imputer (median, indicators, keep_empty) then scaler. d = the imputer's fold-dependent width; ARD needs that, and it is noted in the flags.
  - CR g1 base: nanmean STA, with the in-fold median fallback (never triggered).
  - CR m1: HGB `monotonic_cst` +1 on the three STA columns and mcc_char_yield.
  - LOI m1: RF/ET behind the declared imputer with +1 on total_fr_wtpct and el_P, 0 on indicators. The `MonoImp` wrapper equals make_pipeline when unconstrained (self-test).
  - LOI s1: skrub `StringEncoder(8, random_state=0)` fit on training rows inside each fold (protocols B and C).
  - Report-only g1/m0 are in K and not selectable.
- Round-9 LOI was run from a venv-free copy of the repo, with the pre-patch driver, because `copy_tree` failed on `.venv` symlinks (flag 5). The only code difference is the `.venv*` skip. The prefix is `train_loi_rescore_r8.py` lines 9-637 verbatim (I diffed it). npz sha256 is identical to the kept scratch copy (`1e5989e7…`). The result JSONs are identical too.

## 5. Leakage (item 2)
None found. In every fold, everything fitted is fitted on training rows only:
- imputers, scalers and pipelines (cloned per fold)
- TabPFN preprocessors
- the inner smearing CV
- the GP base median
- StringEncoder
- AFT preprocessing
- cap_fold

Other checks:
- **TTI censored rows** are admitted only when their material is not a test material. This is asserted per fold, and the family-GKF version uses family labels.
- **n1 extra rows** are never predicted or scored.
- **New physics features** are material-level MCC aggregates and contain no label.
- `POST TEST CMT` is used only by the S675 row rule. It is never a feature.

## 6. Adoption gate v2 (item 3)
- `data/v2/adoption_gate.py` is the config `pseudo_code` line for line. The TTI worker's ordered whitespace-insensitive check passed, and I read it myself. Its self-test passes. The consistency checks (1e-9 against the `*_full` fields) pass for every family.
- I re-ran the gate on **all nine** npz files into `/tmp/claude-502/critic_*.json`. Every output JSON equals the recorded one exactly. My own implementation (own draws, own R², no shared code) reproduces q, best, theta, ci_lo and LB to 1e-6 for round-9 pHRR, round-10 LOI and TTI round 3.

| Family | K | q | Selected | theta | wins | ci_lo (inc) | LB | G1 G2 G3 |
|---|---|---|---|---|---|---|---|---|
| r9 pHRR [p1,p2,s1] | 3 | 2.013 | p1 | +0.0796 | 10 | 0.4813 (0.5155) | −0.0479 | T F F |
| r9 FIGRA [p1,p2,s1] | 3 | 1.753 | s1 | +0.0191 | 8 | 0.3326 (0.3464) | −0.0569 | T F F |
| r9 CR [o1,o2*,n1,p1,p2] | 5 | 1.910 | o1 | +0.0156 | 7 | 0.4349 (0.4389) | −0.0098 | F F F |
| r9 LOI [p1,p2] | 2 | 1.776 | p2 | +0.0106 | 7 | 0.3229 (0.3124) | −0.0123 | F T F |
| TTI [a0*,a1,o1,o2,a2,p1,p2] | 7 | 2.078 | p2 | +0.0272 | 7 | 0.2253 (0.0917) | −0.0175 | F T F |
| r10 pHRR [o1,g2,g1*,f1] | 4 | 1.732 | f1 | −0.0488 | 2 | 0.3745 (0.5155) | −0.1253 | F F F |
| r10 FIGRA [g1] | 1 | 1.552 | g1 | +0.0022 | 7 | 0.3523 (0.3464) | −0.0120 | F T F |
| r10 CR [g2,g1*,m1] | 3 | 1.832 | m1 | −0.0026 | 4 | 0.4093 (0.4389) | −0.0147 | F F F |
| r10 LOI [m1,m0*,s1] | 3 | 2.094 | m1 | +0.0149 | 10 | 0.3269 (0.3124) | −0.0055 | T T F |

\* report-only or control: in K, absent from `selectable` in the npz, never selected. Selection is max rkf_mean over selectable members, ties to the first listed. TTI R1 (a1 vs a0) is not triggered. Verdict "incumbent stays" is correct everywhere.

- Family composition and order match each config. TTI order a0, a1, o1, o2, a2, p1, p2 follows the D4d amendment, which overrides gate_v2's B2-before-B4 sort. That sort only matters for exact ties, and there are none.
- **Deviation: LOI rounds 9 and 10 should have been one family.** Both were scored against exp115 on the same 606 rows before either LOI decision: round-9 LOI ran from 22:30 to 01:04, and round 10's LOI gate ran at 23:23. gate_v2 `family` and `anti_gaming` ("everything scored … before that target's adoption decision is one family"; "one gate run per target per round") then require one K=5 run. Two runs were made (K=2 and K=3), and round10_physics.json declared the narrower family. As a critic diagnostic only (not a gate run), the joint K=5 family gives q 2.278, selected r10 m1, LB −0.0073, G3 fail. The outcome does not change: a superset family can only raise q. pHRR, FIGRA and CR were sequential: each round-9 gate ran before round 10 scored that target. For CR the margin was about 90 s, but the rule's letter is met.
- Cross-round multiplicity is not controlled: per-round families were a D1 choice, and round 10 overran the round budget under delegation. pHRR has now had rounds 1, promote, blend 1, round 9 (K=3) and round 10 (K=4) against exp074. Any future pHRR proposal must carry that history. Under the D1 amendment's re-proposal bar, every member of all nine families is barred from re-proposal against the same incumbent.

## 7. Diagnostics that round9 `after_round` requires in the critic brief (report-only)
- **Top-5 groups removed (every selected or "near-miss" candidate fails the established test):**

  | Candidate | Gain | Top-5 groups removed | Other |
  |---|---|---|---|
  | pHRR p1 | +0.0796 10/10 | −0.0221 (3/10) | category-GKF 0.553 vs 0.392 |
  | pHRR p2 | +0.0782 10/10 | +0.0089 (6/10) | category-GKF 0.497 |
  | FIGRA s1 | +0.0191 | −0.0778 (0/10) | |
  | FIGRA p2 | +0.0173 | −0.0222 (2/10) | |
  | CR o1 | +0.0156 | −0.0078 (3/10) | |
  | LOI r9 p2 | +0.0106 | −0.0080 (3/10) | A 0.768, C 0.317 (exp115: A 0.7175, C 0.2475) |
  | LOI r9 p1 | | | A 0.789, C 0.277 |
  | LOI r10 m1 | +0.0149 | −0.0021 (3/10) | C 0.253 |
  | TTI p2 | +0.0272 | −0.0102 (5/10) | |
- pHRR category-GKF: s1 0.258, r10 o1 0.499, f1 0.359 (exp074 0.392). These are report-only and never promote.
- s1 fold factors are in section 4. Smearing hurts pHRR (−0.072, 1/10) and is a non-established +0.019 for FIGRA.

## 8. Wording and honesty (item 6)
The result-JSON notes are clean:
- no "beats" anywhere
- no comparison with 0.58, 0.82, 0.84 or 0.9724 (the `sota` schema fields are not claims)
- TabPFN attribution is present
- "not established" appears wherever the pre-declared rule fires

Three wording problems need correcting:
- **"Near misses" overclaims.** The commit message ce34bb1 (and any STATUS or record text built from it) calls pHRR p1/p2, LOI m1 and TTI p2 near misses. All of them fail the top-5 test, and pHRR p1's sign reverses (−0.022). Write "gain not established (concentrated in ≤5 materials/papers; multiplicity-adjusted LB < 0)" instead.
- **"p1 +0.080 / p2 +0.078 (10/10 paired wins, G2/G3 fail)" conflates the two.** Only p1 was gated (G1 pass, G2 and G3 fail). p2 was not tested, because the rule has no fallback. Its own numbers would pass G2 (0.5677 ≥ 0.5155) and fail G3 (LB −0.0170). Do not imply that p2 failed G2, and do not imply that p2 is a fallback.
- **The rule's phrase misleads for losers.** "gain concentrated in <= 5 materials" is applied mechanically to candidates with negative overall gain (e.g. CR p1 −0.100). The rule requires it, but STATUS text should say "no gain" for those.
- The round-9 LOI result JSONs carry no gate annotation (p2 selected; G1 fail, G2 pass, G3 fail, LB −0.0123, K 2), unlike the other selected candidates. Put it in the round summary.

## 9. Pending record change: exp046 on canonical S675 (item 5). Verdict: SOUND WITH CAVEATS
- **The filter is verified.** I applied the config regex (POS and not NEG, case-insensitive, on `post_test_cmt`) to all 735 rows myself. It flags exactly HDPE|HF25|R3, HDPE|HF25|R4, Nylon|HF25-Frame|R4, Nylon|HF75|R1, PET|HF25|R2 and Overstuffed_Chair_Polyester_Fabric|HF25|R2. These are the config's `flagged_ids`; PET|HF25|R1 is not flagged. `v5.df` (681 rows) minus these 6 gives 675 rows / 76 materials. That equals the npz `row_id` in order, and y and groups match exactly.
- **The pipeline is exp046's.** It is `v5.linear(PoissonRegressor(alpha=0.01, max_iter=3000))` (median impute + indicators + scaler) on PHYS from `train_tti_v5_physics` with `--onset-bc --replicate-table`, scored by `eval_protocol.repeated_group_kfold` and `bootstrap_ci`. The only change is the 6 rows, removed from training and scoring in every fold, and never turned into censored rows. My independent re-score gives **0.317902 [0.091743, 0.506655]**. Its OOF is bitwise equal to the npz incumbent and to the `*_full` fields of `tti_r3_exp046_rows675.json`.
- **D4b was decided before any score**: 69ffda8 at 22:16, step 1 at about 22:30. The decision used the comment text plus the config rule. The config pre-declares that the re-scored number becomes the record whichever way it moves.
- Caveats that must go in the record note:
  1. S675 was named by the orchestrator under owner (Nikhilesh S) delegation, not by Arnav as the config drafted.
  2. The 6 rows' TTI values and their siblings' values were known when the regex was written (disclosed). The cleaning is one-sided: 7 late-recorded rows, biased high, stay.
  3. The +0.0046 is a data-QA change on a different row set, not a model gain and not a paired comparison. Keep exp046 0.3133 [0.0715, 0.4958] on 681 rows as history.
  4. Status stays "PARKED, SHORT OF SOTA", comparable stays "pending-human", and nothing is set against 0.58. TTI model rounds are now closed (`final_round`).

## 10. Caveats that must go into record and STATUS notes
1. All rounds: approvals D1-D5 and the round-10 budget overrun were orchestrator decisions under owner delegation (human-only decisions per the drafted configs).
2. No adoption anywhere. "Gain not established after multiplicity adjustment (LB = x, K = k)", using the table in section 6. Every positive gain fails the top-5 test (section 7).
3. LOI: rounds 9 and 10 formed one family under gate_v2. Quote the joint diagnostic (K 5, LB −0.0073) next to the K=3 figure −0.0055.
4. Environment: every number in these rounds comes from `.venv312` (Python 3.12.12, record library pins). It was recorded only after the runs: for round 9 in 284ed43 (after pHRR, FIGRA and CR round 2 had scored), and for cr_round2 and tti_round3 in 954ad20.
5. TTI S675 record: the four caveats in section 9.
6. A future pHRR proposal against exp074 must cite the accumulated looks (rounds 1, promote, blend 1, 9, 10).

## 11. What the orchestrator must fix
1. The cr_round2 and tti_round3 errata now exist (954ad20). Add one sentence to each: "recorded after the run (run 22:31-23:06); the BLOCKED clause was overridden at run time on an orchestrator instruction".
2. Correct the amendment timestamps that postdate their own commits: "22:45" in 69ffda8, "~23:20" in 284ed43, "~02:00" in 954ad20. Alternatively, add a line saying git commit times govern.
3. Record the LOI family deviation (section 6) in round10_physics.json amendments, or in the round summary, with the joint diagnostic.
4. Round-9 erratum: add that tabpfn was installed into `.venv312` at 22:15, before the D3 commit, and that pHRR, FIGRA, CR round 2 and the start of TTI ran before the erratum was committed.
5. `experiments/runs/tti_round3_flags.md` promises a "Run outcome" section that was never appended. Append it or delete the pointer.
6. Delete the kept LOI scratch copy `/var/folders/02/5t57twj10lqfwpkgwr2px9kc0000gp/T/dravix_rescore_expanded_csy60e_5` (15 MB). round9 requires this, and its npz is verified identical.
7. Replace "near misses" wording, and fix the p1/p2 G2 conflation in any STATUS, handoff or record text (section 8).
8. After the TTI record change:
   - give it a new exp id via `scripts/update_trackers.py`
   - set value 0.3179 and CI [0.0917, 0.5067]
   - re-run conformal_intervals.json's TTI part unchanged (that config says so)
   - make any new tti_s run (demo/predict.py TTI model, macfp_wood_crosslab TTI) use S675, or label it "681 rows, pre-D4 definition" (tti_round3 `scope_of_rule`).
