# Critic: TTI round 2 (emissivity; e1-e4 vs exp046; exp104-107)
Verdict: **SOUND WITH CAVEATS**. The round was declared before it ran, exp046 reproduces, the physics terms match the config, and the emissivity ingest matches FSRI's own script to within 0.002. "No gain, exp046 stays" is correct. The explanation "drop driven by the half-missing feature" is only half the story: on the 42 covered materials alone, emissivity also fails to help. Emissivity shows no detectable signal on this population.

## Pre-declaration and order
- Config a4e6113 at 18:18:35. Script d94d213 at 18:19:08 (mtime 18:19:05). Log and exp104-107 at 18:19:39 (28 s runtime, so the run started at about 18:19:11). OK.
- Authorisation: the config quotes Arnav's Oct 5 words. banned_moves.md lifts the thermal-join ban for emissivity_fsri.csv in this round only. One round, 4 candidates, as declared.
- emissivity_fsri.csv was committed on Oct 4 (b8c06dc) as data only, before this round existed. It is unchanged since.

## Implementation vs config
- The base is v5 imported with --onset-bc --replicate-table, and the 681 / 76 assert passes. exp046 reproduces: 0.3133, CI [0.071, 0.496].
- T_h = (q''/sigma)^(1/4). The code gives 815 / 969 / 1072 K. The config text says 818 / 973 / 1077 K: that arithmetic is slightly off, but the formula is implemented as declared. Immaterial.
- eps_h is np.interp over 600-2000 K at T_h (always >= 815 K, so no clamping).
- eps_s is emissivity_600K.
- q_cr_eps = eps_s·sigma·(T_ig^4 - T0^4) + 10·(T_ig - T0), using the same baseline-corrected onset and T0 = 20 °C as exp046.
- log_q_excess_eps = log(eps_h·q'' - q_cr_eps).
  - All as declared.
  - "NaN when <= 0" never fires (0 rows).
  - The 9 covered rows that lose the term (383 -> 374) have no MCC onset.
- e1 / e2 / e4 use v5.linear (median impute + add_indicator + scaler) with Poisson alpha 0.01. e3 is HGB poisson with v5.HGB on F3 + both terms. All as declared.
  - Because of add_indicator, **e1 already contains a missing-emissivity indicator, which is exactly 1 - has_emissivity**. So e1 = e4 + log_eps_h, and e1 - e4 isolates the emissivity value itself.
- Selection: max mean among e1-e3, the 8/10 rule, ci_lo >= 0.071, and the e4 control. Implemented as declared. e1 0.2833 gets 1/10 wins, so not adopted.
- The rule's "gain matched by e4" branch fires even though nothing gained. This does not change the outcome.
- Leakage: none new. Emissivity is a material-level bench property (room-temperature FTIR on virgin samples), joined on the exact material name.
  - The 17 emissivity materials that do not join (Laminate_Flooring, PEX_Pipe, ...) are absent from cone_replicates_fsri.csv altogether. This is not a spelling failure.

## Emissivity ingest vs FSRI's plot_IS_emissivity_data_html.py
- I ran FSRI's own script in scratch (plotly/git stubbed, file writes captured) on the same raw tree.
- It produces 59 materials, skipping the same 4 (MDF, OSB, two Overstuffed_Chair components).
- Emissivity at all 8 temperatures matches the ingest exactly at 3 d.p. for 58/59 materials. One material differs by 0.002. Absorption coefficients match for all 3 materials with transmission.
- **"Step for step" is not exact for emissivity_std_*.** FSRI computes `sample_df['Std'] = sample_df.std(axis=1)` after adding the 'Mean' column, so its std includes the mean. The ingest uses the replicates only.
  - The std columns differ by up to 0.022 (25 materials > 0.0015).
  - The ingest's version is arguably the right one, but it is not FSRI's.
  - Unused here. Either fix the docstring or note the difference in the CSV provenance.

## Is "no gain; drop driven by the half-missing feature" supported?
- "No gain": yes. e1 / e2 / e3 are -0.030 / -0.042 / -0.156 vs exp046, with 1/10 or fewer wins.
- "Driven by the half-missing feature": largely yes.
  - e4 (indicator only) loses -0.026. e1 loses -0.030, so the indicator accounts for about 85% of e1's drop.
  - A log-target Ridge shows the same thing (PHYS+has: -0.012).
- **But emissivity also fails where it is fully observed.** My checks:
  - Covered-only refit (383 rows / 42 materials, no imputation), same GKF 5x10.
    - The Poisson GLM is unstable on that subset (exp-link blow-ups, R2 << 0 for both arms), so I compared a log-target Ridge on PHYS vs PHYS + log_eps_h.
    - Log-space R2 goes 0.637 -> 0.617 (-0.020, 2/10 wins).
  - Material-level correlation between the mean log residual of exp046 OOF and log eps_h: +0.06 (n = 42). Physics predicts a negative correlation (higher absorptance, faster ignition).
  - Spread: log eps_h has sd 0.12 against 0.46 for log q''. Most materials sit at eps 0.8-0.95, so a multiplicative eps correction is small next to the 25 kW/m² blow-ups found in round 1.
- The log's "eps-rows" line (e1 0.552 vs exp046 0.535) comes from repeat-0 OOF of the full models, restricted to covered rows. It is not a covered-only fit, and the +0.017 does not survive the refit above.
- Correct wording: "Emissivity adds no detectable signal: no gain on the 42 covered materials either, and residual correlation about 0. In the full model the missing-emissivity indicator costs about 0.026, which explains most of e1's drop."

## Caveats to carry
1. exp046 stays: 0.313 [0.071, 0.496]. Keep TTI parked. The thermal-join ban returns for emissivity after this round, as banned_moves.md says.
2. Replace "drop driven by the half-missing feature" with the wording above. The finding is "no signal", not "signal hidden by missingness".
3. The ingest docstring's "reproduces step for step" is true for emissivity and absorption coefficients, but not for emissivity_std_* (FSRI includes the Mean column in its std).
4. Correct the config's T_h values (815 / 969 / 1072 K) if the config is cited.
