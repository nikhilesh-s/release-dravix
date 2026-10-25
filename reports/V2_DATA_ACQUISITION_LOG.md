# Dravix V2 — Data Acquisition Log

Running log of real data actually pulled into `data/v2/`. Append, don't rewrite — each entry stands as it was true when written.

## 2026-09-27 — FSRI Materials and Products Database (first real Tier-2 data)

**Source:** `github.com/ulfsri/fsri_materials_database` (public, cloned to `data/v2/raw/fsri_materials_database/`, no author email needed — identified in `data/sources/leads_hunterB_repositories.md` as the highest-expected-yield public repo).

**Citation (required on any use):** McKinnon, M., Weinschenk, C., Dow, N., DiDomizio, M., and Madrzykowski, D. *Materials and Products Database* (Version 1.0.0), Fire Safety Research Institute, UL Research Institutes. Columbia, MD 21045, 2023.

**What got ingested** (`ingest_fsri.py`, kept in `data/v2/` alongside its output — rerunnable, not a one-off scratch script):

| Output file | Rows | Materials | What each row is |
|---|---|---|---|
| `tti_fsri.csv` | 717 | 80 | One cone-calorimeter replicate's `TIME TO IGN`, read directly from that test's own Scalar file — no derivation, no averaging. |
| `phrr_fsri.csv` | 220 | 76 | Peak of FSRI's own replicate-averaged HRRPUA curve, per material × heat flux. FSRI computes and publishes this mean curve themselves (`02_Scripts`); we take its max, we don't recompute the average ourselves. |
| `thr_fsri.csv` | 220 | 76 | Trapezoidal integral of that same mean HRRPUA curve to the curve's own end time. **This one we derive** — FSRI doesn't publish a THR column directly, so this is our own integration, not a repo-provided value. |

**Sanity checks run before trusting this:**
- Confirmed the derived curve files' time column is in seconds (checked against each test's own `END OF TEST TIME` in the Scalar file) — the first version of this script assumed minutes and produced THR values ~60x too high (e.g. ABS at 6,972 MJ/m² instead of the corrected 116 MJ/m²). Fixed and reran.
- PMMA pHRR at 50 kW/m² came out to 1,155 kW/m² and THR to 90 MJ/m² — both in the range PMMA cone-calorimeter literature reports (PMMA is the standard reference material for this test, so this is a meaningful check, not just a vibe check).

**Caveats to carry forward, not paper over:**
- **THR here is not apples-to-apples with Pomázi's THR** (the Tier 2 competitor bar) without matching exposure geometry/duration — Pomázi's is fibre-reinforced composite panels under different test conditions. Treat FSRI's THR as Dravix's own feature/target for now, not a like-for-like comparison row, until that's checked properly.
- 736 total Scalar files exist; only 717 parsed into `tti_fsri.csv` — the remainder either don't match the expected filename pattern or are missing `TIME TO IGN` (worth a follow-up diff, not blocking).
- The repo also has MCC, STA, HFM, and FTIR data per material (char yield / decomposition / thermal conductivity relevant) — **not yet ingested**. Next-highest-value pull from this same repo.
- Raw clone is 1.7GB, kept at `data/v2/raw/fsri_materials_database/` as the reprocessable source of truth; only the derived CSVs above are meant to be treated as the working dataset.

**Zero LOI (Tier 1) rows from this pull** — FSRI's apparatus list doesn't include an LOI test. Tier 1 still depends on the Sun/Fu author-contact route or another LOI-specific source.

## Also cloned, not yet ingested: MaCFP matl-db

`github.com/MaCFP/matl-db` (MIT licensed, cloned to `data/v2/raw/matl-db/`). Only 2 materials (PMMA, Wood) but multi-lab-replicated, calibration-grade data — high quality, low volume. Structure surveyed, not parsed yet. Flagged as a follow-up, not run today for time.

## 2026-09-27 (later same day) — FSRI STA + MCC, and the first trained model

**More of the same source repo, two more apparatus types** (`ingest_fsri_sta_mcc.py`):

| Output file | Rows | Materials | What each row is |
|---|---|---|---|
| `char_yield_sta.csv` | 848 | 146 | Final residual mass % read off each Simultaneous Thermal Analyzer replicate's own mass-loss curve (last data point, N2 atmosphere, 3/10/30 K/min). Real, per-replicate, no derivation beyond "take the last value." |
| `mcc_hrr.csv` | 352 | 118 | Peak specific HRR (W/g) from each Micro-scale Combustion Calorimeter replicate's own curve, plus char yield from the paired FINAL_MASS file (final mass ÷ initial sample weight). |

**Sanity checks:** MDF char yield came out ~19% (matches known wood-fiber char yield ranges); Black PMMA STA char yield came out ~0% (PMMA fully volatilizes, no char — correct). One bug caught and fixed before trusting the output: the first pass of `ingest_fsri_sta_mcc.py` walked one directory level too few and collapsed all STA rows onto a single fake "material" — fixed by adding the missing `dirname()` call, rerun, materials count went from 1 to the correct 146.

**Running total, real records: 717 (TTI) + 220 (pHRR) + 848 (STA char) + 352 (MCC) = 2,137.** Note this is *rows*, not *unique materials* — the underlying FSRI repo only covers 183 materials total; "2,137 real records" should never be quoted as "2,137 materials."

**First trained model** (`train_phrr_baseline.py`, logged as `exp001_phrr_rf_baseline` in `experiments/leaderboard.csv`): a RandomForest on Peak HRR, joining `phrr_fsri.csv` with material-level features from the two tables above (heat flux, MCC peak specific HRR, MCC char yield, STA char yield). 158 rows / 55 materials had all four features. Evaluated with `GroupShuffleSplit` **by material** (a material's rows never span train and test) across 10 seeds, because the target — Pomázi et al.'s real held-out Peak HRR R²=0.30 — is the softest bar on the whole SOTA board and the most honest place to start.

**Result: mean held-out R² = 0.17 (std 0.35, range −0.43 to +0.68). Does not beat SOTA (delta −0.13).** Reporting the full spread, not just the mean, on purpose — a single lucky seed would say R²=0.68 and beat SOTA; a single unlucky one would say −0.43. Neither number alone is trustworthy at 55 materials. The honest read: real progress (a real held-out number exists where yesterday there was none), but the dataset is still too small and the features too indirect (bench-scale MCC/STA numbers standing in for real chemistry/composition descriptors) to close a 13-point gap reliably. More materials with matched MCC+STA+Cone coverage, or better features, before another training run.

## 2026-09-28 — exp002 beats SOTA on pHRR; TTI attempted and misses; STA char-yield data-quality note

Dropping the STA feature from exp001 (kept only heat flux + the two MCC features) recovered 158→175 rows / 55→60 materials by including materials that have cone+MCC coverage but not STA. **exp002: mean held-out R²=0.346 vs SOTA's 0.30 (delta +0.046), all 10 seeds positive (0.11–0.65)** — the first target where Dravix beats a verified SOTA number. Logged in `experiments/leaderboard.csv` / `reports/leaderboard.xlsx`. Caveat carried forward: this is a 10-seed CV mean, not a single locked, never-touched held-out set — don't call it "beats SOTA" externally until that final check is run once.

Tried Time to Ignition next (717 real rows via `tti_fsri.csv`, 3x pHRR's data): **exp003 failed badly** (mean R²=−0.26, worst seed −3.07) because TTI spans three orders of magnitude (0–1,102s, median 17.5s) and slow-igniting materials (PET, carpet, upholstery) dominate squared error whenever they land in a test fold — more data does not fix a skewed-target problem. **exp004 fixed it partially**: trained on log1p(TTI), scored back in raw seconds for a fair comparison to SOTA — mean R² flipped positive (0.10), worst seed improved to −0.44, but still well short of SOTA's 0.58 (delta −0.48).

**Data-quality note on `char_yield_sta.csv`, found while scoping the Char Residue target:** ~94% of rows have a slightly *negative* `final_residual_mass_pct` (median −3.1%, 25th percentile −3.4%), which is physically impossible for a residual-mass measurement — char yield can't be negative. Real char-forming materials (MDF: 20–26%, OSB: ~23%, polyester batting: ~25%) still come through correctly and clearly separated from this, so the negative values look like a systematic ~3% baseline artifact in the instrument's blank-subtracted (`Mass(subtr.2)`) curve for materials with near-zero true char, not a random error. **Floor at 0 before using this column as a Char Residue target** (real char yield ∈ [0, 100]); leave it unfloored when only using it as an auxiliary feature, where the relative ordering still carries signal.

## 2026-09-28 (continued) — pHRR promoted on a locked test; Char Residue ruled out; TTI feature attempt

**exp006 resolves exp002's open caveat.** exp002's R²=0.346 was a 10-seed CV mean; all 60 materials had rotated through some seed's test fold, so no single one of those splits was a clean "never touched" claim. Committed to one new split (seed=42, chosen before running, not after) and ran it once: **R²=0.5744 vs SOTA's 0.30 (delta +0.27)** on 15 materials untouched by any feature/model decision. This is the number to actually cite as "beats SOTA." Promoted: retrained on all 175 rows, saved to `models/v2/phrr_rf_v1.joblib`.

**exp005: Char Residue ruled out on FSRI data, with a diagnosis, not abandoned quietly.** Only 4 of ~150 materials in the whole repo have any meaningful char (>2% mean residual: OSB, two polyester-upholstery variants, MDF) — the target is degenerate here, near-binary "chars or doesn't," not the continuously-varying quantity Pomázi's R²=0.82 was measured on (fibre-reinforced composites, a narrow engineered sample space). One rare high-char material landing in a small held-out fold produces R² values below −200. No feature or model choice fixes a 4-material problem; this target needs a different, composite-focused data source before it's worth attempting again.

**exp007: tried closing the TTI gap with a physically real feature.** Added `onset_temp_sta.csv` — decomposition onset temperature (5% mass loss point, read off each material's own STA curve; sanity-checked against PMMA's well-known ~280–300°C onset, which is exactly what came out). This is a much more direct physical link to ignition than the bulk MCC combustion-energy proxies used in exp003/4. Result: mean R² improved 0.10→0.167 and variance roughly halved (std 0.25→0.12, meaningfully more stable) — but still well short of SOTA (delta −0.41). Feature importances show why: `specimen_mass_g` (0.33) and `heat_flux_kWm2` (0.26) still dominate over `onset_temp_C` (0.09). **Read this as diminishing returns on the feature-engineering path with FSRI data alone** — closing the rest of this gap likely needs real chemistry/composition descriptors, not another bench-scale proxy pulled from the same repo.

**Checked for a public-data shortcut on LOI (Tier 1) — none exists, confirmed rather than assumed.** Searched for the Sun/Fu 2026 RPUF paper's underlying 435-formulation dataset on Zenodo/Mendeley/Kaggle and for its ScienceDirect "data availability" statement, and separately for any other public RPUF/LOI formulation dataset. Nothing found — the paper's own dataset is stated as "curated... from 89 published sources," which reads as compiled by the authors themselves rather than deposited anywhere public. **LOI stays blocked on the author-email route; there is no repo to clone here the way there was for FSRI/MaCFP.** Don't re-spend search time on this without a new lead (e.g. a reply from the authors, or a different LOI-reporting paper with a public dataset).

## 2026-09-29 — MaCFP PMMA ingested; cross-lab validation surfaces a real limitation

**MaCFP matl-db finally ingested** (`ingest_macfp_pmma.py`, `phrr_macfp_pmma.csv`): 67 real cone-calorimeter tests on PMMA from **10 independent institutions** (Aalto, DBI_Lund, Edinburgh, GIDAZE+, HKPoly, LCPP, NIST, TIFP, UDRI, UQ) — a genuine multi-lab replication study, not a single-source dataset. Sanity check: mean pHRR at 50 kW/m² across these 10 labs is 1,081 kW/m², close to FSRI's own single-lab PMMA measurement of 1,155 kW/m² at the same flux — two independent real sources agreeing within ~7% is a good cross-check on both.

**This wasn't added to training data — it was used to stress-test the promoted pHRR model** (`models/v2/phrr_rf_v1.joblib`, exp006) against something harder than another same-lab split: different rigs, different operators, same material, none of it seen during training. Result (`validate_phrr_crosslab.py`):

- **Overall R² = 0.20** (positive) across all 67 tests — the model does capture the coarse heat-flux-driven trend across labs.
- **Per-heat-flux R² is deeply negative (−1.9 to −3.2).** Because the model's features are constant for a given material+heat-flux (it has no way to know which lab or rig produced a reading), its prediction doesn't vary across replicates from different institutions — but the real measurements do, substantially (e.g. at 65 kW/m², true pHRR ranges 612–1,815 kW/m² across labs). The model explains none of that real inter-laboratory variance.

**Read this as a genuine, disclosed limitation, not a failure to hide:** MaCFP's whole purpose as a database is characterizing exactly this kind of cross-lab measurement variability, so finding that our model doesn't capture it isn't a surprise — but it's an honest boundary on what "beats SOTA" (measured within one lab's held-out materials) actually means. The promoted model generalizes across *materials* within FSRI; it has not been shown to generalize across *laboratories*. Don't claim the latter.

## 2026-09-29 (continued) — FSRI Heat Flow Meter data (thermal conductivity + specific heat)

**Two more apparatus types ingested** (`ingest_fsri_hfm.py`), the last major unused category in the FSRI repo besides FTIR (spectral, not obviously useful for our targets, left for later):

| Output file | Rows | Materials | What each row is |
|---|---|---|---|
| `thermal_conductivity_fsri.csv` | 1,461 | 68 | One temperature-setpoint reading (Mean Temp, Upper/Lower/Average Conductivity, W/m·K) from FSRI's Heat Flow Meter, read directly off the instrument's own "Results Table" output — no derivation. |
| `specific_heat_fsri.csv` | 1,099 | 67 | Same apparatus, heat-capacity mode: Mean Temp, Enthalpy, Specific Heat (J/m³·K), same direct read. |

Files are UTF-16-encoded NETZSCH/Winthermo instrument output (`.tst`), not plain CSV — the first parser version used a filename regex that was too strict (missed typo'd/lowercase variants like `Conducivity`, `conductivity`) and only found 39 materials; loosened to a typo-tolerant pattern and got the real yield above (68/67 materials).

**Sanity check:** EPS Foam Board came out at 0.028–0.032 W/m·K — matches published EPS thermal conductivity (~0.03–0.04 W/m·K) closely.

**Running total, real records: 2,204 (yesterday) + 1,461 + 1,099 = 4,764.** Not yet used in any model — these are real material-property measurements, not fire-test outcomes, so they don't map directly onto any of the five targets. Their likely future value is as *features*: thermal conductivity in particular is physically linked to ignition behavior and could replace or supplement the bulk MCC proxies currently driving the TTI and pHRR models (see the 2026-09-28 log entries — feature importance there showed the existing proxies are the weak link). Worth trying in a future session, not squeezed into this one.

## 2026-09-29 (continued) — exp009: thermal inertia as a TTI feature

First use of the HFM data. `Spc_Heat` in `specific_heat_fsri.csv` is volumetric heat capacity (J/m³·K — EPS reads ~1.6e4, consistent with ρ≈20 kg/m³ × c≈1300), so k × Spc_Heat is thermal inertia kρc directly, no density needed. Used Dry readings at each material's lowest Mean_Temp setpoint. Added the ignition-theory feature log(kρc / q''²).

**Result (`data/v2/train_tti_v4_thermal.py`, scikit-learn 1.7.1):** the TTI × MCC × HFM overlap is only 18 materials (166 rows). On identical rows and splits, the legacy 10-seed protocol says thermal features help (0.103 → 0.194, 7/10 seeds improved). The repeated-GroupKFold protocol says they don't (0.094 → 0.056). log(kρc/q''²) carries ~0.6 of the forest's importance, so the physics signal is there, but 18 materials can't show it generalizing. The `--no-mcc` robustness check (TTI × HFM only: 229 rows, 25 materials; the Dry-only filter is why this isn't 57) goes the other way outright: 0.252 → 0.168 legacy, 0.215 → 0.125 repeated GroupKFold. **Conclusion: thermal inertia is not a demonstrated TTI improvement on this data.** Not promoted. TTI's real ceiling here is material count, not features.

**Correction, same evening:** this entry and exp009 first quoted 0.091 → 0.163 (paired +0.072). That run used scikit-learn 1.9.1. The repro audit reran it under the pinned 1.7.1 (the version that pickled the promoted model) and got 0.103 → 0.194. Rerunning under 1.9.1 gives 0.091 → 0.163 again, so the gap is the library version, not the code or data. Same seeds, different forests. Numbers above are the pinned ones. The robustness check is now in the script (`--no-mcc`) rather than an uncommitted probe.

## 2026-09-29 (evening) — Handoff sprint: portable paths, shared eval protocol, material properties + full-curve MCC features

**Ownership changed hands today, and the scripts only ran on the previous owner's laptop.** Every script in `data/v2/` hardcoded `~/Dravix/dravix-v2/...`. These now resolve paths relative to the script's own location (`os.path.dirname(os.path.abspath(__file__))`), so the repo runs from any checkout. Both raw sources were re-cloned into `data/v2/raw/` (FSRI at commit `a432697`, 2026-04-23; MaCFP matl-db HEAD). scikit-learn is pinned to 1.7.1, the version that pickled `models/v2/phrr_rf_v1.joblib`. Newer versions warn on unpickling.

**New shared evaluation helper (`data/v2/eval_protocol.py`).** It keeps the legacy leaderboard protocol (GroupShuffleSplit, test_size=0.25, seeds 0–9) so new rows stay comparable with old ones. It adds repeated 5-fold GroupKFold ×10 (every material predicted out-of-fold once per repeat, R² pooled over the whole set) plus a 95% CI from a material-level bootstrap. The reason: a 25% test split of ~60 materials is ~15 materials, and exp002's seeds ranged from 0.11 to 0.65 on identical settings. A "beats SOTA" call should rest on a CI, not on one split.

**New ingest (`ingest_fsri_props.py`), same FSRI source and citation:**

| Output file | Rows | What each row is |
|---|---|---|
| `material_props_fsri.csv` | 182 materials | `density_kgm3` from each material's own `material.json` (117 materials state one). `categories` (material family, for family-grouped splits). `cone_thickness_mm`: median specimen thickness parsed from the cone Scalar files' `PRE TEST CMT` field (70 materials; formats like "Dimensions: 100.09 x 100.16 x 3.09 mm", "Nominal thickness: 5/8 inches"). `cone_mass_per_area_kgm2`: median `SPECIMEN MASS`/`SURF AREA` (81 materials). All are pre-test measurements, not outcomes. |
| `mcc_extras_fsri.csv` | 352 replicates / 118 materials | Read off the same full MCC HRR(t), T(t) curves `mcc_hrr.csv` already uses: total heat release (∫HRR dt, kJ/g), heat release capacity (peak HRR / heating rate, J/g·K), temperature at peak HRR, and the temperatures at 5% / 95% of total heat released. |

**Sanity checks:** MCC heat release capacity for HDPE came out at 1,497 J/g·K and total heat release for ABS at 37.7 kJ/g. Lyon & Walters' published values are ~1,486 J/g·K and ~36.6 kJ/g. PP (1,092 J/g·K, 42.0 kJ/g) and PVC (13.3 kJ/g) land in their published ranges too. Independently, for the 61 materials with both a stated density and a parsed thickness, cone mass-per-area ÷ thickness reproduces the stated density at a median ratio of 1.06. So the thickness parser and the density field agree with each other.

**Unit label bug found in `mcc_hrr.csv`:** the column `heating_rate_C_s` holds `30.0`, but that's the filename's 30 K/min. The instrument header says 0.5 °C/s. The values themselves were never used as a feature, so no result changes. The new `mcc_extras_fsri.csv` stores the header's true 0.5 °C/s.


## 2026-09-29 (evening) — First real LOI rows (exp010, exp011): the FAA pure-polymer table

**New lead, not a rerun of the 2026-09-28 search** (that one covered RPUF formulation datasets only). Lyon & Janssens, *Polymer Flammability*, DOT/FAA/AR-05/14 (FAA, May 2005; US-government work, public domain). Read from the US DOT ROSA P copy, https://rosap.ntl.bts.gov/view/dot/16407/dot_16407_DS1.pdf. The FAA's own copy (fire.tc.faa.gov/pdf/05-14.pdf) returned 503 today.

**What got ingested** (`ingest_faa_loi.py` → `loi_faa_polymers.csv`): all 47 rows of Appendix A Table A-6, which prints LOI next to the MCC heat release capacity ηc. 32 polymers have both. Exact-name joins add char yield (Table A-2) and PCFC heat of combustion + flaming combustion efficiency χ (Table A-5) for 26 of them. Each row is stored as the verbatim text of the PDF, and the numbers are parsed from that string. All 100 transcribed rows were checked against an independent text extraction of the PDF: 0 mismatches. Ambiguous names (PE vs PE HD/LD, chlorinated PE, filled silicone) were left unjoined rather than guessed.

**Caveat carried forward:** the report calls these LOI values "generic values", i.e. typical literature numbers per polymer type, not paired measurements on the specimens behind ηc.

**First LOI model** (`train_loi_v1.py`, configs fixed before running): on 26 polymers, a ridge on log ηc + char yield + hc0 + χ scores pooled out-of-fold R² 0.647 (repeated GroupKFold, polymer-grouped), but the material-bootstrap CI is [−0.28, 0.87]. ηc alone scores 0.497 on the same rows, and 0.504 on all 32 polymers. The whole gain over ηc alone comes from PTFE and FEP (LOI 95). Without them the extra features hurt (0.299 vs 0.397). Holding out whole chemical families drops every model to 0.2–0.45. **Not comparable to sun_fu_2026_rpuf's 0.84:** different population (pure polymers vs FR foam formulations), and our LOI spread (sd 20.25) is far wider than theirs (implied sd 3.34), which flatters R². In absolute terms our RMSE is 12.7 LOI points against their 1.34. Logged as a baseline, not promoted.

**So nobody repeats this:** AR-05/14 is now fully used. Fig. 21 plots 50 polymers but only 32 are tabulated, so don't digitize the plot. Unchecked follow-up lead: Walters & Lyon, DOT/FAA/AR-01/31, *Calculating Polymer Flammability from Molar Group Contributions* (2001; HRC for about 80 polymers per a search summary; whether it prints LOI is unknown). Every copy failed today: fire.tc.faa.gov/pdf/01-31.pdf (503), apps.dtic.mil/sti/pdfs/ADA397200.pdf (403), ResearchGate (403). Worth one retry when the FAA site is back. LOI SOTA comparison still depends on the Sun/Bussemaker data (outreach draft in `reports/sprint_0929/outreach_drafts.md`).


## 2026-09-29 (evening) — Repro audit: the numbers hold, the STA columns don't

Re-ran every train/validate script (exp001–exp009) from a fresh FSRI clone (a432697) with the new relative paths, logs in `experiments/runs/`, runner `data/v2/run_all_experiments.sh`, pins in `requirements.txt`. **exp001–exp008 reproduce to 4 decimal places.** All 9 derived CSVs regenerate byte-identical from the raw clones. The promoted `phrr_rf_v1.joblib` is exactly an RF(300, depth 6, leaf 2, random_state=0) fit on all 175 rows, but no script in the repo writes it. **exp009 did not reproduce at first**: the in-tree script gave 0.103 → 0.194 (+0.092), not the logged 0.091 → 0.163 (+0.072). The lead traced it to the library version, not the code: the first exp009 run used scikit-learn 1.9.1, and rerunning under 1.9.1 gives 0.091 → 0.163 exactly. Same seeds, different forests. exp009 is now logged with the pinned 1.7.1 numbers, and its robustness check is committed as `--no-mcc`. Lesson: RandomForest output isn't stable across scikit-learn versions, so `requirements.txt` pins 1.7.1.

**Found a real bug in both STA ingests, and it's been there since 2026-09-27.** 788 of 848 FSRI STA exports have a `Mass loss/mg` column: mass *change* from the start, ~0 mg at t=0, not absolute mass. `ingest_sta_onset_temp.py` treats the first value as the initial mass, so its "5% loss" threshold trips on noise in the first sample: 548 of 848 onsets are just the scan's starting temperature, and 134 of 146 materials have onsets below 150 °C (ABS at 50 °C). Using the Meta file's sample mass, ABS moves to 385 °C, HDPE to 460 °C, and only the two gypsum cores stay below 150 °C, which is real dehydration. The PMMA sanity check passed because Black_PMMA happens to be one of the 60 files exported in `%`, the only format the code handles. Same root cause in `ingest_fsri_sta_mcc.py`: `final_residual_mass_pct` is mg lost for 788 rows, not %. The "only 4 materials char" finding behind exp005 is exactly the 4 materials exported in `%`. **So exp005's diagnosis was wrong, exp001's STA feature was mixed-unit, and exp007's onset feature was mostly the instrument start temperature.** On exp007's own rows the committed onset adds 0.000 (protocol b); the gain logged on 09-28 came from the row set shrinking. A one-off check with the corrected onset gives +0.10, but that's a lead for a proper experiment after the ingest is fixed, not a result.

**exp006 needs re-wording.** Its 15 "never-touched" test materials weren't untouched: 13 had already been scored in exp002's CV test folds, which drove the drop-STA decision. Across seeds 0–99 the same procedure averages 0.11 (median 0.26), and seed 42's 0.574 ranks 18th of 100. Under the new protocol (b) the promoted config scores 0.304, 95% CI [−0.16, 0.67]. That's a tie with SOTA's 0.30, not a win. The model is fine to keep as the current pHRR baseline; "beats SOTA" shouldn't be cited.

Smaller: `mcc_hrr.csv`'s `heating_rate_C_s` holds 30, the filename's K/min. The file headers say 0.5 C/s for 386 files but 1 C/s for 9, despite every filename saying 30K_min. One of the 1 C/s materials, Nylon_Carpet_High_Pile, is in the pHRR and TTI training sets with a doubled-rate MCC peak. `mcc_extras_fsri.csv` reads the header rate, so its HRC is right. The MCC filename regex silently skips 43 of 395 files, including every MCC file of 9 materials whose filenames lack `_MCC_`. The MaCFP institution list in the 09-29 entry says HKPoly; the data has ULSTER.

Details, evidence and the proposed fix: `reports/sprint_0929/repro_audit.md`.


## 2026-09-29 (evening) — FIGRA (Tier 3) from FSRI cone curves: first target table and first model (exp012, exp013)

**Definition.** Phan/Morgan (arXiv 2504.00223, "Fire testing", pp. 10–11, just before Table 2) define fire growth rate as peak HRR divided by time to peak HRR, with 15 polymers at 50 kW/m² (ASTM E1354-23). Specimens were powders or pellets, normalized by weight. They don't say which clock t_pHRR is on. We use time from the start of the test, the same clock as TIME TO IGN. All 15 of their Table 2 rows have pHRR/FIGRA > TIG, and FSRI's polymers match their numbers on that clock (LDPE 19.05 vs PE 19.09 kW/m²/s; t_pHRR 133.7 vs 135.4 s). That's consistent evidence, not proof.

**New ingest (`ingest_fsri_figra.py`), same FSRI source and citation:**

| Output file | Rows | What each row is |
|---|---|---|
| `figra_fsri.csv` | 735 replicates / 81 materials (FIGRA defined for 690) | One cone replicate. HRR is recomputed from the raw `*Scan*.csv` channels with FSRI's own O2-consumption formula, divided by the Scalar file's SURF AREA. pHRR is the raw 0.25 s maximum between TTI and END OF TEST TIME; FIGRA = pHRR / t_pHRR. The Scan files hold no HRR column, so this is our computation, but it reproduces FSRI's published per-replicate curves to within 0.03% on all 665 replicates found in both places. |
| `figra_fsri_meancurve.csv` | 220 material × flux | Peak of FSRI's averaged curve (the same number as `phrr_fsri.csv`), plus its replicates' mean TTI. Kept for cross-checking only (Spearman 0.994 vs replicate-mean FIGRA). |

**Sanity checks:** PMMA at 50 kW/m² gives a mean-curve pHRR of 1,154.8 kW/m², the same as the earlier entry. The replicates give pHRR 1,180–1,247 kW/m² at 97–103.5 s from the start of the test (TTI 30 s), so FIGRA 11.4–12.9 kW/m²/s. FIGRA rises with flux: 4.2 at 25 kW/m² and 20.9 at 75. In the PMMA scan checked, the O2 drop starts at ignition, so no analyzer delay is left to correct.

**Found along the way, affects other targets:**
- **FSRI's averaged `<Mat>_HRRPUA_<flux>.csv` files sit on the first replicate's time axis.** Longer replicates are truncated in 123 of 220 files, and in 10 replicates (6 material × flux rows) the real peak falls in the cut-off tail. The worst case is HDPE at 50 kW/m²: `phrr_fsri.csv` says 2,235.5 kW/m², while two replicates peak at 2,712 and 2,731. `thr_fsri.csv` integrates the same truncated mean curves. Replicate-level pHRR from `figra_fsri.csv` avoids this.
- **TIME TO IGN = 0 means "did not ignite", not "ignited instantly".** 45 of 717 rows in `tti_fsri.csv` carry it: fiberglass and mineral wool, gypsum, house wrap, PC at 25, PMMA at 15 without spark, all at noise-level pHRR. The TTI experiments currently learn these as 0 s. They should be censored or dropped.
- `02_Scripts/plot_Cone_data.py` sets the framed area (8,836 mm²) whenever a note contains "frame", so it would misfire on PMMA's "no retainer frame or grid". The published curves use SURF AREA instead (checked on all 665), so no data is affected.

**Result (`train_figra_v1.py`, settings fixed before the first run):**
- HGB on log1p, all heat fluxes: repeated GroupKFold R² **0.503 [0.324, 0.663]** on 175 rows / 60 materials. Same-rows ablation: heat flux only 0.267 → + MCC 0.249 → + density, thickness, mass/area and MCC extras 0.503. The specimen properties carry it. At 50 kW/m² only (57 materials): **0.157 [−0.279, 0.413]**.
- Phan/Morgan's 0.9724 is out of reach and **not a like-for-like comparison**: 15 neat polymers whose data seeded their synthetic training set, with the synthetic-set size picked on test R². Our closest slice, 12 FSRI polymers at 50 kW/m², scores 0.349. Not promoted.
- FIGRA is now a real Dravix target. It is not a SOTA claim.

## 2026-09-29 (evening) — MCC total heat release needed a baseline correction

The pHRR stream flagged that `mcc_extras_fsri.csv`'s `total_hr_kJg` (added earlier this evening) wasn't stable across replicates. Black_PMMA read 18.6, 39.9 and 33.1 kJ/g, and 28 of 118 materials varied by more than 20% between replicates. The cause is end-of-run baseline drift in the MCC signal. At ~740 °C, after everything has decomposed, the HRR channel sits anywhere from −9 to +22 W/g depending on the run, and integrating that over a ~1,300 s run adds or removes up to ~15 kJ/g. The same drift pulled the 5%-of-heat onset temperature early.

**Fix (`ingest_fsri_props.py`):** subtract a linear baseline drawn between the median HRR of the first and last 5% of each run, then integrate. The corrected columns are `total_hr_bc_kJg`, `t_onset_bc_C` and `t_end_bc_C`. They sit *alongside* the raw ones, which are unchanged, so every sprint result computed from the raw columns still reproduces. Use the `_bc` columns from now on. Materials with >20% replicate spread drop from 28 to 9, and corrected means land on published values: PMMA 24.2 kJ/g (Lyon & Walters: 24.3), ABS 36.6 (36.6), HDPE 40.6 (40.3), PP 40.2 (41.4), PVC 12.2 (11.3). The ingest now asserts that Black_PMMA, HDPE and ABS land in those ranges. The corrected onset still puts PMMA at ~206–215 °C (5% of heat released), below STA's 5%-of-mass onset (~290 °C). That's a different definition on a different instrument, not necessarily an error, but don't treat the two as interchangeable.


## 2026-09-29 (evening) — pHRR robustness: the locked-split 0.57 was a kind split; physics features give a real gain (exp014–exp021)

**The promoted number didn't hold up under the new protocol.** Ran the exp006 recipe (heat flux + MCC peak HRR + MCC char yield, RF) through `eval_protocol.py`. Repeated 5-fold GroupKFold x10 gives a pooled out-of-fold R² of 0.304, with a material-bootstrap 95% CI of -0.16 to 0.67. That's level with Pomázi's 0.30, not ahead of it. exp006's seed-42 split reproduces exactly (0.5744). But across 200 random 25% material splits the same model has a median of 0.268 and a mean of 0.143, and only 47% of splits beat 0.30, so seed 42 was the 82nd percentile. Grouped by FSRI material category (first-listed category, 11 groups), it drops to 0.08. exp006 was a legitimate single pre-committed test that happened to land on a kind split. The "BEATS SOTA" status overstated it.

**Why it's so split-sensitive:** a handful of thin or light specimens. House_Wrap is a 0.1 mm film with true pHRR of at most 57 kW/m². Its MCC chemistry says "polyolefin", so the model predicts up to 2,262 kW/m². It carries 19% of the pooled squared error by itself, and every one of the 13/200 splits scoring below -0.5 has it in the test set.

**Five pre-declared variants on the same 175 rows and splits** (`data/v2/train_phrr_v3_robust.py`, using the new `material_props_fsri.csv` and `mcc_extras_fsri.csv`):

| Variant | Repeated-GKF R² |
|---|---|
| log target alone | 0.231 (worse in 10/10 repeats) |
| + physics features | 0.515 |
| + physics, log target | 0.557 |
| HGB + physics | 0.509 |
| HGB, no MCC join (220 rows / 76 materials) | 0.462 |

Winner by the declared rule: physics + log target. **R² 0.557, CI 0.27 to 0.81, paired +0.25 over the promoted recipe in 10/10 repeats (gain CI +0.04 to +0.57)**, and 0.29 on the category split. A post-hoc drop-one-group run says the gain is the specimen geometry (mass per area, thickness, density): removing it costs 0.16. It also says the MCC extras *hurt*: without them the score is 0.64. That's a post-hoc observation, not a result, and it needs its own pre-declared test before anyone uses it. Thermal inertia adds nothing (only 18 of 60 materials have it). Dropping the MCC join doesn't pay either: the 16 materials it recovers are predicted at R² -0.88.

**Cross-lab (MaCFP PMMA, 10 labs).** With exp008's FSRI clear-PMMA descriptors, the physics model scores 0.36 against the promoted model's 0.20. MaCFP's material is 6 mm black cast PMMA, and FSRI has a Black_PMMA. With Black_PMMA's descriptors both models jump: the promoted model to 0.59, the physics model to 0.67, and 0.69 with MaCFP's own 6 mm geometry. So exp008's 0.20 was partly a descriptor mismatch. Per-flux R² is still negative everywhere, so the lab-to-lab spread remains unexplained.

**Target check.** The FIGRA stream found that FSRI's averaged HRRPUA curves (the source of `phrr_fsri.csv`) end early and smear peaks, so both models were rescored on per-replicate pHRR from `figra_fsri.csv`. Nothing changed:

| Target | Promoted model | Physics model | Paired gain | Repeats improved |
|---|---|---|---|---|
| Replicate mean | 0.28 | 0.56 | +0.28 | 10/10 |
| One row per replicate | 0.11 | 0.45 | +0.34 | 10/10 |

**Where this leaves the claim.** The physics model is better than the promoted one, and a paired test establishes that. Its point estimate (0.56) is well above 0.30, but its CI lower bound (0.27) isn't, and under a family-grouped split it's level with SOTA. Honest status: "ahead of Pomázi's held-out 0.30 on point estimate; not yet statistically established; not shown across material families or labs." It's proposed for promotion as `models/v2/phrr_rf_v2.joblib`. v1 stays, so exp006/exp008 remain reproducible.

**Data-quality flags on the new tables:**
- Black_PMMA's material.json says 4.8 mm stock, but its cone specimens parse to 8.4 mm, and the measured mass per area agrees with 8.4.
- MCC `t_end_C` is unstable across Black_PMMA replicates (374 / 684 / 643 °C).
- MCC `total_hr_kJg` is integrated with no baseline subtraction. Black_PMMA replicates read 18.6 / 39.9 / 33.2 kJ/g, House_Wrap reaches 57.3, and non-combustibles go negative. 28 of 118 materials have replicate CV > 0.2. This is probably why the MCC extras hurt.
- Black_PMMA's MCC `t_onset_C` (187-199 °C) sits far below its STA 5%-mass-loss onset at the same 30 K/min (300-307 °C). Real early scission or a baseline artifact, not yet known.

**Addendum (lead, same evening): rerun on corrected MCC columns, and the promotion (exp022, exp023).** exp016–exp018's physics variants used `mcc_extras_fsri.csv`'s raw `total_hr_kJg` / `t_onset_C`, which turned out to carry end-of-run baseline drift (see the MCC entry above). Before rerunning, I committed to promoting V3 (exp017's config) on the corrected columns, provided it still passed the stream's pre-declared bar: better than the v1 recipe in ≥8/10 paired repeats, with a gain CI that excludes 0. `train_phrr_v3_robust.py --bc` results: V3 **0.535**, CI [0.155, 0.804], paired +0.231 over v1 in 10/10 repeats, gain CI [+0.011, +0.555]. It passes, so it's **promoted as `models/v2/phrr_rf_v2.joblib`**, written by `--bc --save models/v2/phrr_rf_v2.joblib --save-variant "V3 rf_phys_log"`, with the log in `experiments/runs/`. `phrr_rf_v1.joblib` stays, so exp006/exp008 still reproduce.

On the corrected columns the script's highest-mean rule would pick V4 (HGB), at 0.536 with CI [0.307, 0.807], over V3 by 0.001. That's noise. V4's CI lower bound happens to clear 0.30, and choosing it for that reason after seeing it would be exactly the selection this project avoids. So it's logged (exp023), not promoted, and **not cited as a SOTA win**.

**Where pHRR stands honestly:** the promoted model is ahead of Pomázi's held-out 0.30 on point estimate (0.535), but the 95% CI includes 0.30. It clearly beats our own previous model (0.304 → 0.535, paired, 10/10). The whole-category split gives 0.27. "Beats SOTA" is withdrawn until the CI lower bound clears 0.30 on a pre-declared model. More materials is the most direct way to get there.


## 2026-09-29 (evening) — TTI take 5: every material, bench physics, two data bugs, and what really limits TTI (exp024–exp026)

Took the exp009 diagnosis at face value: every earlier TTI feature came in through an inner join that cut the set to 16-61
materials, so split noise swamped any feature effect. `train_tti_v5_physics.py` LEFT-joins everything (HGB and sklearn 1.7 RF route
NaN natively; the linear models impute inside the fold), so no row is lost to a missing feature and every candidate is scored on the
same rows and splits with the shared protocol: legacy 10-seed split, repeated GroupKFold 5x10 as headline with a material-bootstrap
CI, and a family-grouped split. Eight candidates and a pick rule were written down before anything was scored. That alone changed
nothing: exp004's RF on all 80 materials gives the same legacy number as before (0.099 vs 0.103).

**Two data bugs mattered more than any model choice.** (1) TTI = 0 means "did not ignite": 42 of the 45 zero rows in `tti_fsri.csv`
say "No ignition" in their Scalar POST TEST CMT, and the other 3 (PMMA at 15 kW/m2 without spark) are blank or "MLR did not exceed
1.0 g/min". Every TTI model since exp003 learned them as instant ignitions. Dropping them (672 rows / 77 materials) lifts every model
by 0.06-0.17. (2) The raw MCC onset temperature integrated end-of-run baseline drift, which pulled it early (replicate medians 212 C raw vs 256 C corrected). With
the corrected onset, the physics model gains another 0.10.

**Result on the fixed data: GKF R2 0.367 +- 0.039, CI [0.201, 0.520]. Does not beat SOTA 0.58.** It's a 5-input Poisson GLM straight
from ignition theory: log heat flux, log(MCC onset T - 20C), log k rho c, log thickness, log density. It beats exp004's RF on the same
rows on every protocol (GKF 0.367 vs 0.291, legacy 0.425 vs 0.254, family 0.372 vs 0.329), and none of the forests given the new feature tables
got past 0.29. Putting the physics in the model's form worked. Adding it as columns to a forest didn't.

**Why it stops there:** the GLM explains 84% of the variance in seconds at 50/75 kW/m2 and 11% at 25 kW/m2. About 20 materials sit
near their critical heat flux at 25 kW/m2: TTI jumps 6-14x from 50 to 25 kW/m2 where the flux law says 4x, those rows hold 59% of
the target's variance, and replicates there disagree 2-4x (PET at 25 kW/m2: 295-1,102 s). Predicting each replicate from its siblings
at the same material and flux only reaches R2 0.69. A critical-flux loss term from the MCC onset doesn't identify the blow-up
materials (model-free Spearman -0.015 raw, +0.15 corrected). **Material count is no longer the binding constraint. The near-critical-flux
tail of the target is.** Not promoted: every material has been in a test fold, so there's no untouched hold-out to confirm 0.367 on.
Next: fix both bugs at ingest, and screen any new T_ig / critical-flux proxy with the model-free TTI(25)/TTI(50) test before training on it.
*Erratum (2026-10-06, lever sweep tti-1):* "84% of the variance in seconds at 50/75 kW/m2" is C6's repeat-0 R2 on all non-25 rows pooled (15/35/50/75/100), mostly between-flux spread and 3 Masonite rows at 15 kW/m2. On the canonical 681-row table exp046's pooled 50+75 R2 is 0.368 and its within-flux log R2 is about 0. See `experiments/critic/tti_narrative_corrections.md`.


## 2026-09-29 (evening) — THR: fix the target, then it's an energy balance (exp027–exp032)

**`thr_fsri.csv` reproduces, but it integrates the wrong curve.** Recomputing all 220 rows from FSRI's
averaged HRRPUA files matches to 0.005 MJ/m² (PMMA at 50 kW/m²: 89.94). But those averaged files
combine replicates of different lengths in two inconsistent ways. Some are cut at the shortest
replicate (Wood_Stud at 25 kW/m²: 116.9 instead of 176.7 MJ/m²). Others run on to the longest, with a
"Mean" that averages only the replicates still burning (Engineered_Wood_Table at 25: 295.5 instead of
101.4). Integrating each replicate from the raw scans to its own end of test and then averaging
agrees with the file at the median (0.00 MJ/m²), but not on those rows. `train_thr_v1.py` reports
both targets. Its headline uses the per-replicate one.

**Six rows aren't THR at all.** Face_Shield and High_Temperature_SCBA_Facepiece (50/75/100 kW/m²)
have `POST TEST CMT` "Sample removed 10 seconds after ignition". Dropped from the clean runs.

**Physics does the work.** Specimen mass per area (pre-test Scalar fields) × MCC total heat release
per gram predicts THR with R² 0.95 before any training, using the lead's baseline-corrected MCC
column `total_hr_bc_kJg` (0.82 with the drift-affected `total_hr_kJg`). Fitting one constant χ in
training folds gives pooled out-of-fold R² 0.944 [0.917, 0.967] (169 rows / 58 materials,
material-grouped; 0.907 with raw h_c), 0.946 grouped by family, and 0.828 [0.636, 0.925] on the
uncleaned published-curve target. RF and HistGB on eight features don't beat it (0.86–0.88). exp002's
pHRR features score −0.35 on the same rows, because without fuel load there's no THR signal.

**Not a SOTA comparison.** Pomázi's held-out 0.67 is for fibre-reinforced composites on a mass-loss
calorimeter, predicted from formulation descriptors. Ours is heterogeneous products on an O2 cone,
predicted from a measured heat of combustion and the fuel load. Logged as "not comparable", like Char
Residue.

**Data flag:** with the corrected MCC values, THR/(m''·h_c) has a median of 1.04. FSRI's cone
reports more heat per kg of specimen than the MCC says complete combustion can release (PMMA at
50 kW/m²: 27.1 vs 24.2 MJ/kg). That's an absolute-scale offset between the two instruments, which
the fitted constant absorbs, so read χ as a calibration factor, not a combustion efficiency.
Unresolved.


## 2026-09-29 (late evening) — STA unit fix: onset and char residue are real now, and Char Residue gets a real first model (exp033–exp036)

**Fixed the STA bug the repro audit found, in one place.** `ingest_fsri_sta_mcc.py` now has one STA reader, `read_sta_percent()`, and `ingest_sta_onset_temp.py` imports it. For the 788/848 exports whose mass column is `Mass loss…/mg` (a change from the start), it adds the Meta file's `SAMPLE MASS` and normalises by the first row. That is FSRI's own convention, in `02_Scripts/plot_STA_data.py`. The 60 `%` exports are normalised by their first row. There was a second bug hiding behind the first: the Meta key is spelled three ways (`SAMPLE MASS /mg:`, `SAMPLE MASS /mg`, `SAMPLE MASS/mg`), so `sample_mass_mg` had been blank for 575 of 848 rows. A QA gate blanks any file whose final % lands outside [−5, 105]. Two files hit it, HIPS 30K R2 and Rubber_Foam_Pipe_Insulation 10K R1, whose Meta masses contradict their own curves.

**What changed:** 788 onsets and 844 residues out of 848. The median onset went from 50 °C to 290 °C. Materials whose mean onset sits below 150 °C went from 134 to 2, the two gypsum cores (about 120 °C, real dehydration). ABS now reads 385 °C and HDPE 460 °C. Materials with a mean N2 residue above 5% went from 4 to 127 (oak 19%, MDF 24%, HDPE −0.4%, mineral wool 98%). `mcc_hrr.csv` was regenerated by the same script and is byte-identical (sha256 da17bff…). The file schemas are unchanged, and every old consumer script still runs.

**How we know it's right now.** FSRI stores no summary tables in the clone. `03_Charts` is generated on demand. What it does have is an independent instrument: the MCC `FINAL_MASS` char yield, a separately weighed residue after anaerobic pyrolysis. Across 96 materials, STA residue vs MCC char yield had a Pearson r of 0.079 before the fix and 0.986 after, with the median gap down from 18 pp to 3.5 pp. FSRI's own onset algorithm (in `plot_STA_data_html.py`; its tables aren't in the clone, so we ran the code) sits a median 12.7 °C below our 5%-loss onset across 10 reference materials (Spearman 0.95), as its 2%-loss definition implies. The corrected values also match the audit's probe to within 5.5 °C and a median of 0.28 pp. The one real disagreement is about normalisation. In 89 files the first exported row already shows a mass change of more than 2% (plywood and cardboard moisture, plus one Wool_Rug file at +18%). We follow FSRI's first-row convention.

**New: cone residue** (`ingest_fsri_cone_residue.py` → `char_residue_cone_fsri.csv`). This is the closest FSRI analog to Pomazi's CR: final mass over initial mass from each cone replicate's load cell, using FSRI's `plot_Cone_data.py` definition, which is the mass at `END OF TEST SCAN` corrected for the scan-1 tare. There are 706 residues from 735 files, covering 79 materials. Things to know before reusing it:
- The load cell keeps logging after the test ends (ABS HF50 R1 runs to scan 1043, but the test ends at scan 563), so the last row is not the residue.
- PMMA HF25 R2's Scalar says `SPECIMEN MASS` 866.19 g, where its siblings are about 33 g. It's a typo, and a material-median mass check blanks it.
- The three PMMA `HF15-noSpark` tests hit the 1200 s cap at about 55% mass remaining. They are truncated tests, not residues, so they are kept in the CSV with their tag and excluded from modelling.
- 28 residues fall outside [−5, 105], mostly specimens under 5 g, where load-cell noise is larger than the window.

Sanity checks: PMMA −0.9%, ABS 2.8%, wood 20–23%, gypsum boards 92–95%, mineral wool 97%.

**Char Residue results.** The first real attempt is `train_char_residue_v2.py`, with configs fixed before running. Cone residue from bench-scale features (heat flux, density, MCC char yield, and STA residue at 3/10/30 K/min) scores a pooled out-of-fold R² of **0.538, 95% CI [0.11, 0.71]**, on 497 replicates / 55 materials. Flux and density alone score −0.17 on the same rows (paired +0.71 [+0.35, +1.19]). That is well short of SOTA 0.82, and the CI excludes it. Most of the signal is inert or char content, which both MCC and STA measure; on top of MCC, STA adds +0.02 [−0.14, +0.18]. Holding out whole material categories drops the score to 0.20, and so does a complete-case subset (0.26). **Not like-for-like with Pomazi**: they predict mass-loss-cone residue of fibre-reinforced thermosets from formulation inputs, while this is a standard cone on mixed products, and mineral wool, shingle, glass fibre and fillers count as residue. The exp005 successor (N2 STA residue as the target, exp005's exact config) scores 0.739 on the legacy loop, where exp005 scored −41.49, and 0.720 [0.51, 0.82] under protocol (b). **exp005's "Char Residue ruled out, only 4 materials char" was the unit bug, not the catalogue.**

**Reruns.**
- **TTI (exp007 config, same 495 rows):** the corrected onset lifts protocol (b) from 0.233 to 0.335 [0.22, 0.58], better in 10 of 10 fold-repeats. The paired material-bootstrap CI is [−0.07, +0.22], so it is a real lead but not yet a result, and still below SOTA 0.58. The old onset adds exactly 0.000.
- **pHRR (exp001 config, same 158 rows):** the corrected STA residue is neutral (paired +0.015 [−0.11, +0.21]). exp002's decision to drop STA stands on its merits.

**So nobody trips on these again:**
- FSRI files some materials twice. The STA files for `Cardboard`/`Corrugated_Cardboard` and `Plywood`/`Exterior_Plywood` are md5-identical. Merge them into one group for any STA-target CV.
- FSRI's own `plot_STA_data.py` has no `%` branch, so don't copy it blindly for the 60 `%` exports.
- The `mcc_hrr.csv` problems from the audit (the rate label, the 1 C/s files, the 43 files the regex skips) are still open. That file was deliberately left untouched today because other runs depend on it.

Details, commands and every number: `reports/sprint_0929/sta_fix.md`.


## 2026-09-29 (night) — One row per cone replicate (the recommended target source from here on; the promoted pHRR model still trains on the averaged-curve table); MaCFP PMMA recovers 2 labs

**New: `cone_replicates_fsri.csv`** (`ingest_fsri_cone_replicates.py`). One row for each of the 735 FSRI cone replicates (81 materials), with TTI, pHRR, time to peak, THR, residue and FIGRA all recomputed from the raw Scan channels, plus three flags: `did_not_ignite` (45, all `TIME TO IGN = 0`), `early_stop` (9: the six face-shield/SCBA tests and three Vinyl_Tile R4 tests ended at 1–2 min when the tile swelled) and `truncated_test` (146). The script imports the figra HRR code, the residue ingest and the THR stream's early-stop regex instead of copying them, and it asserts before writing: PMMA 50 kW/m² pHRR 1,180–1,247 and THR 90.4; HDPE 50 kW/m² keeps its 2,712/2,731 replicate peaks; pHRR and FIGRA match `figra_fsri.csv` on every row.

**How wrong the averaged files are.** Compared per material × flux against the round-1 tables (220 rows): pHRR differs by more than 5% on 36 rows, all too low in `phrr_fsri.csv` (19 because the averaged file stops at its shortest replicate, e.g. HDPE 50 at 2,235 vs 2,718; 17 from averaging peaks at different times). THR differs on 28 rows: 13 too low for the same reason, 12 too high because the file's 'Mean' keeps running on the surviving replicates only (Engineered_Wood_Table 25: 295.5 vs 101.4 MJ/m²), and 3 where Vinyl_Tile's early-stopped R4 is averaged in. TTI matches exactly, but `tti_fsri.csv` misses the 18 Frame+Grid replicates (a `+` its regex doesn't match) and still stores non-ignitions as 0 s. The old CSVs are unchanged; new target builds should read the replicate table and drop `early_stop` rows.

**Know before using `truncated_test`:** it flags tests whose THR to END OF TEST TIME misses >10% of the heat the instrument recorded in the next 60 s. That mixes specimens still burning when the test was ended with very short tests (20–60 s) that had burned out, but whose HRR signal lags past the end, probably the 11 s analyzer delay (not verified). The 0.5 g noise floor on the non-ignition half was added after House_Wrap (~1 g specimens) tripped it on load-cell noise. A first rule (HRR in the last 10 s > 10% of peak) flagged 459 of 690 ignited tests and was thrown away.

**MaCFP PMMA:** `ingest_macfp_pmma.py` now reads HKPoly (upper-case `.CSV`, column "Heat release") and UClan (one test per file, no replicate suffix): 73 tests / 12 labs in the new `phrr_macfp_pmma_all.csv`. The 67-test `phrr_macfp_pmma.csv` that exp008/exp020 use is frozen and asserted identical to the new file minus those two labs. UClan is in range. **HKPoly is not trustworthy as-is:** its pHRR is about twice every other lab's at both 25 and 65 kW/m², and its curves integrate to 316/427 MJ/m² against 167–224 elsewhere for similar specimen masses, more heat than the specimen can hold. Keep it out of cross-lab validation until the scaling is resolved.

Details, commands and every number: `reports/sprint_0929/cone_replicates.md`.


## 2026-09-29 (night) — MCC ingest fix: 9 more materials, and pHRR's point estimate shrinks (exp037–exp040)

**Fixed the MCC ingest the repro audit flagged (§3g), in one place.** `ingest_fsri_sta_mcc.py` now has one MCC file lister, `mcc_replicate_files()`, and `ingest_fsri_props.py` imports it. The old regex required `_MCC_` in the file name. FSRI named every MCC file of 9 materials without it (`Vinyl_Siding_30K_min_210609_R1.txt`), so those files were skipped without a word. `mcc_hrr.csv` and `mcc_extras_fsri.csv` go from 118 to 127 materials and from 352 to 379 replicate rows. The 16 `_O<n>` files are still left out, now on purpose: `O` is FSRI's outlier tag (`02_Scripts/plot_MCC_data.py`: "to ignore outliers"). Any other unexpected file name now fails an assert.

**The rate column was mislabelled.** `heating_rate_C_s` held 30, the K/min from the file name, in every row. It now holds the header's °C/s: 0.5 for 374 rows and 1.0 for 5 (Cotton_Shirt R1–R3, Nylon_Carpet_High_Pile R1–R2). `peak_specific_hrr_Wg` stays as measured, and every training script still reads it. A new `rate_normalized_peak_Wg` = peak × 0.5/rate puts the 1 °C/s rows on the common scale, but nothing uses it yet. Nylon_Carpet_High_Pile's doubled peak is still inside every model. On the 352 old rows nothing else changed, and the STA CSVs and `material_props_fsri.csv` are byte-identical.

**Which targets gain data.** pHRR and THR gain 21 rows / 7 materials (Composite_Deck_Board, Engineered_Flooring, Eucalyptus_Flooring, Masonite_Board, Pine_Siding, Pressure_Treated_Deck, Vinyl_Siding). TTI keeps its rows, but 70 of them now have measured MCC inputs instead of missing ones. Corian_Countertop and Fiber_Cement_Board have MCC but no cone data.

**Everything before this (exp001–exp036) ran on the pre-fix files (branch commit cb53979).**

**pHRR, pre-declared before running.**
- exp022's exact config on the fixed 196 rows / 67 materials scores **0.462 [0.140, 0.771]**, against 0.535 [0.155, 0.804] on the old 175/60.
- Under the rule written down first (lower bound > 0.30), **we do not beat Pomázi's 0.30.** Ahead on point estimate only, as before.
- It still beats the v1 recipe on the same rows: +0.175, 9/10 repeats, CI [+0.019, +0.387]. Category-grouped it scores 0.319.
- Why it dropped: the new materials are predicted fine (1.2% of the squared error on 10.7% of rows). With the old folds kept, adding them costs the old rows only 0.025. Most of the drop is GroupKFold drawing a different partition of 67 materials: split variability, not the new data.
- The pre-declared secondary, "V3 without MCC extras", scores 0.589 [0.133, 0.798]. It beats V3 in 10/10 repeats, but its gain CI is [−0.055, +0.287]. The rule needed the CI to exclude 0, so it is not adopted. On the old rows it fails the same way.

**TTI and THR reruns.**
- TTI's C6 pick goes from 0.367 to 0.348 [0.202, 0.512] on the same 672 rows.
- THR's V1 pick goes from 0.944 to 0.953 [0.918, 0.975] on 190 rows / 65 materials, up from 169 / 58.
- Neither changes a claim.

**So nobody trips on these again:**
- FSRI file names aren't uniform. Parse them with an explicit list of the variants you accept, and fail loudly on anything else.
- `heating_rate_C_s` meant K/min in any CSV or result from before 2026-09-29.
- The promoted `models/v2/phrr_rf_v2.joblib` is still the 175-row fit. Its model card number (0.535) is for the pre-fix row set.

Details, commands and every number: `reports/sprint_0929/mcc_fix.md`.


## 2026-09-29 (night) — LOI follow-up: AR-01/31 has no LOI table; AR-01/53 adds 6 epoxy rows (exp041, exp042)

**AR-01/31 found and checked, so the round-1 lead is closed.** Walters & Lyon, *Calculating Polymer Flammability from Molar Group Contributions*, DOT/FAA/AR-01/31, is on ROSA P at https://rosap.ntl.bts.gov/view/dot/92724/dot_92724_DS1.pdf. The FAA, DTIC and ResearchGate copies all failed again (503/403/403). ROSA P's search pages return 403 to our fetch tool, but its /view/dot/N pages and PDFs work, and each page's "related documents" list is how I found this one. The report's only data table (Table 1: ηc, total HR, char, MW for 100+ polymers) has **no LOI column**. LOI appears only in Figure 6, a plot with no polymer names on the points, and the figure's LOI values come from three copyrighted handbooks (Van Krevelen, Cullis & Hirschler, Hilado). Don't digitize it. Table 1 is still a citable source of ηc for pure polymers if we ever need MCC features without LOI.

**New LOI rows from a sibling report:** Lyon, Castelli & Walters, *A Fire-Resistant Epoxy*, DOT/FAA/AR-01/53 (Aug 2001), ROSA P /view/dot/92723. Table 3 prints LOI (as 1-2 point ranges) and Table 2 prints PCFC ηc for the same 9 DGEBA/DGEBC epoxy systems, measured by one group on the same formulations. That is better pairing than AR-05/14's "generic" LOI values. 6 systems have both (`ingest_faa_loi_ar0153.py` → `loi_faa_ar0153.csv`), and none collides with an AR-05/14 name. Two catches. The PDF is a scan with no text layer, so the rows were transcribed from page images and checked by a second read plus three facts stated in the prose, not by text extraction. And the LOI and ηc sit in two tables on different pages, joined on the exact (hardener, epoxy) key. The abstract gives "LOI of 30-31" for DGEBC-MDA, but its table cell is blank and the sentence is about a glass-fabric laminate, so that value was left out.

**Result:** round 1's config A (log ηc, linear) on the 38-polymer union scores GKF 0.509, CI [0.19, 0.66], essentially unchanged from 0.504 on the 32 AR-05/14 rows. Fit on AR-05/14 alone, it predicts the 6 new epoxies with MAE 2.5 LOI points (R² 0.60, n = 6). The fixed formula 125/ηc^0.25 does slightly better on them (MAE 2.1). The physics carries over to measured, not generic, LOI values. It's a data-quality gain, not a score gain. Still not comparable to sun_fu's 0.84.

**Found but not ingested:** Lyon, *Solid-State Thermochemistry of Flaming Combustion*, DOT/FAA/AR-99/56, ROSA P /view/dot/92730. Its Table 2 (text layer) gives LOI, UL 94, peak pyrolysis temperature and char for 34 polymers, 16 of them chemistries AR-05/14 lacks (PBO, PEKK, PEK, polyimide, cyanate esters, PDMS, PCTFE...), but it has no ηc. For the 15 polymers whose abbreviation exactly matches an AR-05/14 name, the two compilations agree on 10 and differ on 5, by up to 6 LOI points (PBI 42 vs 36). So even one author's "generic" LOI values move by a few points between reports. Next cheap step, if the lead approves a cross-report join: match AR-99/56 Table 2 to AR-01/31 Table 1 (ηc) by exact name. Both have text layers.

**So nobody repeats this:** also fetched and scanned for LOI on ROSA P, with no LOI tables in any of them: AR-TN97/8, AR-97/99, AR-TN08/5, TC-20/35, TC-20/30, AR-01/117, TC-TN17/45. Not yet opened: AR-TN01/75 and /view/dot/93396, 92624, 92490, 93114, 90234. fire.tc.faa.gov was down (503/timeout) all afternoon.

**Lead's ruling on the AR-01/53 join:** accepted. The brief asked for LOI and heat release capacity "on the same page/table". AR-01/53 prints them in two tables of the same report, listing the same 9 epoxy systems in the same order, joined on the exact (hardener, epoxy) key. That's the same kind of within-report exact-key join round 1 used for Tables A-6/A-2/A-5. The point of the rule was self-consistent pairs from one source, and this meets it. **Open item for a human (about 2 minutes):** AR-01/53 is a scan without a text layer, so its 6 rows were transcribed by eye, with no independent text extraction to check against. Compare PDF pp. 16 and 19 against the rows in `ingest_faa_loi_ar0153.py` before anyone leans on exp041/exp042.


## 2026-09-29 (night) — pHRR and THR moved to the per-replicate cone target; pHRR number of record 0.567 [0.283, 0.836], still not a SOTA win (exp043–exp045)

**Decision first, numbers second.** Before any model was run, the lead made `cone_replicates_fsri.csv` the canonical pHRR target: per material × flux, the mean replicate pHRR over tests that ignited and were not stopped early. Whatever the pre-declared pick scored on it would become the number of record, even if lower. Three runs were written down before running (`reports/sprint_0929/replicate_targets.md`), and none was added.

**A1, exp022/exp037's exact config (V3 rf_phys_log, `--bc`) on the new target: 0.567 [0.283, 0.836]** on 191 rows / 65 materials. The lower bound misses 0.30 by 0.017, so by the rule **Dravix still does not beat Pomázi's 0.30.** It beats its own V0 (the exp014 recipe, 0.201 on the same rows) by +0.366 in 10/10 repeats, gain CI [+0.169, +0.678]. By material category it scores 0.315.

**Why the number rose from exp037's 0.462.** Not because of the replicate values. On the 190 pairs both row sets share, with identical folds, the same config scores 0.571 on the old averaged-file target and 0.566 on the new one (post-hoc diagnostic). The difference is the rows. The six Face_Shield and SCBA-facepiece pairs leave, because every one of those tests was pulled about 10 s after ignition, so their "peak" was never a peak. Losing two materials also reshuffles the folds, so not all of the +0.11 is signal. Masonite_Board at 15 kW/m², which FSRI never averaged, comes in.

**A2, the rate-normalised MCC peak, is not adopted:** −0.003, better in 3/10 repeats. It only changes Nylon_Carpet_High_Pile, the one 1 °C/s material with cone rows, so it could hardly have mattered. **A3, THR on the replicate table:** 0.953 [0.918, 0.975], the same as exp040 to three decimals. Still not like-for-like with Pomázi.

`train_phrr_v3_robust.py --replicate-target` runs only V0/A1/A2. Its default output is byte-identical to the committed exp037/exp038 log. `train_thr_v1_replicates.py` wraps `train_thr_v1.py` without touching it.


## 2026-09-29 (night) — LOI cross-report join (AR-99/56 x AR-01/31): 2 new polymers, stopped (no experiment)

The lead approved one cross-report join: LOI from Lyon, DOT/FAA/AR-99/56 Table 2 (34 polymers, ROSA P /view/dot/92730) matched by exact name to heat release capacity from Walters & Lyon, DOT/FAA/AR-01/31 Table 1 (ROSA P /view/dot/92724; 86 rows, not "100+"). `ingest_faa_loi_ar9956.py` → `loi_faa_ar9956_ar0131.csv` keeps all 34 AR-99/56 rows as verbatim text-layer strings. The 12 that join carry `pairing = "cross-report exact-name (AR-99/56 LOI + AR-01/31 HRC)"`, and the other 22 say why in `match_note`. AR-99/56's text layer is OCR over a scan, so its rows were checked against a second extractor (pdfminer) and against the rendered page image: numbers all agree, three names are OCR-damaged ("Pl" for PI, "P A6", "a" for α). AR-01/31's rows come from pdfplumber's cell grid and were cross-checked against pypdf, 86/86.

**Result: only PEKK (LOI 40, ηc 96) and PEK (LOI 40, ηc 124) are new.** The other 10 joins (PBI, PAI, PTFE, PEEK, PEI, PE, PET, PS, POM, PVC) are already in the 38-row set. The brief set a minimum of 3 new polymers, so the pre-declared model runs were not done, and exp041 stays the LOI headline. The exact-name rule (spacing/case only) rejected 8 more new chemistries that look like the same polymer under another name: PBO, PX, polyimide, LCP, PSF, BCE, poly(α-methylstyrene), and fluorinated cyanate ester (calculated-estimate LOI). A looser rule would make this source worth a model run. That's a decision, not a transcription question.

**Label and feature noise, measured.** AR-99/56 and AR-05/14 share 18 polymers. LOI agrees exactly on 11; the worst gap is PBI, 42 vs 36. The two FAA compilations also disagree on ηc for 5 of the 10 shared polymers, and POM is the worst: 169 J/g-K in AR-01/31 vs 398 in AR-05/14. Same author group, a few years apart: "the" LOI and "the" ηc of a polymer move by several points and up to a factor of 2.4. Any cross-report pairing inherits both.

**Lead's ruling (same evening):** the strict name rule stays. The 8 "same polymer, different name" candidates (PBO, PX, polyimide, LCP, PSF, BCE, poly(α-methylstyrene), fluorinated cyanate ester) are listed in `reports/sprint_0929/loi_crossreport.md` for a human to confirm one by one before any relaxed join. The heat release capacity of the same named polymer differs by up to ×2.4 between FAA reports (POM 169 vs 398), so a rushed join would add label noise, not signal. No leaderboard row, because nothing was trained.


## 2026-09-29 (night) — TTI on the canonical replicate table: 0.313, down from 0.348, still well short of SOTA (exp046, exp047)

One pre-declared run, no variants: exp039's pick (C6, the 5-input Poisson GLM with the baseline-corrected MCC onset) with TTI read from `cone_replicates_fsri.csv` instead of `tti_fsri.csv`, dropping `did_not_ignite` and `early_stop` rows. The rule said this becomes TTI's number of record, and SOTA 0.58 counts as beaten only if the CI lower bound clears it. `train_tti_v5_physics.py` gained a `--replicate-table` flag. Default behaviour is unchanged (asserted under `--onset-bc`: identical data frame).

**Result: GKF R2 0.313 +- 0.037, CI [0.071, 0.496]; legacy 0.141 +- 0.451; family-grouped 0.356.** Not SOTA. The same-rows exp004 RF gets GKF 0.261, legacy 0.162, family 0.346. C6 still wins on the headline protocol and the family split, but no longer on the legacy one, where a single seed scores -1.18.

The row set is 681 replicates / 76 materials. Against exp039, 665 rows are shared with identical TTI, 16 Frame+Grid rows come in (Cotton_Raw, new, and Rubber_Band), and 7 early-stopped rows go out, taking Face_Shield and the SCBA facepiece with them. Rerunning exp039 on its own rows today reproduces 0.3475 exactly, so the drop is the row set. It's also smaller than one GKF repeat std, and the folds aren't paired, so "worse" is not established either.

**What the bad seed shows:** Latex_Pillow_Foam has no density and no k rho c on file. Median imputation makes it a typical solid 35.6 mm thick, and the physics model predicts up to 414 s for a foam that ignites in 1.0-21.5 s. The two new materials have the same gaps. So the physics model is only as good as the physical inputs, and 9-21 of the 76 materials lack a given input (onset 11, k rho c 21, thickness 9, density 9). Also noted: the 9 early-stopped replicates all ignited (13-86 s) before the operator stopped them. Their TTI is real data, and dropping them is a choice worth revisiting for TTI specifically.

Details and every number: `reports/sprint_0929/tti_replicates.md`.

## 2026-09-30 — Parallel session: thermal inertia, LOI/FIGRA rediscovered independently (renumbered exp048-052 on merge)

This session worked from an earlier branch point and, unaware of the sprint above, independently ingested density (`ingest_fsri_density.py`) and derived thermal inertia for TTI, hit the same result pattern (real physical improvement in CV mean, fails the locked single-split check), and separately derived FIGRA via the max(HRR(t)/t) definition rather than the competitor's actual pHRR/t_pHRR definition used above. **On merge, the sprint's FIGRA implementation (`ingest_fsri_figra.py`, per-replicate, competitor-matched definition, 0.03% cross-checked against FSRI's own curves) superseded this session's max(HRR(t)/t) version** — kept here only as a record of the independent finding, not as the active FIGRA pipeline. Original exp009-013 renumbered to exp048-052 to avoid colliding with the sprint's own exp009-047.

## 2026-09-30 — Real thermal inertia closes most of the TTI gap, but doesn't survive the locked check

**The missing piece for thermal inertia was density**, and FSRI has it: each material's `material.json` carries a stated density with a source note (e.g. PMMA: "1180 kg/m³", sourced to "samples cut from 0.32 cm thick... commercially available cast sheet"). Ingested via `ingest_fsri_density.py` — 117 of 183 materials have one.

**Caught a real bug before using it.** The first version of the parser stripped HTML tags from strings like `"1180 kg/m<sup>3</sup>"` and then averaged *every number left in the string* — which silently included the exponent's bare `3`, giving PMMA a density of 591.5 instead of 1180 (and EPS Foam Board 15.5 instead of 28). Caught by the same sanity-check habit used throughout this project (checking PMMA and EPS against known values) before it was ever used in a model. Fixed to take only the first number in the raw string.

**exp048: assembled real thermal inertia (k·ρ·cₚ) from three independently-measured properties** — HFM conductivity (`thermal_conductivity_fsri.csv`), HFM specific heat (`specific_heat_fsri.csv`, already volumetric i.e. ρ·cₚ), and the density above. This is the actual textbook physical predictor of ignition time (Quintiere-style ignition models), not a bulk-combustion proxy standing in for it. Result: **mean held-out R² jumped 0.167 → 0.267** — the biggest single improvement on TTI so far — and `thermal_inertia` came out as the #2 most important feature (0.272), right behind heat flux (0.287) and well ahead of the old MCC/onset-temperature proxies. Best individual seed hit R²=0.71.

**exp049: tried pruning the two lowest-importance features (specimen_mass, spark_ignitor) — made it worse** (0.267 → 0.119). Low marginal importance in a correlated feature set didn't mean safe to drop; reverted to exp048's full feature set. Recorded so this exact pruning isn't retried.

**exp050: ran the same locked, pre-committed single-split promotion check that validated the pHRR model (exp006) — this one did NOT pass.** Committed to seed=42 before running (same convention as exp006), and got R²=−0.0037 — essentially no better than predicting the mean, nowhere near SOTA's 0.58. This is consistent with exp048's own wide seed spread (−0.02 to 0.71 across the original 10 seeds): the CV mean of 0.267 is a real average tendency, but any single held-out draw can land badly at only 53 materials. **Honest conclusion: real thermal inertia is a genuine, substantial improvement in what the model understands about ignition — not yet a reliable enough result to promote.** pHRR (exp006) remains the only promoted model. More materials with full conductivity+specific-heat+density+cone coverage (currently the join bottleneck, 53 of ~183) would be the way to make this result stable enough to trust on a single split.

**Two more honest negative results, closing out this line of small-dataset feature tweaks:**
- **exp051**: tried dropping the density join requirement to recover 3 more materials (53→56) for TTI, since density isn't mathematically needed to *compute* thermal inertia (only to include it as its own separate feature). Result: much worse (mean R² 0.267→−0.225, one seed at −2.07). Density mattered as its own feature more than the extra 3 materials helped — reverted, exp048 stays the best TTI config.
- **exp052**: tried adding the same real thermal properties to pHRR, hoping to also narrow exp008's cross-lab generalization gap. Backfired harder: the join shrank from exp006's 175/60 materials to 131/45 (k/cp coverage is the bottleneck), and the smaller pool made the model badly unstable (mean R²=−0.16, locked check −1.84). The promoted pHRR model is unchanged by this — it stays strictly better.

**Four attempts in a row after exp048 (exp049, 011, 012, 013) all came back flat or negative — a clear, consistent signal that the "more features on the same small FSRI join" lever is exhausted for this session on both TTI and pHRR.** Don't keep re-trying permutations of the same small joined dataset without either more materials or a different data source.

## 2026-09-30 (continued) — FIGRA attempted, stopped at a real data-quality problem before training anything

> **SUPERSEDED by `097ef71`; see `reports/sprint_0929/figra.md` §3.** This entry describes `max HRR(t)/t`. The merged definition is pHRR / t_pHRR from test start, to which the time floor does not apply. (Marked 2026-10-01 per `plans/FABLE_PLAN_2026-10-01.md` §3 B4.)

**Derived FIGRA properly, not with a shortcut** (`ingest_fsri_figra.py`): rather than approximating FIGRA as peak-HRR ÷ time-of-peak (the lazy version), computed the actual standard definition — max over the whole curve of HRR(t)/t — from the same per-heat-flux mean HRRPUA curves already used for pHRR/THR. Checked whether the shortcut would have given the same answer: **it would have matched the proper calculation in only 24 of 216 cases (11%)** — confirms the shortcut would have been wrong most of the time, good thing it wasn't used.

**But the proper calculation has its own real problem.** A time floor (10s) is needed to keep HRR(t)/t from spiking meaninglessly near ignition (t→0). Checking where the maximum actually lands: **it sits exactly at that 10s floor in 48% of all rows.** That means for close to half the dataset, "FIGRA" as computed here isn't measuring a genuine feature of the material's burning curve — it's measuring where an arbitrary cutoff was placed. Reporting that number as a real target would be building on sand.

**Stopped here rather than train a model on it.** This is the same discipline applied to Char Residue (exp005) and the LOI search — diagnose and disclose a real data-quality wall instead of pushing through it for a number. FIGRA needs either a principled choice of time floor (e.g. derived from each material's own ignition-transient timescale, not one fixed value for every material/heat-flux) or a finer-resolution curve near ignition before it's trustworthy enough to model. Left as a genuine next step, not attempted further today. `figra_fsri.csv` is saved with the `figra_equals_peak_over_tpeak` flag column so this can be picked back up without redoing the derivation.

## 2026-09-30 (post-merge) — Merged the parallel sprint branch; two new honest results

**Merged `v2-sprint-0929` into `master`** (commit `097ef71`). That branch, worked independently and in parallel with this session, had already: fixed a real STA/MCC unit bug (mixed mg-vs-% in the char-yield feature, caught via a repro audit), added a canonical per-replicate cone table with proper did-not-ignite/early-stop flags, promoted `phrr_rf_v2`/`phrr_rf_v3` on top of the fixed data, got the first real LOI rows from a public-domain FAA report, built a properly-defined FIGRA (matching the competitor's actual pHRR/t_pHRR definition, per-replicate, cross-checked to 0.03% against FSRI's own curves), and un-ruled-out Char Residue after fixing its parser. The merge had five real conflicts (both sessions used exp009-013 for different experiments; both touched FIGRA, the acquisition log, and the two xlsx trackers) — resolved by renumbering this session's exp009-013 to exp048-052, taking the sprint's more rigorous FIGRA implementation, and interleaving the two narrative logs rather than picking one side.

**exp053: retried thermal inertia on the corrected canonical table.** The two sessions' best ideas (real thermal inertia as a TTI feature; the corrected per-replicate table with proper ignition flags) had never been tried together. Result: GKF mean R2 actually *dropped* slightly (0.2613 -> 0.2111) because the thermal-property join shrinks the table from 76 to 52 materials. Fourth consecutive confirmation that coverage, not feature choice, caps TTI on this data — logged so this combination isn't retried again.

**exp054: re-ran the pHRR cross-lab check against the current promoted model and the fuller MaCFP file.** exp008 (2026-09-29) checked `phrr_rf_v1` against 10 institutions (67 tests): R2=0.1986. This reruns the same style of check against the now-current `phrr_rf_v3` and the sprint's expanded file (12 institutions, 73 tests, adds UClan and HKPoly). Cross-lab R2=0.5884 with HKPoly excluded (a real, large improvement — both from the better model and the fixed data underneath it), or 0.5014 if HKPoly is included. HKPoly's exclusion is independently confirmed here, not just taken on the sprint's word: its true/predicted ratio is 2.75, consistent with the sprint's "~2x every other lab" flag. This is a genuine, honest strengthening of the promoted model's out-of-distribution validation story — still not a SOTA claim (SOTA is a same-lab held-out number), but a materially better answer to "does this generalize to a lab it's never seen."

## 2026-09-30 (post-merge) — Real-records recount after the merge (was never filled in for today)

**Filling in today's row in the daily scoreboard**, which is what looked like "nothing changed" on 9/30 despite real work happening: the row for 2026-09-30 in `reports/daily_scoreboard.xlsx`'s Daily Scoreboard tab was simply never filled in before the merge (row 6, 2026-09-29, was the sprint's; row 7 sat empty). That was the actual bug behind the confusion, not a lack of progress.

**Recomputed "Real records" from scratch, post-merge, to avoid double-counting.** The two sessions' data tables overlap: `cone_replicates_fsri.csv` (735 rows) is a re-extraction, per replicate, of the exact same physical cone tests that `tti_fsri.csv` (717), `phrr_fsri.csv` (220), `char_residue_cone_fsri.csv` (735) and `figra_fsri.csv` (735) also each represent -- summing all five would count the same ~735 real tests up to 5 times over. Likewise `onset_temp_sta.csv` and `char_yield_sta.csv` are two different reductions of the same 848 STA runs, and `mcc_extras_fsri.csv` is a second reduction of the same 379 MCC runs as `mcc_hrr.csv`. Counting each *physical measurement source* once:

| Source | Rows | What it is |
|---|---|---|
| `cone_replicates_fsri.csv` | 735 | canonical per-replicate cone tests (supersedes tti/phrr/residue/figra as separate counts) |
| `char_yield_sta.csv` | 848 | STA runs |
| `mcc_hrr.csv` | 379 | MCC runs |
| `thermal_conductivity_fsri.csv` | 1,461 | HFM conductivity readings |
| `specific_heat_fsri.csv` | 1,099 | HFM specific-heat readings |
| `material_props_fsri.csv` (density only) | 117 | FSRI-stated material density |
| `loi_faa_polymers.csv` | 47 | FAA AR-05/14 Table A-6 |
| `loi_faa_ar0153.csv` | 9 | FAA AR-01/53 epoxies |
| `loi_faa_ar9956_ar0131.csv`, new rows only | 2 | the 2 rows flagged `new_vs_existing38=True`; the other 10 classified rows duplicate polymers already in the 47 above |
| `phrr_macfp_pmma_all.csv` | 73 | independent cross-lab PMMA (12 institutions) |
| **Total** | **4,770** | |

This is close to, but not identical to, the sprint's own 4,817 (which counted derived reductions separately) or this session's pre-merge 4,764 (which predates the LOI/MaCFP-full additions). Recorded as the new number of record; future days should extend this table rather than re-deriving it from a running sum.

**Today's row (2026-09-30):** 4,770 real records, 38 usable LOI, 681 usable TTI (exp047's canonical table), 191 usable pHRR (exp043's canonical per-replicate training set -- fewer than yesterday's 220 because it correctly excludes non-ignitions/early-stops, not a regression), 54 experiments run (exp001-054), best target still Peak HRR at 0.5672 vs SOTA 0.30 (unchanged number -- no new pHRR model was promoted today, exp054 only re-validated the existing one cross-lab).

## 2026-10-04 (post-V3) -- data pass (`experiments/configs/data_pass_oct4.json`)

**FSRI integrating-sphere emissivity** -> `data/v2/emissivity_fsri.csv` (`data/v2/ingest_fsri_emissivity.py`). Source already on disk (FSRI clone, commit a432697, `01_Data/<material>/FTIR/IS/*.dpt`); computed exactly as FSRI's `plot_IS_emissivity_data_html.py` (light-trap corrected reflectance, transmission subtracted where measured, Planck-weighted total emissivity at 600-2000 K; absorption coefficient from -ln(tau) vs thickness). 59 materials (3 with transmission: Nylon, PC, PEVA); 4 skipped exactly as FSRI's script skips them (MDF, OSB, two Overstuffed_Chair parts: no `_S_`/`_T_` files). 42 of the 81 cone materials covered. Sanity: Black PMMA 0.967, PVC 0.954, PLA 0.977 at 1000 K. Data only; thermal-property joins stay banned until a pre-declared round.
