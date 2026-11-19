# Dravix cone-calorimeter literature corpus, v2 preview (datasheet)

Nikhilesh Suravarjjala, Arnav Saini | 2026-11-19 | branch `dataset-v2`

**Status.** Preview on GitHub. We plan to deposit this version on Zenodo soon; it has no DOI yet. The v1 subset is already archived at [10.5281/zenodo.23654307](https://doi.org/10.5281/zenodo.23654307).

**Purpose.** A tabular dataset of flame-retarded polymer formulations and their cone-calorimeter results, extracted from the published literature, for training and benchmarking formulation-to-fire-property models.

**Bottom line.** 5,788 rows from 1,016 papers, CC BY 4.0. Same 45 columns and extraction rules as v1, plus one flag column, `in_release_v1`. Every value is copied from a table or explicit text, never read off a figure, and every row carries a locator for where it came from.

## 1 - Files

| file | content |
|---|---|
| `cone_lit_corpus_v2.csv` | the data: 5,788 rows x 46 columns, UTF-8, comma-separated, one header row |
| `LICENSE_DATA.md` | data licence (CC BY 4.0, with caveats), the same as v1 |
| `SHA256SUMS` | checksum of the CSV |

Columns, units, the meaning of one row, extraction rules, and the QA procedure are documented in the v1 datasheet that ships with the Zenodo record (doi above). Nothing in this file changes them. The folds in v1's `splits_v1.csv` cover v1 rows only; there are no folds for the added rows yet.

## 2 - How this relates to v1

| rows | count | how to find them |
|---|---|---|
| Carried over from v1, unchanged | 4,528 | `in_release_v1 = 1`. All 45 original columns are identical to the v1 file, row for row |
| Textile and coated-substrate rows (removed from v1) | 604 | `in_release_v1 = 0` and `material_form` is `textile` or `coated_substrate`. Outside Dravix's modelling scope |
| Wave-2 rows, extracted after the Dravix engine was built | 655 (116 papers) | `in_release_v1 = 0`, any other `material_form`, `row_id` not `PMC10343274-22`. This is the set the engine was scored on once |
| One corrected row | 1 | v1 `PMC10343274-21` (pHRR 176) is replaced by `PMC10343274-22` (pHRR 122) after field verification. `in_release_v1 = 0` |
| Total | 5,788 | 1,016 papers |

Rows come from three source sets: `oa` 3,871 (Europe PMC open access), `lib` 1,533 and `lpc` 384 (full texts accessed through a college library). No full text is distributed.

## 3 - Values per property

Counts of rows with a value above zero, and the number of papers they come from.

| property | column | rows | papers |
|---|---|---|---|
| Time to ignition | `tti_s` | 4,386 | 766 |
| Peak heat release rate | `phrr_kWm2` | 5,384 | 966 |
| Total heat released | `thr_MJm2` | 4,516 | 841 |
| Average heat release rate | `avg_hrr_kWm2` | 1,059 | 174 |
| Limiting oxygen index | `loi_pct` | 2,550 | 578 |
| Char residue (cone) | `char_residue_pct` | 2,040 | 401 |
| UL-94 rating (as printed, rows non-blank) | `ul94` | 2,140 | |

## 4 - Quality checks and limits

- The 0.22% field error rate in the v1 datasheet comes from a random 20% sample of papers in an earlier 835-paper version. It does not cover the rest of v1 or this release.
- The wave-2 extraction was checked field by field: 7,523 checks, 29 values corrected, 10 values removed.
- No separate error rate is reported for the textile and coated-substrate rows.
- Textile and coated rows, and rows from different material forms, are mixed in one file. Filter on `material_form` before modelling.
- Flame-retardant names are as printed in the papers; synonyms are not merged.
- Not for fire-safety certification.

## 5 - Licence

CC BY 4.0, with the same caveats as v1: see `LICENSE_DATA.md`. The source papers remain under their publishers' terms; please cite them when you use their values (the `doi` column identifies each paper).

## 6 - How to cite

Until the Zenodo v2 record exists, cite the v1 record for the shared rows and name this file and its SHA-256 for the added rows:

Suravarjjala, N., & Saini, A. (2026). *Dravix cone-calorimeter literature benchmark* (Version v1) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.23654307

Report extraction errors by opening an issue with the row's `row_id` and DOI.
