# Dravix

Machine-learning screening of polymer fire risk from cone-calorimeter data. Predicts time to ignition, peak and average heat release rate, and total heat released for flame-retarded formulations, and tests those predictions against a literature benchmark.

[![Licence: Apache-2.0](https://img.shields.io/badge/licence-Apache--2.0-blue.svg)](LICENSE)
[![Software DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23653502.svg)](https://doi.org/10.5281/zenodo.23653502)
[![Dataset DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23654307.svg)](https://doi.org/10.5281/zenodo.23654307)
[![Live site](https://img.shields.io/badge/site-dravix.materiamse.com-762123)](https://dravix.materiamse.com)

Dravix is a screening aid. It helps rank candidates before a fire test. It does not replace the test and is not for certifying a material.

## Evidence

| Item | Where |
|---|---|
| Software record, v2.0.0 | [10.5281/zenodo.23653502](https://doi.org/10.5281/zenodo.23653502) |
| Benchmark dataset, v1, CC BY 4.0 | [10.5281/zenodo.23654307](https://doi.org/10.5281/zenodo.23654307) |
| Full dataset, v2 preview: 5,788 rows from 1,016 papers, CC BY 4.0 (Zenodo deposit planned) | [`dataset-v2/`](dataset-v2/) on branch `dataset-v2` |
| Protocol, results and comparisons | [`BENCHMARK.md`](BENCHMARK.md) |
| Pre-declared configs, critic reviews, run logs | `experiments/` |
| Live site and engine demo | [dravix.materiamse.com](https://dravix.materiamse.com), [dravix-engine.materiamse.com](https://dravix-engine.materiamse.com) |

## Honest results

The headline protocol is paper-grouped 5-fold cross-validation (10 repeats): the model predicts papers it has never seen. Pooled R2, mean over repeats, shipped model:

| Target | Rows | Papers | Paper-grouped R2 | Random-split R2 |
|---|---|---|---|---|
| Peak HRR | 4,017 | 725 | 0.286 | 0.604 |
| Total heat released | 3,355 | 619 | 0.244 | 0.683 |
| Average HRR | 759 | 130 | 0.234 | 0.677 |
| Time to ignition | 3,368 | 598 | -0.247 | 0.376 |

One later test: 655 rows from 116 unseen papers, scored once with the script and config committed beforehand. R2 was 0.509 (pHRR), 0.486 (THR), 0.642 (average HRR) and 0.186 (TTI), with wide intervals. That population differs from the grouped protocol, so it does not prove better generalization.

## Limitations

- Random row splits overstate skill on new materials. Use the paper-grouped protocol.
- Time to ignition is worse than predicting the mean on unseen papers (R2 below zero).
- Training-fit R2 near 1 is memorization, not skill.
- No result is marked comparable with a published model. `BENCHMARK.md` section 6 lists each comparison with its caveats.
- Extraction error in the dataset is 0.22% on a 20% sample. The rest is unchecked.
- Dravix was built with substantial help from AI coding agents (Anthropic Claude) under pre-declared configs and independent critic passes. The authors are responsible for the content.

## Quickstart

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r api/requirements.txt
.venv/bin/python api/build_model.py                 # trains the engine (~1 min)
.venv/bin/python api/build_uncertainty.py           # conformal intervals and OOD flag
DRAVIX_PASSWORD=change-me DRAVIX_SECRET=$(openssl rand -hex 32) \
  DRAVIX_CORS_ORIGINS=http://localhost:5173 .venv/bin/uvicorn api.main:app --port 8000
# in a second shell:
DRAVIX_PASSWORD=change-me .venv/bin/python api/smoke_test.py http://localhost:8000 http://localhost:5173
```

The dashboard is in `dashboard/` (`npm install && npm run dev`, with `VITE_API_BASE_URL=http://localhost:8000`). To reproduce the numbers of record, use Python 3.12 with `requirements-record-py312.txt` (scikit-learn 1.7.1). Other versions build different folds.

## Layout

| Path | What |
|---|---|
| `engine/` | formulation-to-property model (no model files shipped) and the CFRS screening score |
| `api/` | FastAPI service: predictions, conformal intervals, out-of-distribution flag |
| `dashboard/` | web front end for the API |
| `data/v2/` | training, evaluation and ingest scripts, and the literature snapshot |
| `experiments/` | pre-declared configs, critic reviews, results and run logs |
| `benchmarks/` | published competitor numbers with caveats |
| `reports/` | write-ups and model cards |

Some result files rely on data that is not redistributed (the BME reconstruction, wave-2 exam rows, FSRI-derived tables). Their configs, logs and results are included; the inputs are not. `experiments/scoreboard_entries.json` entries 2 and 3 quote withdrawn numbers from a variant that leaked. They stay because the file is append-only.

## How to cite

Suravarjjala, N., & Saini, A. (2026). *Dravix: formulation-to-cone-calorimeter property models and benchmark code* (Version 2.0.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.23653502

Suravarjjala, N., & Saini, A. (2026). *Dravix cone-calorimeter literature benchmark* (Version v1) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.23654307

Machine-readable metadata is in [`CITATION.cff`](CITATION.cff).

## Feedback

Issues and pull requests are welcome, especially reports of extraction errors in the dataset (cite the row's `row_id` and DOI).

## Licence

Code: Apache-2.0 (`LICENSE`, `NOTICE`). The benchmark dataset is CC BY 4.0. The copy at `data/v2/cone_lit_benchmark_v1.csv` is identical and carries the same licence.

Authors: Nikhilesh Suravarjjala, Arnav Saini. We thank Jaanya Gupta, Dravix's business lead.
