# Dravix

[![Licence: Apache-2.0](https://img.shields.io/badge/licence-Apache--2.0-blue.svg)](LICENSE)

Dravix predicts cone-calorimeter results (time to ignition, peak and average heat release rate, total heat released) for flame-retarded polymer formulations, scores them with a composite fire-risk screening score (CFRS Score), and benchmarks the predictions on data extracted from the published literature.

Authors: Nikhilesh Suravarjjala, Arnav Saini. Version 2.0.0. See `CITATION.cff` to cite.

| | |
|---|---|
| Website | https://dravix.materiamse.com |
| Engine (web app) | https://dravix-engine.materiamse.com |
| Blog post (Materia MSE) | https://www.materiamse.com/blog/tri-valley-students-beat-georgia-tech-ustc-bme-budapest |

**Read this first.** Dravix is a screening aid. It is not a substitute for a fire test and is not for certifying a material. Held-out accuracy depends heavily on how the data is split: random row splits overstate skill on new materials, so use the paper-grouped protocol. The protocol, the reference results and every comparison with published models (with its caveats) are in `BENCHMARK.md`.

## Quick start

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r api/requirements.txt
.venv/bin/python api/build_model.py                 # trains the engine (~1 min), writes the compact model
.venv/bin/python api/build_uncertainty.py           # conformal intervals and OOD flag
DRAVIX_PASSWORD=change-me DRAVIX_SECRET=$(openssl rand -hex 32) \
  DRAVIX_CORS_ORIGINS=http://localhost:5173 .venv/bin/uvicorn api.main:app --port 8000
# in a second shell:
DRAVIX_PASSWORD=change-me .venv/bin/python api/smoke_test.py http://localhost:8000 http://localhost:5173
```

Then run the dashboard (`dashboard/`, `npm install && npm run dev`, with `VITE_API_BASE_URL=http://localhost:8000`).

## Layout

| path | what |
|---|---|
| `engine/` | the formulation-to-property model; no model files are shipped |
| `engine/score/` | the CFRS Score (module `cfrps.py`, weights `weights/cfrs_weights_*.json`) |
| `api/` | FastAPI service: predictions, conformal intervals, out-of-distribution flag, CFRS |
| `dashboard/` | web front end for the API (source only) |
| `data/v2/` | training, evaluation and ingest scripts; the literature snapshot `cone_lit_benchmark_v1.csv` |
| `experiments/` | pre-declared configs, independent critic reviews, results and run logs |
| `experiments/challenger/` | the USTC-style pHRR runs (`RESULTS_BEST_OF_N.md`) |
| `benchmarks/` | published competitor numbers with their caveats |
| `reports/` | write-ups and model cards |

## Notes

- To reproduce the numbers of record, use Python 3.12 with `requirements-record-py312.txt` (scikit-learn 1.7.1); other scikit-learn versions build different forests and folds from the same seeds.
- `data/v2/train_litcone_gt_round1.py` rewrites the snapshot CSV when it runs and expects extraction files that are not in this repository. Use the shipped snapshot as is.
- Some result files come from data that is not redistributed here: the BME reconstruction, the wave-2 exam rows, the FSRI-derived tables and the same-data checks on other groups' datasets. Their configs, logs and results are included; the inputs are not.
- Local user paths were replaced with `~`. Otherwise configs and critic files are verbatim, so some cite internal planning files and the private development repository, neither of which is published.
- `experiments/scoreboard_entries.json` entries 2 and 3 quote withdrawn numbers from a variant that leaked; they are kept because the file is append-only.
- Dravix was built with substantial help from AI coding agents (Anthropic Claude) under pre-declared configs and independent critic passes. The authors are responsible for the content.

## Feedback

Issues and pull requests are welcome, especially reports of extraction errors in the benchmark dataset (cite the row's `row_id` and DOI).

## Acknowledgements

We thank Jaanya Gupta, Dravix's business lead, for her work on the project.

## Licence

Code: Apache-2.0 (`LICENSE`, `NOTICE`). Benchmark dataset: CC BY 4.0, published as a separate dataset record; the copy at `data/v2/cone_lit_benchmark_v1.csv` is identical and is also CC BY 4.0, not Apache-2.0. Patent pending: U.S. provisional application No. 64/054,493.
