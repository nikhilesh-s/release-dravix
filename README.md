# Dravix

Dravix predicts cone-calorimeter results (time to ignition, peak and average heat release rate, total heat released) for flame-retarded polymer formulations, and benchmarks that prediction on data extracted from the published literature.

Authors: Nikhilesh Suravarjjala, Jaanya Gupta, Arnav Saini. Version 2.0.0. See `CITATION.cff` to cite.

**Read this first.** Dravix is a screening aid. It is not a substitute for a fire test and is not for certifying a material. Held-out accuracy depends heavily on how the data is split: random row splits overstate skill on new materials, so use the paper-grouped protocol. The protocol and the reference results are in `BENCHMARK.md`.

## Layout

| path | what |
|---|---|
| `engine/` | the formulation-to-property model (`dravix_engine.py train` rebuilds it in about 35 s; no model files are shipped) |
| `api/` | FastAPI service around the engine (`api/build_model.py` builds the model it serves) |
| `dashboard/` | web front end for the API (source only) |
| `data/v2/` | training, evaluation and ingest scripts; the literature snapshot `cone_lit_benchmark_v1.csv` |
| `experiments/` | pre-declared configs, independent critic reviews, results and run logs |
| `benchmarks/` | published competitor numbers with their caveats |
| `reports/` | write-ups and model cards |

## Environment

Python 3.12 with the pins in `requirements-record-py312.txt` reproduces the numbers of record. `requirements.txt` builds the engine; `api/requirements.txt` runs the API.

```bash
uv venv --python 3.12 .venv312
uv pip install --python .venv312/bin/python -r requirements-record-py312.txt
```

## Notes

- `data/v2/train_litcone_gt_round1.py` rewrites the snapshot CSV when it runs and expects extraction files that are not in this repository. Use the shipped snapshot as is.
- Local user paths in logs and scripts were replaced with `~`. Configs and critic files are verbatim.
- `experiments/scoreboard_entries.json` entries 2 and 3 quote withdrawn numbers from a variant that leaked; they are kept because the file is append-only.
- Dravix was built with substantial help from AI coding agents (Anthropic Claude) under pre-declared configs and independent critic passes. The authors are responsible for the content.

## Licence

Pending. See `LICENSE.PENDING`.
