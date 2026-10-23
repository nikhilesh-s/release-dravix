# Dravix Engine

Formulation → cone-calorimeter predictions (TTI, pHRR, THR, average HRR).

```bash
pip install -r requirements.txt
python3 engine/dravix_engine.py train                          # ~2 min; writes engine/dravix_engine.joblib
python3 engine/dravix_engine.py predict --json engine/example.json
python3 engine/server.py                                       # demo UI at http://localhost:8765
```

- **Model:** the declared litcone primary blend (HGB + ExtraTrees + Ridge, log1p target, v3 features, clipped to the training range).
- **Training data:** fit on all 4,529 rows from 806 papers in `data/v2/cone_lit_benchmark_v1.csv`.
- **Components string:** `name|role|amount; ...`, where role is `matrix`, `FR`, `synergist` or `filler` (same format as the dataset).

**Accuracy.** Quote held-out numbers only, from `experiments/runs/litcone_gt_round1_*.log`:
- **P2:** random 5×10 cross-validation.
- **P3:** paper-grouped 5×10, which is harder because whole papers are unseen.

The in-sample / P1 numbers are training-data fits under the competitor's protocol, so they are not held-out evidence.
