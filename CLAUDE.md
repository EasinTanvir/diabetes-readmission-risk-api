# Diabetes 30-day readmission model

Ranks discharged diabetic patients by risk of readmission within 30 days and flags the top ~10%.
Data: UCI "Diabetes 130-US Hospitals" (diabetic_data.csv, IDS_mapping.csv). Original work: Diabetes.ipynb.

## Rules
- Never modify diabetic_data.csv, IDS_mapping.csv or Diabetes.ipynb. They are read-only sources.
- Never retrain or re-save model/pipeline.pkl without an explicit request. Its sha256 is in
  eval/reference_pipeline_sha256.txt.
- Code lives in src/, tests in tests/, evaluation scripts and outputs in eval/, the model in model/.
- Use .venv (Python 3.12, versions pinned in requirements.txt). Do not upgrade libraries without
  re-running eval/03_data.py, eval/05_validation.py and the test suite.
- The test set has been used. Never tune, select or change anything based on test results.
- Anything learned from data (drop_cols, keep-lists, gender_mode, columns, schema category lists)
  comes from TRAINING rows only. The cutoff comes from validation. Never fit on val or test.
- The score is for ranking only (class_weight="balanced" inflates it). Never call it or present it
  as a probability. The API field is "risk_score".
- Splits are by patient_nbr only. Any cross-validation must use GroupKFold with patient groups.
- prepare() must not change its input and must match eval/notebook_replica.py exactly on clean data.
- Schema, group_diag or scoring changes must keep eval/03_data.py and tests/test_clean_scores.py
  passing: clean validation rows must give bit-identical scores to eval/reference_val_scores.npy.

## Reference numbers (validation, cutoff 0.6411, top 10%)
ROC-AUC 0.670, avg precision 0.230, recall 0.242, precision 0.282, lift 2.42x,
confusion [[12100,1071],[1319,421]]. Any change of more than 0.01 means something broke.

## Commands
```
.venv\Scripts\python eval\03_data.py            # data + prepare() vs notebook check
.venv\Scripts\python eval\05_validation.py      # validation metrics
.venv\Scripts\python -m pytest -q               # schema, scoring, clean-score and API tests
.venv\Scripts\uvicorn src.api:app --port 8000   # run the API
```
