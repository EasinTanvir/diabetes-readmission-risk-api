# Diabetes 30-day readmission risk API

Ranks discharged diabetic patients by their risk of readmission within 30 days, and flags the
top ~10% (`risk_score >= cutoff`, where the cutoff is the 90th percentile of validation scores).

> **`risk_score` is for ranking only. It is not a probability.** The model is trained with
> `class_weight="balanced"`, which inflates the scores. Compare patients with each other; do not
> read 0.40 as "40% chance".

Model: HistGradientBoostingClassifier trained on the UCI "Diabetes 130-US Hospitals" data
(see `CLAUDE.md` and `eval/` for how it was rebuilt and evaluated). Validation: ROC-AUC 0.670,
average precision 0.230; at the cutoff, recall 0.24, precision 0.28, lift 2.4x.

## Setup (Windows)

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

`model/pipeline.pkl` and `model/schema_values.json` must be present. To rebuild them:
`.venv\Scripts\python src\train.py`, then `.venv\Scripts\python src\build_schema_values.py`.
Retraining is not needed to run the API.

## Run

```powershell
.venv\Scripts\uvicorn src.api:app --port 8000
```

Interactive docs: http://127.0.0.1:8000/docs

```powershell
curl.exe http://127.0.0.1:8000/health
curl.exe -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" --data "@examples/predict_request.json"
```

## Endpoints

- `GET /health`: status, `model_version` (first 12 characters of the sha256 of pipeline.pkl), `cutoff`, library versions.
- `POST /predict`: body is a JSON list of patients (at most 1000). Each patient is validated on its own.
  A bad row gets an `error`, and the other rows are still scored. Each result has:
  `id` (patient_nbr or encounter_id, logging only), `risk_score`, `flagged`, `warnings`, `error`.

## Input fields

All 31 model fields use the same names and values as `diabetic_data.csv`. Fields marked
"Unknown" may be `"?"`, `null` or left out. All other fields are required.

| Field | Rule |
|---|---|
| race | Caucasian, AfricanAmerican, Hispanic, Asian, Other; Unknown allowed |
| gender | Female, Male, Unknown/Invalid (-> most common training value, with a warning) |
| age | `[0-10)` ... `[90-100)` exactly |
| admission_type_id, discharge_disposition_id, admission_source_id | integer codes from IDS_mapping.csv (`"1"` accepted). Discharge 11, 13, 14, 19, 20, 21 (died/hospice) are rejected: out of model scope |
| time_in_hospital | integer 1-14 |
| num_lab_procedures, num_procedures, num_medications, number_outpatient, number_emergency, number_inpatient, number_diagnoses | integer >= 0 (warning above the training maximum) |
| payer_code, medical_specialty | any text; a value not seen in training -> `Other` with a warning; Unknown allowed |
| diag_1, diag_2, diag_3 | ICD-9 code (`428`, `250.83`, `V45`, `E880`); spaces and case are normalized; a number is read as a code; junk -> Unknown with a warning |
| max_glu_serum | `>200`, `>300`, `Norm`; `"None"`/null/missing = not tested |
| A1Cresult | `>7`, `>8`, `Norm`; `"None"`/null/missing = not tested |
| metformin, repaglinide, glimepiride, glipizide, glyburide, pioglitazone, rosiglitazone, insulin | `No`, `Steady`, `Up`, `Down` |
| change | `Ch`, `No` |
| diabetesMed | `Yes`, `No` (exact, so `yes` is rejected) |

The other raw CSV columns (weight and 15 unused drugs) are accepted and ignored. Any other
unknown field is ignored, with a warning. Every warning is also logged by the `readmission` logger.

## Tests

```powershell
.venv\Scripts\python -m pytest -q
```

Covers every robustness case (each must give an error or a logged fallback, never a silent
score), batches with bad rows, and the API. The suite also checks that all 14,911 validation
rows give exactly the same scores as the evaluated model.
