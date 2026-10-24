"""FastAPI service: rank discharged diabetic patients by 30-day readmission risk.

Run:  .venv\\Scripts\\uvicorn src.api:app --port 8000

risk_score is for ranking only (class_weight="balanced" inflates it). It is not a probability.
"""
import logging
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.scoring import SCORE_NOTE, Scorer

MAX_ROWS = 1000

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("readmission")

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model once, before the app accepts requests
    state["scorer"] = Scorer()
    log.info("model %s loaded, cutoff %.4f", state["scorer"].model_version, state["scorer"].cutoff)
    yield
    state.clear()


app = FastAPI(title="Diabetes readmission risk ranking", version="1.0.0", lifespan=lifespan)


class Health(BaseModel):
    status: str
    model_version: str
    cutoff: float
    versions: dict
    note: str


class PatientResult(BaseModel):
    id: Optional[Any] = None
    risk_score: Optional[float] = None
    flagged: Optional[bool] = None
    warnings: list[str] = []
    error: Optional[str] = None


class PredictResponse(BaseModel):
    model_version: str
    cutoff: float
    note: str
    n_scored: int
    n_errors: int
    results: list[PatientResult]


@app.get("/health", response_model=Health)
def health():
    s = state["scorer"]
    return Health(status="ok", model_version=s.model_version, cutoff=s.cutoff,
                  versions=s.versions, note=SCORE_NOTE)


# list[Any], not list[Patient]: each row is validated on its own, so one bad row
# returns an error for that row only instead of rejecting the whole request
@app.post("/predict", response_model=PredictResponse)
def predict(patients: list[Any]):
    if not patients:
        raise HTTPException(status_code=422, detail="send a non-empty list of patients")
    if len(patients) > MAX_ROWS:
        raise HTTPException(status_code=413, detail=f"at most {MAX_ROWS} patients per request")
    s = state["scorer"]
    results = [PatientResult(**vars(r)) for r in s.score(patients)]
    n_errors = sum(r.error is not None for r in results)
    log.info("scored %d rows, %d errors", len(results) - n_errors, n_errors)
    return PredictResponse(model_version=s.model_version, cutoff=s.cutoff, note=SCORE_NOTE,
                           n_scored=len(results) - n_errors, n_errors=n_errors, results=results)
