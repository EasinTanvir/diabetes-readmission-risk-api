"""API tests with FastAPI's TestClient (runs the lifespan, so the model loads once)."""
import json

import pytest
from fastapi.testclient import TestClient

from conftest import ROOT
from src.api import MAX_ROWS, app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client, scorer):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["cutoff"] == scorer.cutoff
    assert body["model_version"] == scorer.model_version
    assert body["versions"]["sklearn"] == "1.6.1"
    assert "not a probability" in body["note"]


def test_predict_example(client, scorer):
    patients = json.loads((ROOT / "examples" / "predict_request.json").read_text())
    r = client.post("/predict", json=patients)
    assert r.status_code == 200
    body = r.json()
    assert body["n_scored"] == 2 and body["n_errors"] == 1
    clean, fallback, invalid = body["results"]

    assert clean["risk_score"] == scorer.score([patients[0]])[0].risk_score
    assert clean["warnings"] == [] and clean["error"] is None
    assert clean["id"] == patients[0]["patient_nbr"]
    assert any("Astrology" in w for w in fallback["warnings"])
    assert any("number 428" in w for w in fallback["warnings"])
    assert invalid["risk_score"] is None and invalid["flagged"] is None
    for f in ["age", "diabetesMed", "time_in_hospital"]:
        assert f + ":" in invalid["error"]


def test_never_called_probability(client):
    patients = json.loads((ROOT / "examples" / "predict_request.json").read_text())
    text = client.post("/predict", json=patients).text.replace("it is not a probability", "")
    assert "probab" not in text.lower()


def test_bad_bodies(client):
    assert client.post("/predict", json=[]).status_code == 422
    assert client.post("/predict", json={"race": "Caucasian"}).status_code == 422
    assert client.post("/predict", json=[{}] * (MAX_ROWS + 1)).status_code == 413
    r = client.post("/predict", json=[42, "x", None])
    assert r.status_code == 200 and r.json()["n_errors"] == 3
