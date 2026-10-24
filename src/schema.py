"""Input validation for one raw patient row (pydantic v2).

validate_patient(record) -> (raw_row, warnings, errors)
  raw_row  the 31 model input fields in the same form as diabetic_data.csv
           ("?" for unknown, NaN for a test not done, ints for IDs and counts),
           ready for src.prep.prepare(). None when the row is rejected.
  warnings deliberate fallbacks (e.g. unseen specialty -> "Other"), each also logged.
  errors   "field: message" strings. A rejected row is never scored.

Allowed values come from the training split and IDS_mapping.csv (model/schema_values.json).
patient_nbr / encounter_id are accepted for logging only and never reach the model.
"""
import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, ValidationInfo, field_validator, model_validator

log = logging.getLogger("readmission")

ROOT = Path(__file__).resolve().parent.parent
VALUES = json.loads((ROOT / "model" / "schema_values.json").read_text(encoding="utf-8"))

UNKNOWN = "?"                     # how the source data writes "unknown"
NOT_TESTED = float("nan")         # how pandas reads "None" in max_glu_serum / A1Cresult
DIED_OR_HOSPICE = {11, 13, 14, 19, 20, 21}
DIAG_PATTERN = re.compile(r"(V\d+|E\d+|\d+)(\.\d+)?")

INPUT_FIELDS = ["race", "gender", "age", "admission_type_id", "discharge_disposition_id",
                "admission_source_id", "time_in_hospital", "payer_code", "medical_specialty",
                "num_lab_procedures", "num_procedures", "num_medications", "number_outpatient",
                "number_emergency", "number_inpatient", "diag_1", "diag_2", "diag_3",
                "number_diagnoses", "max_glu_serum", "A1Cresult", "metformin", "repaglinide",
                "glimepiride", "glipizide", "glyburide", "pioglitazone", "rosiglitazone",
                "insulin", "change", "diabetesMed"]
INT_FIELDS = ["admission_type_id", "discharge_disposition_id", "admission_source_id",
              "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
              "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses"]
ID_FIELDS = ["admission_type_id", "discharge_disposition_id", "admission_source_id"]
LOG_ID_FIELDS = ["patient_nbr", "encounter_id"]

Drug = Literal["No", "Steady", "Up", "Down"]
AgeGroup = Literal["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
                   "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"]


def _is_missing(v):
    return v is None or (isinstance(v, float) and math.isnan(v)) or (isinstance(v, str) and v.strip() == "")


def _warn(info: ValidationInfo, msg):
    info.context["warnings"].append(msg)


class Patient(BaseModel):
    # extra keys are allowed so they can be reported as warnings, then dropped
    model_config = ConfigDict(extra="allow")

    patient_nbr: Optional[int | str] = None
    encounter_id: Optional[int | str] = None

    # "?", None, missing -> Unknown (as in training)
    race: Any = Field(default=None, validate_default=True)
    payer_code: Any = Field(default=None, validate_default=True)
    medical_specialty: Any = Field(default=None, validate_default=True)
    diag_1: Any = Field(default=None, validate_default=True)
    diag_2: Any = Field(default=None, validate_default=True)
    diag_3: Any = Field(default=None, validate_default=True)
    # "None", None, missing -> NotTested
    max_glu_serum: Any = Field(default=None, validate_default=True)
    A1Cresult: Any = Field(default=None, validate_default=True)

    # required, exact values
    gender: Literal["Female", "Male", "Unknown/Invalid"]
    age: AgeGroup
    metformin: Drug
    repaglinide: Drug
    glimepiride: Drug
    glipizide: Drug
    glyburide: Drug
    pioglitazone: Drug
    rosiglitazone: Drug
    insulin: Drug
    change: Literal["Ch", "No"]
    diabetesMed: Literal["Yes", "No"]

    # required integers ("3" accepted, 3.5 / "abc" / true rejected)
    admission_type_id: int
    discharge_disposition_id: int
    admission_source_id: int
    time_in_hospital: int = Field(ge=1, le=14)
    num_lab_procedures: int = Field(ge=0)
    num_procedures: int = Field(ge=0)
    num_medications: int = Field(ge=0)
    number_outpatient: int = Field(ge=0)
    number_emergency: int = Field(ge=0)
    number_inpatient: int = Field(ge=0)
    number_diagnoses: int = Field(ge=0)

    @field_validator(*INT_FIELDS, mode="before")
    @classmethod
    def no_bool(cls, v):
        if isinstance(v, bool):
            raise ValueError("must be an integer, not true/false")
        return v

    @field_validator(*VALUES["count_max_in_training"], mode="after")
    @classmethod
    def above_training_max(cls, v, info: ValidationInfo):
        top = VALUES["count_max_in_training"][info.field_name]
        if v > top:
            _warn(info, f"{info.field_name}: {v} is above the training maximum ({top}), scored anyway")
        return v

    @field_validator("gender", mode="after")
    @classmethod
    def gender_fallback(cls, v, info: ValidationInfo):
        if v == "Unknown/Invalid":
            _warn(info, "gender: 'Unknown/Invalid' -> most common training value")
        return v

    @field_validator(*ID_FIELDS, mode="after")
    @classmethod
    def known_id_code(cls, v, info: ValidationInfo):
        col = info.field_name
        if v not in VALUES["id_codes"][col]:
            raise ValueError(f"code {v} is not in IDS_mapping.csv")
        if col == "discharge_disposition_id" and v in DIED_OR_HOSPICE:
            raise ValueError(f"code {v} means died or hospice: out of model scope (not in training)")
        if v not in VALUES["id_codes_seen_in_training"][col]:
            _warn(info, f"{col}: code {v} is valid but never seen in training")
        return v

    @field_validator("race", mode="before")
    @classmethod
    def check_race(cls, v, info: ValidationInfo):
        if _is_missing(v):
            _warn(info, "race: missing -> Unknown")
            return UNKNOWN
        if v == UNKNOWN or v in VALUES["race"]:
            return v
        raise ValueError(f"{v!r} is not a known race; allowed: {VALUES['race']} or '?'")

    @field_validator("medical_specialty", "payer_code", mode="before")
    @classmethod
    def check_open_category(cls, v, info: ValidationInfo):
        col = info.field_name
        if _is_missing(v):
            _warn(info, f"{col}: missing -> Unknown")
            return UNKNOWN
        if not isinstance(v, str):
            raise ValueError("must be text")
        if v == UNKNOWN or v in VALUES[col]:
            return v
        _warn(info, f"{col}: {v!r} not seen in training -> Other")
        return "Other"

    @field_validator("diag_1", "diag_2", "diag_3", mode="before")
    @classmethod
    def check_diag(cls, v, info: ValidationInfo):
        col = info.field_name
        if _is_missing(v):
            _warn(info, f"{col}: missing -> Unknown")
            return UNKNOWN
        if isinstance(v, bool):
            _warn(info, f"{col}: {v!r} is not a diagnosis code -> Unknown")
            return UNKNOWN
        if isinstance(v, (int, float)):
            _warn(info, f"{col}: number {v!r} read as code {str(v)!r}")
            v = str(v)
        if not isinstance(v, str):
            _warn(info, f"{col}: {v!r} is not a diagnosis code -> Unknown")
            return UNKNOWN
        code = v.strip().upper()
        if code == UNKNOWN:
            return UNKNOWN
        if not DIAG_PATTERN.fullmatch(code):
            _warn(info, f"{col}: {v!r} is not a valid ICD-9 code -> Unknown")
            return UNKNOWN
        if code != v:
            _warn(info, f"{col}: {v!r} normalized to {code!r}")
        return code

    @field_validator("max_glu_serum", "A1Cresult", mode="before")
    @classmethod
    def check_lab(cls, v, info: ValidationInfo):
        col = info.field_name
        allowed = [">200", ">300", "Norm"] if col == "max_glu_serum" else [">7", ">8", "Norm"]
        if _is_missing(v) or v == "None":
            return NOT_TESTED
        if v == UNKNOWN:
            _warn(info, f"{col}: '?' -> NotTested")
            return NOT_TESTED
        if v in allowed:
            return v
        raise ValueError(f"{v!r} is not allowed; use one of {allowed}, 'None' or null")

    @model_validator(mode="after")
    def report_extra(self, info: ValidationInfo):
        for k in (self.model_extra or {}):
            if k not in VALUES["unused_raw_columns"]:
                _warn(info, f"{k}: unknown field ignored")
        return self

    def raw_row(self):
        return {f: getattr(self, f) for f in INPUT_FIELDS}

    def row_id(self):
        return self.patient_nbr if self.patient_nbr is not None else self.encounter_id


def _format_error(e):
    loc = ".".join(str(p) for p in e["loc"]) or "row"
    msg = e["msg"]
    if e["type"] == "missing":
        msg = "required field is missing"
    elif e["input"] is None and e["type"] != "value_error":
        msg = "required field is null"
    return f"{loc}: {msg}"


def validate_patient(record):
    """Validate one raw record. Returns (raw_row | None, warnings, errors, row_id)."""
    row_id = None
    if isinstance(record, dict):
        row_id = next((record[k] for k in LOG_ID_FIELDS if record.get(k) is not None), None)
    else:
        return None, [], [f"row: must be a JSON object, got {type(record).__name__}"], None

    ctx = {"warnings": []}
    try:
        p = Patient.model_validate(record, context=ctx)
    except ValidationError as exc:
        return None, ctx["warnings"], [_format_error(e) for e in exc.errors()], row_id
    for w in ctx["warnings"]:
        log.warning("row %s: %s", p.row_id(), w)
    return p.raw_row(), ctx["warnings"], [], p.row_id()
