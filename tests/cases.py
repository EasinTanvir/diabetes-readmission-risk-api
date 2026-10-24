"""The robustness cases from eval/07_robustness.py, each with its required outcome.

Outcome:
  ("error", field)     row rejected, and the error message names the field
  ("warning", text)    row scored, and a warning containing text is returned (and logged)
  ("ok", None)         a legitimate input: scored with no warning
"""
from src.schema import INPUT_FIELDS

MISSING = object()   # marker: remove the field from the row

CASES = [
    # unseen / wrong category values
    ("race unseen", {"race": "Martian"}, ("error", "race")),
    ("medical_specialty unseen", {"medical_specialty": "Astrology"}, ("warning", "medical_specialty: 'Astrology' not seen in training -> Other")),
    ("payer_code unseen", {"payer_code": "ZZ"}, ("warning", "payer_code: 'ZZ' not seen in training -> Other")),
    ("gender M", {"gender": "M"}, ("error", "gender")),
    ("gender Unknown/Invalid", {"gender": "Unknown/Invalid"}, ("warning", "gender: 'Unknown/Invalid'")),
    ("admission_type_id 99", {"admission_type_id": 99}, ("error", "admission_type_id")),
    ("admission_type_id '1' string", {"admission_type_id": "1"}, ("ok", None)),
    ("admission_type_id 1.5", {"admission_type_id": 1.5}, ("error", "admission_type_id")),
    ("admission_type_id 'abc'", {"admission_type_id": "abc"}, ("error", "admission_type_id")),
    ("discharge died (11)", {"discharge_disposition_id": 11}, ("error", "discharge_disposition_id")),
    ("discharge hospice (13)", {"discharge_disposition_id": 13}, ("error", "discharge_disposition_id")),
    ("discharge 30 valid, unseen", {"discharge_disposition_id": 30}, ("warning", "never seen in training")),
    ("A1Cresult '8'", {"A1Cresult": "8"}, ("error", "A1Cresult")),
    ("insulin 'Increased'", {"insulin": "Increased"}, ("error", "insulin")),
    ("metformin 'up' lowercase", {"metformin": "up"}, ("error", "metformin")),
    ("diabetesMed 'yes' lowercase", {"diabetesMed": "yes"}, ("error", "diabetesMed")),
    ("change 'Yes'", {"change": "Yes"}, ("error", "change")),
    # glucose / A1C not-tested spellings
    ("max_glu_serum 'None'", {"max_glu_serum": "None"}, ("ok", None)),
    ("max_glu_serum None", {"max_glu_serum": None}, ("ok", None)),
    ("max_glu_serum '?'", {"max_glu_serum": "?"}, ("warning", "max_glu_serum: '?' -> NotTested")),
    ("max_glu_serum '>200'", {"max_glu_serum": ">200"}, ("ok", None)),
    # diagnosis codes
    ("diag V45", {"diag_1": "V45"}, ("ok", None)),
    ("diag E880", {"diag_1": "E880"}, ("ok", None)),
    ("diag 250.83", {"diag_1": "250.83"}, ("ok", None)),
    ("diag '?'", {"diag_1": "?"}, ("ok", None)),
    ("diag v45 lowercase", {"diag_1": "v45"}, ("warning", "diag_1: 'v45' normalized to 'V45'")),
    ("diag ' 428 ' spaces", {"diag_1": " 428 "}, ("warning", "normalized to '428'")),
    ("diag ''", {"diag_1": ""}, ("warning", "diag_1: missing -> Unknown")),
    ("diag 'abc'", {"diag_1": "abc"}, ("warning", "diag_1: 'abc' is not a valid ICD-9 code -> Unknown")),
    ("diag None", {"diag_1": None}, ("warning", "diag_1: missing -> Unknown")),
    ("diag 428 number", {"diag_1": 428}, ("warning", "diag_1: number 428 read as code '428'")),
    ("diag 250.01 float", {"diag_1": 250.01}, ("warning", "read as code '250.01'")),
    ("diag true", {"diag_1": True}, ("warning", "is not a diagnosis code -> Unknown")),
    # ages, ranges, types
    ("age [100-110)", {"age": "[100-110)"}, ("error", "age")),
    ("age 55 number", {"age": 55}, ("error", "age")),
    ("number_inpatient -1", {"number_inpatient": -1}, ("error", "number_inpatient")),
    ("number_inpatient -3", {"number_inpatient": -3}, ("error", "number_inpatient")),
    ("time_in_hospital -5", {"time_in_hospital": -5}, ("error", "time_in_hospital")),
    ("time_in_hospital 0", {"time_in_hospital": 0}, ("error", "time_in_hospital")),
    ("time_in_hospital 15", {"time_in_hospital": 15}, ("error", "time_in_hospital")),
    ("time_in_hospital 3.5", {"time_in_hospital": 3.5}, ("error", "time_in_hospital")),
    ("time_in_hospital true", {"time_in_hospital": True}, ("error", "time_in_hospital")),
    ("num_medications 10000", {"num_medications": 10000}, ("warning", "num_medications: 10000 is above the training maximum")),
    ("number_emergency '2' string", {"number_emergency": "2"}, ("ok", None)),
    ("time_in_hospital '3' string", {"time_in_hospital": "3"}, ("ok", None)),
    ("extra unknown column", {"foo": "bar"}, ("warning", "foo: unknown field ignored")),
    ("unused raw column weight", {"weight": "?"}, ("ok", None)),
]

# '?', None and missing, one field at a time, over all 31 model input fields
UNKNOWN_OK = ["race", "payer_code", "medical_specialty", "diag_1", "diag_2", "diag_3"]
LABS = ["max_glu_serum", "A1Cresult"]

for f in INPUT_FIELDS:
    for label, value in [("'?'", "?"), ("None", None), ("missing", MISSING)]:
        if f in UNKNOWN_OK:
            expect = ("ok", None) if value == "?" else ("warning", f"{f}: missing -> Unknown")
        elif f in LABS:
            expect = ("warning", f"{f}: '?' -> NotTested") if value == "?" else ("ok", None)
        else:
            expect = ("error", f)
        CASES.append((f"{f} {label}", {f: value}, expect))


def apply(base, change):
    row = dict(base)
    for k, v in change.items():
        if v is MISSING:
            row.pop(k, None)
        else:
            row[k] = v
    return row
