"""
CardioPredict AI — Inference Utilities (v2 — Stacking Ensemble)
===============================================================
Loaded once at Flask startup via lru_cache.

Maps the rich frontend Assessment form → UCI-derived feature vector
→ StackingClassifier → risk score + SHAP explanation.

Optional advanced fields the frontend can POST for higher accuracy:
  cp, exang, oldpeak, ca, thal, restecg, thalach
"""

import json, time
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import shap

# ── Config (inline — no external import needed) ──────────────────────────────
from pathlib import Path as _P
_BASE      = _P(__file__).parent
_ART       = _BASE / "artifacts"

MODEL_FILE         = _ART / "stack_model.joblib"
SCALER_FILE        = _ART / "scaler.joblib"
BACKGROUND_FILE    = _ART / "shap_background.joblib"
FEATURE_NAMES_FILE = _ART / "feature_names.json"
METRICS_FILE       = _ART / "metrics.json"
MODEL_VER          = "2.0.0"

FEATURE_DISPLAY_NAMES = {
    "age":"Age", "sex":"Sex", "cp":"Chest Pain Type",
    "trestbps":"Resting Blood Pressure", "chol":"Cholesterol",
    "fbs":"Fasting Blood Sugar", "restecg":"Resting ECG",
    "thalach":"Max Heart Rate", "exang":"Exercise-Induced Angina",
    "oldpeak":"ST Depression", "slope":"ST Slope",
    "ca":"Major Vessels (CA)", "thal":"Thalassemia",
    "age_chol_ratio":"Age × Cholesterol",
    "bp_heart_load":"BP × Heart Rate Load",
    "exang_oldpeak_inter":"Exang × ST Depression",
    "age_thalach_deficit":"HR Reserve Deficit",
    "chol_age_norm":"Age-Normalised Cholesterol",
    "cp_sex_inter":"Chest Pain × Sex",
    "ca_thal_score":"CA × Thalassemia Score",
    "risk_cluster":"Combined Risk Cluster",
    "hr_reserve_ratio":"HR Reserve Ratio",
    "bp_chol_product":"BP × Cholesterol",
    "oldpeak_slope_inter":"ST Depression × Slope",
}


# ═══════════════════════════════════════════════════════════════════════════
# ARTIFACT LOADING
# ═══════════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=1)
def _load_artifacts() -> dict:
    """Load model + scaler + metadata once; cached for all requests."""
    try:
        model         = joblib.load(MODEL_FILE)
        scaler        = joblib.load(SCALER_FILE)
        background    = joblib.load(BACKGROUND_FILE)
        feature_names = json.loads(Path(FEATURE_NAMES_FILE).read_text())
        metrics       = json.loads(Path(METRICS_FILE).read_text())
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Artifact missing: {exc}. Run train_model.py first.") from exc

    # Build SHAP explainer using XGBoost base learner (fast TreeExplainer)
    # The stacking model exposes its named steps via .named_estimators_
    xgb_model = None
    try:
        xgb_model = model.named_estimators_["xgb"]
        booster   = xgb_model.get_booster()
        explainer = shap.TreeExplainer(booster)
    except Exception:
        # Fallback: KernelExplainer on full stacking model
        explainer = shap.KernelExplainer(
            model.predict_proba, background[:50])

    return {
        "model": model, "scaler": scaler, "explainer": explainer,
        "background": background, "feature_names": feature_names,
        "metrics": metrics,
    }


def get_feature_names(): return _load_artifacts()["feature_names"]
def get_metrics():       return _load_artifacts()["metrics"]


# ═══════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING  (must mirror train_model.py exactly)
# ═══════════════════════════════════════════════════════════════════════════

def _engineer(base: dict) -> list:
    """Add the same engineered features as train_model.py."""
    age      = base["age"]
    chol     = base["chol"]
    trestbps = base["trestbps"]
    thalach  = base["thalach"]
    exang    = base["exang"]
    oldpeak  = base["oldpeak"]
    slope    = base["slope"]
    cp       = base["cp"]
    sex      = base["sex"]
    ca       = base["ca"]
    thal     = base["thal"]
    fbs      = base["fbs"]

    age_chol_ratio      = age * chol
    bp_heart_load       = trestbps * thalach
    exang_oldpeak_inter = exang * oldpeak
    age_thalach_deficit = (220 - age) - thalach
    chol_age_norm       = chol / (age + 1)
    cp_sex_inter        = cp * sex
    ca_thal_score       = ca * thal
    risk_cluster        = (int(exang) + int(fbs)
                           + int(cp == 4) + int(thal == 7))
    hr_reserve_ratio    = thalach / (220 - age + 1)
    bp_chol_product     = trestbps * chol / 10_000
    oldpeak_slope_inter = oldpeak * slope

    return [
        age, sex, cp, trestbps, chol, fbs, base["restecg"], thalach, exang,
        oldpeak, slope, ca, thal,
        # engineered
        age_chol_ratio, bp_heart_load, exang_oldpeak_inter,
        age_thalach_deficit, chol_age_norm, cp_sex_inter, ca_thal_score,
        risk_cluster, hr_reserve_ratio, bp_chol_product, oldpeak_slope_inter,
    ]


# ═══════════════════════════════════════════════════════════════════════════
# FORM → FEATURE VECTOR
# ═══════════════════════════════════════════════════════════════════════════

def map_form_to_features(form: dict) -> np.ndarray:
    """Convert Assessment form JSON → feature vector (1, n_features)."""
    def _int(k, d):   return int(form.get(k) or d)
    def _float(k, d): return float(form.get(k) or d)
    def _str(k, d):   return str(form.get(k) or d).lower()

    age         = _int("age", 50)
    sex         = 1 if _str("sex","male") == "male" else 0
    sbp         = _int("sbp", 120)
    chol        = _int("totalCholesterol", 200)
    blood_sugar = _int("bloodSugar", 100)
    heart_rate  = _int("heartRate", 70)
    weight      = _float("weight", 70.0)
    height      = _float("height", 170.0)
    bmi         = weight / ((height/100)**2) if height > 0 else 25.0

    diabetes     = _str("diabetesHistory","no") == "yes"
    hypertension = _str("hypertension","no") == "yes"
    prev_cardiac = _str("previousCardiac","no") == "yes"
    family_hist  = _str("familyHistory","no") == "yes"
    obesity      = _str("obesityStatus","no") == "yes"
    smoking      = _str("smokingStatus","never")
    activity     = _str("physicalActivity","moderate")
    stress       = _str("stressLevel","moderate")
    diet         = _str("dietQuality","fair")

    risk_points = (
        int(diabetes) + int(hypertension) + int(smoking == "current")
        + 0.5*int(smoking == "former") + int(family_hist) + int(prev_cardiac)
        + 0.5*int(obesity or bmi >= 30) + 0.5*int(stress == "high")
        + 0.5*int(activity in ["sedentary","low"])
        + 0.5*int(diet in ["poor","fair"])
    )

    # UCI fields — use direct value if provided by clinician
    def _direct(key, fallback):
        v = form.get(key)
        return int(v) if v is not None and v != "" else fallback

    if "cp" in form and form["cp"] is not None:
        cp = int(form["cp"])
    elif prev_cardiac:                     cp = 4
    elif hypertension or stress == "high": cp = 2
    else:                                  cp = 3

    trestbps = sbp
    fbs      = int(blood_sugar > 120 or diabetes)

    if "restecg" in form and form["restecg"] is not None:
        restecg = int(form["restecg"])
    elif hypertension and (age > 50 or bmi >= 30): restecg = 2
    elif prev_cardiac:                              restecg = 1
    else:                                           restecg = 0

    if "thalach" in form and form["thalach"] is not None:
        thalach = int(form["thalach"])
    else:
        mult    = {"sedentary":0.58,"low":0.68,"moderate":0.78,"high":0.88}.get(activity, 0.75)
        thalach = min(int((220-age)*mult), 202)

    if "exang" in form and form["exang"] is not None:
        exang = int(form["exang"])
    else:
        exang = int(prev_cardiac or
                    (activity in ["sedentary","low"] and (hypertension or diabetes)))

    if "oldpeak" in form and form["oldpeak"] is not None:
        oldpeak = float(form["oldpeak"])
    else:
        oldpeak = min(risk_points*0.45 + max(0, age-45)*0.03, 6.0)

    if "slope" in form and form["slope"] is not None:
        slope = int(form["slope"])
    else:
        slope = 1 if oldpeak < 0.5 else (2 if oldpeak < 2.0 else 3)

    if "ca" in form and form["ca"] is not None:
        ca = int(form["ca"])
    else:
        ca = min(int(risk_points*0.55), 3)

    if "thal" in form and form["thal"] is not None:
        thal = int(form["thal"])
    elif prev_cardiac and family_hist: thal = 7
    elif prev_cardiac:                 thal = 6
    else:                              thal = 3

    base = dict(age=age, sex=sex, cp=cp, trestbps=trestbps, chol=chol,
                fbs=fbs, restecg=restecg, thalach=thalach, exang=exang,
                oldpeak=oldpeak, slope=slope, ca=ca, thal=thal)

    vec = _engineer(base)
    return np.array(vec, dtype=float).reshape(1, -1)


def _align_features(X, feature_names):
    exp, act = len(feature_names), X.shape[1]
    if act == exp:   return X
    if act < exp:    return np.hstack([X, np.zeros((1, exp-act))])
    return X[:, :exp]


# ═══════════════════════════════════════════════════════════════════════════
# SHAP PER SAMPLE
# ═══════════════════════════════════════════════════════════════════════════

def compute_shap(X_scaled: np.ndarray, feature_names: list) -> list:
    explainer = _load_artifacts()["explainer"]
    raw_sv    = explainer.shap_values(X_scaled)

    if isinstance(raw_sv, list):
        sv = np.array(raw_sv[1]).squeeze()
    else:
        sv = np.array(raw_sv).squeeze()
    if sv.ndim == 2:
        sv = sv[0] if sv.shape[0] == 1 else sv[:, 1]
    sv = sv.flatten()[:len(feature_names)]

    items = []
    for i, (name, val) in enumerate(zip(feature_names, sv)):
        disp = FEATURE_DISPLAY_NAMES.get(name, name.replace("_"," ").title())
        items.append({
            "name":      disp,
            "impact":    round(abs(float(val)), 4),
            "direction": "increases" if val > 0 else "decreases",
            "value":     round(float(X_scaled[0, i]), 4),
        })
    items.sort(key=lambda x: x["impact"], reverse=True)
    return items[:7]


# ═══════════════════════════════════════════════════════════════════════════
# RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════

_RECS = {
    "High": [
        "Immediate referral to a cardiologist is strongly recommended.",
        "Initiate lipid-lowering therapy assessment (statin consideration).",
        "Lifestyle modification plan: structured cardiac rehabilitation.",
        "Continuous BP monitoring and antihypertensive therapy review.",
        "Schedule stress ECG and echocardiogram within 2 weeks.",
    ],
    "Moderate": [
        "Schedule follow-up cardiovascular assessment within 3 months.",
        "Counsel on dietary changes: Mediterranean-style diet recommended.",
        "Encourage 150+ minutes of moderate aerobic activity per week.",
        "Monitor HbA1c and fasting glucose every 6 months.",
        "Consider preventive statin therapy evaluation with physician.",
    ],
    "Low": [
        "Continue current healthy lifestyle habits.",
        "Annual cardiovascular screening recommended.",
        "Maintain blood pressure below 120/80 mmHg.",
        "Keep LDL cholesterol below 100 mg/dL.",
        "Annual physical activity assessment encouraged.",
    ],
}


# ═══════════════════════════════════════════════════════════════════════════
# MAIN INFERENCE
# ═══════════════════════════════════════════════════════════════════════════

def run_inference(form_data: dict) -> dict:
    arts          = _load_artifacts()
    model         = arts["model"]
    scaler        = arts["scaler"]
    feature_names = arts["feature_names"]

    X_raw    = map_form_to_features(form_data)
    X_raw    = _align_features(X_raw, feature_names)
    X_scaled = scaler.transform(X_raw)

    risk_score = float(model.predict_proba(X_scaled)[0, 1])
    risk_level = "High" if risk_score >= 0.70 else ("Moderate" if risk_score >= 0.40 else "Low")
    confidence = min(0.99, 0.70 + abs(risk_score - 0.50) * 0.6)

    try:
        top_features = compute_shap(X_scaled, feature_names)
    except Exception:
        top_features = []

    from datetime import datetime, timezone
    return {
        "id":           f"pred_{int(time.time()*1000)}",
        "risk_score":   round(risk_score, 4),
        "risk_level":   risk_level,
        "confidence":   round(confidence, 4),
        "timestamp":    datetime.now(timezone.utc).isoformat(),
        "patient_id":   str(form_data.get("patientId","PT-UNKNOWN")),
        "top_features": top_features,
        "recommendations": _RECS[risk_level],
        "model_version":   MODEL_VER,
        "model_type":      "StackingClassifier (XGB+LGB+RF+ET+SVC → LR)",
        "data_source":     "Live ML API v2",
    }
