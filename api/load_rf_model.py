from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
import joblib
import numpy as np
from pathlib import Path

app = FastAPI()

model = joblib.load("rf-model/turnover_model_with_mock.pkl")
EXPECTED_FEATURES = int(getattr(model, "n_features_in_", 0))
MODEL_FEATURE_NAMES = [str(name) for name in getattr(model, "feature_names_in_", [])]
BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR.parent / "web"
MOCKUP_HTML = WEB_DIR / "index.html"
RECOMMENDATION_HTML = WEB_DIR / "recommendation.html"


def to_camel_case(name: str) -> str:
    parts = str(name).replace("-", "_").split("_")
    if not parts:
        return ""
    head = parts[0]
    tail = "".join(part[:1].upper() + part[1:] for part in parts[1:] if part)
    return f"{head}{tail}"


def resolved_model_feature_names() -> list[str]:
    if MODEL_FEATURE_NAMES:
        return MODEL_FEATURE_NAMES
    if EXPECTED_FEATURES > 0:
        return [f"feature_{i}" for i in range(1, EXPECTED_FEATURES + 1)]
    return []


RESOLVED_MODEL_FEATURE_NAMES = resolved_model_feature_names()
API_FEATURE_NAMES = [to_camel_case(name) for name in RESOLVED_MODEL_FEATURE_NAMES]

def risk_level(prob):
    if prob < 0.3:
        return "Low"
    elif prob < 0.6:
        return "Medium"
    else:
        return "High"


def to_feature_vector(data: dict) -> np.ndarray:
    # Accept list payloads and named-feature payloads for readable API input.
    if "features" in data:
        raw_features = data["features"]
    elif all(
        key in data
        for key in [
            "job_satisfaction",
            "organizational_commitment",
            "work_stress",
        ]
    ):
        raw_features = [
            data["job_satisfaction"],
            data["organizational_commitment"],
            data["work_stress"],
        ]
    else:
        raise HTTPException(
            status_code=422,
            detail=(
                "Invalid payload. Send {'features': {...}} with named keys or "
                "{'features': [...]} with ordered values."
            ),
        )

    if isinstance(raw_features, dict):
        values = []
        missing = []
        for model_name, api_name in zip(RESOLVED_MODEL_FEATURE_NAMES, API_FEATURE_NAMES):
            if api_name in raw_features:
                values.append(raw_features[api_name])
            elif model_name in raw_features:
                # Also allow snake_case model field names for convenience.
                values.append(raw_features[model_name])
            else:
                missing.append(api_name)

        if missing:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Missing required feature fields.",
                    "missing": missing,
                    "expected_keys": API_FEATURE_NAMES,
                },
            )

        raw_features = values

    if not isinstance(raw_features, list):
        raise HTTPException(
            status_code=422,
            detail="'features' must be either a JSON object or JSON array.",
        )

    try:
        features = [float(x) for x in raw_features]
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=422,
            detail="All feature values must be numeric.",
        )

    if EXPECTED_FEATURES and len(features) != EXPECTED_FEATURES:
        raise HTTPException(
            status_code=400,
            detail={
                "message": f"Model expects {EXPECTED_FEATURES} features but received {len(features)}.",
                "expected_keys": API_FEATURE_NAMES,
            },
        )

    return np.array(features, dtype=float).reshape(1, -1)

@app.get("/")
def root():
    return {"message": "API is running"}


@app.get("/ui")
def ui_mockup():
    return FileResponse(MOCKUP_HTML)


@app.get("/recommendation")
def recommendation_page():
    return FileResponse(RECOMMENDATION_HTML)


@app.get("/model-info")
def model_info():
    return {
        "expected_features": EXPECTED_FEATURES,
        "model_feature_names": RESOLVED_MODEL_FEATURE_NAMES,
        "api_feature_names": API_FEATURE_NAMES,
        "request_example": {
            "features": {name: 0.0 for name in API_FEATURE_NAMES}
        },
    }

@app.post("/predict")
def predict(data: dict):
    features = to_feature_vector(data)

    prob = model.predict_proba(features)[0][1]

    return {
        "risk_score": float(prob),
        "risk_level": risk_level(prob)
    }


# careerGrowth: โอกาสเติบโตในสายอาชีพ
# jobSatisfaction: ความพึงพอใจในงาน
# organizationalCommitment: ความผูกพันต่อองค์กร
# compensation: ความพึงพอใจด้านค่าตอบแทน
# training: ความเพียงพอ/คุณภาพของการอบรม
# managementSupport: การสนับสนุนจากหัวหน้า/ผู้บริหาร
# skillMatch: งานตรงกับทักษะที่มี
# roleClarity: ความชัดเจนของบทบาทหน้าที่
# majorMatch: ความตรงระหว่างสาขาที่เรียนกับงาน