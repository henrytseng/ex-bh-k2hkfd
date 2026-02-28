import json
import uuid
from typing import Any, Dict, List, Literal, Optional
from uuid import UUID

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

app = FastAPI(
    title="Call Outcome Prediction API",
    description="API to predict outcomes of calls based on metadata and events.",
)

# ---- In-memory model registry ----

MODEL_REGISTRY: Dict[str, dict] = {}
ACTIVE_MODEL_ID: Optional[str] = None

# ---- Feature definitions ----

CATEGORICAL_FEATURES = [
    "agent_id", "call_purpose",
    "caller_phone_type", "time_of_day", "day_of_week",
]
NUMERICAL_FEATURES = [
    "total_duration",
    "agent_speech_count",
    "user_speech_count",
    "silence_count",
    "tool_call_count",
    "total_agent_ms",
    "total_user_ms",
    "total_silence_ms",
    "total_agent_words",
    "total_user_words",
    "turn_count",
    "silence_ratio",
    "user_speech_ratio",
    "avg_user_words_per_turn",
    "avg_agent_words_per_turn",
    "survey_completion_rate",
]
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES


# ---- Feature extraction ----

def extract_features(events: list, metadata: dict, survey_completion_rate: float = 0.0) -> dict:
    """
    Extract features from a normalized event list and a metadata dict.

    Events must be pre-normalized via normalize_events() so that:
      - Timestamp key is "timestamp" (integer seconds)
      - Speech events have type "agent_speech" or "user_speech" with flat
        "duration_ms" (int) and "words" (int) fields

    Silence is computed from gaps between consecutive speech events rather
    than from explicit silence-type events (which do not exist in the schema).
    """
    ts_start = next((e["timestamp"] for e in events if e["type"] == "call_start"), 0)
    ts_end = next((e["timestamp"] for e in events if e["type"] == "call_end"), None)
    total_duration = float(ts_end - ts_start) if ts_end is not None else 0.0

    agent_speeches = [e for e in events if e["type"] == "agent_speech"]
    user_speeches  = [e for e in events if e["type"] == "user_speech"]
    tool_calls     = [e for e in events if e["type"] == "tool_call"]

    total_agent_ms    = sum(e.get("duration_ms", 0) for e in agent_speeches)
    total_user_ms     = sum(e.get("duration_ms", 0) for e in user_speeches)
    total_agent_words = sum(e.get("words", 0) for e in agent_speeches)
    total_user_words  = sum(e.get("words", 0) for e in user_speeches)

    # Compute silence from gaps between consecutive speech events.
    # Each speech event occupies [ts, ts + duration_ms/1000] seconds.
    # Any positive gap to the next event's start is silence.
    SILENCE_THRESHOLD_MS = 500
    speech_events = sorted(
        agent_speeches + user_speeches,
        key=lambda e: e["timestamp"],
    )
    total_silence_ms = 0.0
    silence_count = 0
    for i in range(len(speech_events) - 1):
        prev_end_s   = speech_events[i]["timestamp"] + speech_events[i].get("duration_ms", 0) / 1000.0
        next_start_s = speech_events[i + 1]["timestamp"]
        gap_ms = (next_start_s - prev_end_s) * 1000.0
        if gap_ms > 0:
            total_silence_ms += gap_ms
            if gap_ms >= SILENCE_THRESHOLD_MS:
                silence_count += 1

    total_audio_ms = total_agent_ms + total_user_ms + total_silence_ms
    silence_ratio = total_silence_ms / total_audio_ms if total_audio_ms > 0 else 0.0
    user_speech_ratio = (
        total_user_ms / (total_agent_ms + total_user_ms)
        if (total_agent_ms + total_user_ms) > 0 else 0.0
    )

    # Count speaker turns (each agent↔user alternation = 1 turn)
    turn_count = 0
    prev_speaker: Optional[str] = None
    for e in speech_events:
        speaker = "agent" if e["type"] == "agent_speech" else "user"
        if speaker != prev_speaker:
            turn_count += 1
            prev_speaker = speaker

    return {
        # Categorical
        "agent_id":          metadata.get("agent_id", "unknown"),
        "call_purpose":      metadata.get("call_purpose", "unknown"),
        "caller_phone_type": metadata.get("caller_phone_type", "unknown"),
        "time_of_day":       metadata.get("time_of_day", "unknown"),
        "day_of_week":       metadata.get("day_of_week", "unknown"),
        # Numerical
        "total_duration":           total_duration,
        "agent_speech_count":       float(len(agent_speeches)),
        "user_speech_count":        float(len(user_speeches)),
        "silence_count":            float(silence_count),
        "tool_call_count":          float(len(tool_calls)),
        "total_agent_ms":           float(total_agent_ms),
        "total_user_ms":            float(total_user_ms),
        "total_silence_ms":         total_silence_ms,
        "total_agent_words":        float(total_agent_words),
        "total_user_words":         float(total_user_words),
        "turn_count":               float(turn_count),
        "silence_ratio":            silence_ratio,
        "user_speech_ratio":        user_speech_ratio,
        "avg_user_words_per_turn":  float(total_user_words / len(user_speeches))  if user_speeches  else 0.0,
        "avg_agent_words_per_turn": float(total_agent_words / len(agent_speeches)) if agent_speeches else 0.0,
        "survey_completion_rate":   float(survey_completion_rate),
    }


def feature_dict_to_arrays(feature_dicts: list, cat_encoder: OrdinalEncoder):
    """Convert a list of feature dicts into a single (n_samples, n_features) numpy array."""
    cat_rows = [[fd[f] for f in CATEGORICAL_FEATURES] for fd in feature_dicts]
    num_rows = [[fd[f] for f in NUMERICAL_FEATURES]  for fd in feature_dicts]
    X_cat = cat_encoder.transform(cat_rows)
    X_num = np.array(num_rows, dtype=np.float64)
    return np.hstack([X_cat, X_num])


def normalize_events(events: list) -> list:
    """
    Normalize events from any input format to the internal format expected by
    extract_features().

    Handles three source schemas:
      - Training data: {"timestamp": int, "type": "speech_detected",
                        "data": {"speaker": "agent"|"user", "duration_ms": int, "words": int}}
      - API (CallEvent): {"timestamp": int, "type": "agent_speech"|"user_speech",
                          "data": {"speaker": str, "duration_ms": int, "words": int}}
      - Passthrough: call_start, call_end, tool_call events with no data payload

    All events are emitted with a "timestamp" key and flat duration_ms/words fields.
    """
    normalized = []
    for e in events:
        ts = e.get("timestamp", e.get("ts", 0))
        etype = e["type"]

        if etype == "speech_detected":
            data = e.get("data") or {}
            speaker = data.get("speaker", "agent")
            normalized.append({
                "timestamp": ts,
                "type": "agent_speech" if speaker == "agent" else "user_speech",
                "duration_ms": data.get("duration_ms", 0),
                "words": data.get("words", 0),
            })
        elif etype in ("agent_speech", "user_speech"):
            data = e.get("data") or {}
            normalized.append({
                "timestamp": ts,
                "type": etype,
                "duration_ms": data.get("duration_ms", e.get("duration_ms", 0)),
                "words": data.get("words", e.get("words", 0)),
            })
        else:
            # Passthrough (call_start, call_end, tool_call)
            e_copy = {k: v for k, v in e.items() if k not in ("ts", "data")}
            normalized.append({**e_copy, "timestamp": ts})

    return normalized


def api_events_to_raw(api_events: list) -> list:
    """Convert API CallEvent Pydantic objects to normalized dicts for extract_features."""
    return normalize_events([e.model_dump() for e in api_events])


# ---- Data Models ----

class CallData(BaseModel):
    speaker: str
    duration_ms: int
    words: int


class CallEvent(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    timestamp: int
    type: str
    data: Optional[CallData] = None


class CallMetadata(BaseModel):
    agent_id: str
    org_id: Optional[str] = None
    call_purpose: str
    caller_phone_type: Optional[str] = "unknown"
    time_of_day: Optional[str] = "unknown"
    day_of_week: Optional[str] = "unknown"


CallOutcome = Literal["completed", "abandoned", "transferred", "error"]


# ---- Train Endpoint Models ----

class TrainingRequest(BaseModel):
    training_data_path: str


class TrainingMetrics(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1: float


class TrainingResponse(BaseModel):
    model_id: str
    metrics: TrainingMetrics


# ---- Predict Endpoint Models ----

class PredictionRequest(BaseModel):
    call_id: UUID
    events_so_far: List[CallEvent]
    metadata: CallMetadata


class TopFactor(BaseModel):
    feature: str
    impact: float
    value: float | int


class PredictionResponse(BaseModel):
    predicted_outcome: CallOutcome
    confidence: float
    risk_score: float = Field(..., description="Probability of failure (1 - P(completed))")
    top_factors: List[TopFactor]


# ---- Feature Importance Models ----

class FeatureImportance(BaseModel):
    feature: str
    importance: float


class FeatureImportanceResponse(BaseModel):
    model_id: str
    rankings: List[FeatureImportance]


# ---- Endpoints ----

@app.post("/api/model/train", response_model=TrainingResponse)
async def train_model(request: TrainingRequest):
    """
    Train a RandomForest classifier on historical call data.

    Extracts the following feature groups from each call:
    - Categorical: agent_id, call_purpose, caller_phone_type,
                   time_of_day, day_of_week
    - Numerical: duration, speech counts/durations/words, silence stats,
                 turn-taking metrics, tool call count, survey completion rate
    """
    global ACTIVE_MODEL_ID

    try:
        with open(request.training_data_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Training data file not found: {request.training_data_path}",
        )
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON in training file: {exc}")

    calls = data["calls"] if isinstance(data, dict) and "calls" in data else data
    if not calls:
        raise HTTPException(status_code=400, detail="Training data contains no calls.")

    # --- Feature extraction ---
    feature_dicts: list = []
    outcomes: list = []
    for call in calls:
        fd = extract_features(
            normalize_events(call["events"]),
            call["metadata"],
            call.get("survey_completion_rate", 0.0),
        )
        feature_dicts.append(fd)
        outcomes.append(call["outcome"])

    # --- Build feature matrix ---
    cat_rows = [[fd[f] for f in CATEGORICAL_FEATURES] for fd in feature_dicts]
    num_rows = [[fd[f] for f in NUMERICAL_FEATURES]  for fd in feature_dicts]

    cat_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_cat = cat_encoder.fit_transform(cat_rows)
    X_num = np.array(num_rows, dtype=np.float64)
    X = np.hstack([X_cat, X_num])

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(outcomes)

    # --- Train / test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Fit model ---
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    # --- Evaluate ---
    y_pred = clf.predict(X_test)
    metrics = TrainingMetrics(
        accuracy=float(accuracy_score(y_test, y_pred)),
        precision=float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        recall=float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        f1=float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
    )

    # --- Feature importances ---
    feature_importance = [
        {"feature": name, "importance": float(imp)}
        for name, imp in sorted(
            zip(ALL_FEATURES, clf.feature_importances_), key=lambda x: -x[1]
        )
    ]

    # Store normalisation stats for computing per-prediction impacts
    feature_means = X_train.mean(axis=0)
    feature_stds  = X_train.std(axis=0) + 1e-8

    # --- Register model ---
    model_id = f"model_{uuid.uuid4().hex[:8]}"
    MODEL_REGISTRY[model_id] = {
        "clf":                clf,
        "cat_encoder":        cat_encoder,
        "label_encoder":      label_encoder,
        "feature_importance": feature_importance,
        "feature_means":      feature_means,
        "feature_stds":       feature_stds,
        "metrics":            metrics,
    }
    ACTIVE_MODEL_ID = model_id

    return TrainingResponse(model_id=model_id, metrics=metrics)


@app.post("/api/predict", response_model=PredictionResponse)
async def predict_outcome(request: PredictionRequest):
    """
    Predict the outcome of an ongoing call based on events so far and metadata.
    Uses the most recently trained model.
    """
    if ACTIVE_MODEL_ID is None or ACTIVE_MODEL_ID not in MODEL_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail="No trained model available. Call POST /api/model/train first.",
        )

    m = MODEL_REGISTRY[ACTIVE_MODEL_ID]
    clf: RandomForestClassifier = m["clf"]
    cat_encoder: OrdinalEncoder   = m["cat_encoder"]
    label_encoder: LabelEncoder   = m["label_encoder"]

    # Convert API event format → raw dict format
    raw_events = api_events_to_raw(request.events_so_far)
    meta = request.metadata.model_dump()
    fd = extract_features(raw_events, meta)

    X = feature_dict_to_arrays([fd], cat_encoder)  # shape (1, n_features)

    proba = clf.predict_proba(X)[0]
    pred_idx = int(np.argmax(proba))
    predicted_outcome: str = label_encoder.inverse_transform([pred_idx])[0]
    confidence = float(proba[pred_idx])

    # risk_score = 1 - P(completed)
    classes = list(label_encoder.classes_)
    risk_score = (
        1.0 - float(proba[classes.index("completed")])
        if "completed" in classes
        else 1.0 - confidence
    )

    # Top factors: importance × normalised feature value → signed impact
    feature_vals = X[0]
    impacts = clf.feature_importances_ * (feature_vals - m["feature_means"]) / m["feature_stds"]
    top_indices = np.argsort(np.abs(impacts))[::-1][:5]
    top_factors = [
        TopFactor(
            feature=ALL_FEATURES[i],
            impact=round(float(impacts[i]), 4),
            value=round(float(feature_vals[i]), 4),
        )
        for i in top_indices
    ]

    return PredictionResponse(
        predicted_outcome=predicted_outcome,
        confidence=round(confidence, 4),
        risk_score=round(risk_score, 4),
        top_factors=top_factors,
    )


@app.get("/api/model/{model_id}/importance", response_model=FeatureImportanceResponse)
async def get_feature_importance(model_id: str):
    """Return global feature importance rankings for a trained model."""
    if model_id not in MODEL_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")

    rankings = [
        FeatureImportance(feature=fi["feature"], importance=fi["importance"])
        for fi in MODEL_REGISTRY[model_id]["feature_importance"]
    ]
    return FeatureImportanceResponse(model_id=model_id, rankings=rankings)


def main():
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
