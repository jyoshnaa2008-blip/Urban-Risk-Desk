"""
Urban Risk Desk - FastAPI backend
Place this file in the same folder as:
  nyc311_clean.csv
  priority_by_zip_type.csv
  results.json
  clusters_311.json
  map_coords.json
  model_311.joblib                 (optional but needed for /api/predict)
  scenarios.db                     (created automatically)

Run:
  python -m pip install -r requirements.txt
  uvicorn app_prepared:app --reload --host 127.0.0.1 --port 8000

This API reports observed dataset statistics and saved model results separately.
The scenario simulator is a transparent weighted-priority heuristic, not an ML model.
"""

from __future__ import annotations

import json
import math
import sqlite3
import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
LOG = logging.getLogger("urban-risk-desk")

# The project is split into functional folders. Each asset kind lists the
# directories that may contain it, in priority order, plus the numbered
# fallback filenames created when files are downloaded more than once.
ASSET_DIRS = {
    "dataset": ["data", "ml", BASE_DIR.name],
    "priority": ["data", "ml", BASE_DIR.name],
    "metrics": ["ml", "data", BASE_DIR.name],  # ml/ has the real results.json; data/ only has README.md
    "clusters": ["clustering", "ml", BASE_DIR.name],
    "coordinates": ["data", BASE_DIR.name],
    "model": ["ml", BASE_DIR.name],
    "html": ["frontend", BASE_DIR.name],
    "geojson": ["frontend", BASE_DIR.name],
    "lib": ["frontend", BASE_DIR.name],
}
FILE_CANDIDATES = {
    "dataset": ["nyc311_clean.csv"],
    "priority": ["priority_by_zip_type(1).csv", "priority_by_zip_type.csv"],
    "metrics": ["results(5).json", "results(4).json", "results.json"],
    "clusters": ["clusters_311_rebuilt.json", "clusters_311.json"],
    "coordinates": ["map_coords(1).json", "map_coords.json"],
    "model": ["model_311(4).joblib", "model_311.joblib"],
    "html": ["urban-risk-desk(5).html", "urban-risk-desk.html"],
    "geojson": ["nyc_borough_boundaries.geojson"],
}
DB_PATH = BASE_DIR / "scenarios.db"

def locate(kind: str) -> Optional[Path]:
    for filename in FILE_CANDIDATES[kind]:
        for directory in ASSET_DIRS[kind]:
            candidate = PROJECT_ROOT / directory / filename
            if candidate.exists():
                return candidate
    return None

def read_json(kind: str, default):
    path = locate(kind)
    if not path:
        return default
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        LOG.exception("Could not read %s", path.name)
        return default

app = FastAPI(
    title="Urban Risk Desk API",
    description="NYC 311 complaint delay analytics and transparent resource prioritization.",
    version="1.0.0",
)

# Development CORS. For deployment, replace "*" with the exact frontend origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the cleaned dataset once at startup. Only the columns needed by the API
# are loaded to reduce memory use.
DATA_COLUMNS = [
    "unique_key", "created_date", "closed_date", "agency", "complaint_type",
    "incident_zip", "status", "borough", "channel", "latitude", "longitude",
    "slow_48h",
]
df: Optional[pd.DataFrame] = None
priority_df: Optional[pd.DataFrame] = None
map_coords = []
model_bundle = None
model_load_error: Optional[str] = None

@app.on_event("startup")
def load_assets():
    global df, priority_df, map_coords, model_bundle, model_load_error

    dataset_path = locate("dataset")
    if dataset_path:
        try:
            header = pd.read_csv(dataset_path, nrows=0).columns.tolist()
            usecols = [c for c in DATA_COLUMNS if c in header]
            df = pd.read_csv(dataset_path, usecols=usecols, low_memory=False)
            for col in ["latitude", "longitude", "slow_48h"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            for col in ["incident_zip", "borough", "complaint_type", "agency", "channel", "status"]:
                if col in df.columns:
                    df[col] = df[col].fillna("Unknown").astype(str)
            LOG.info("Loaded %s rows from %s", f"{len(df):,}", dataset_path.name)
        except Exception:
            LOG.exception("Dataset load failed")
            df = None
    else:
        LOG.warning("nyc311_clean.csv not found beside app_prepared.py")

    
priority_path = locate("priority")
if priority_path:
    try:
        priority_df = pd.read_csv(
            priority_path,
            dtype={"incident_zip": str}
        )

        priority_df = priority_df.rename(columns={
            "incident_zip": "zip",
            "complaint_type": "pt",
            "complaints": "volume",
            "expected_slow": "avg_risk"
        })

        priority_df["zip"] = (
            priority_df["zip"]
            .astype(str)
            .str.replace(r"\.0$", "", regex=True)
        )

        for c in ["volume", "avg_risk", "actual_slow", "priority_score"]:
            if c in priority_df:
                priority_df[c] = pd.to_numeric(
                    priority_df[c], errors="coerce"
                )

        LOG.info("Loaded %s priority groups", len(priority_df))

    except Exception:
        LOG.exception("Priority CSV load failed")
        priority_df = None

    map_coords = read_json("coordinates", [])
    if not isinstance(map_coords, list):
        map_coords = []

    # joblib files are Python pickle-based artifacts.
    # Try loading the model with version warnings suppressed.
    model_path = locate("model")
    if model_path:
        try:
            # Suppress sklearn version warnings for model loading
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
                import joblib
                model_bundle = joblib.load(model_path)
                LOG.info("Loaded model bundle from %s", model_path.name)
                model_load_error = None
        except Exception as exc:
            LOG.exception("Model load failed: %s", exc)
            model_bundle = None
            model_load_error = f"Model loading failed: {type(exc).__name__}: {exc}"
    else:
        model_load_error = "Model file not found."


def init_scenarios_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS scenarios (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                crews_available INTEGER NOT NULL,
                heat_weight REAL NOT NULL,
                street_weight REAL NOT NULL,
                noise_weight REAL NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
init_scenarios_db()

def require_dataset() -> pd.DataFrame:
    if df is None:
        raise HTTPException(
            status_code=503,
            detail="Dataset unavailable. Put nyc311_clean.csv in the same folder as app_prepared.py and restart the server.",
        )
    return df

def clean_zip(value) -> str:
    if pd.isna(value):
        return "Unknown"
    return str(value).replace(".0", "").strip()

def get_recent_rate(column: str, value: str, fallback: float = 0.0) -> float:
    """Estimate a historical slow rate using a 7-day window with a 3-day outcome lag.

    This mirrors the feature timing in the training pipeline: only outcomes old
    enough to have been observed are included. If the available window has too
    few records, use the historical group rate as a documented fallback.
    """
    data = require_dataset()
    if column not in data.columns or "slow_48h" not in data.columns:
        return fallback
    if "created_date" not in data.columns:
        return fallback

    created = pd.to_datetime(data["created_date"], errors="coerce")
    slow = pd.to_numeric(data["slow_48h"], errors="coerce")
    now = pd.Timestamp.now()
    window_end = now - pd.Timedelta(days=3)
    window_start = now - pd.Timedelta(days=10)
    group_mask = data[column].astype(str).str.upper() == str(value).upper()
    time_mask = (created >= window_start) & (created < window_end)
    recent = slow[group_mask & time_mask].dropna()
    if len(recent) >= 20:
        return float(recent.mean())

    historical = slow[group_mask].dropna()
    return float(historical.mean()) if len(historical) else fallback

class PredictionRequest(BaseModel):
    complaint_type: str = Field(..., min_length=1)
    agency: str = Field(..., min_length=1)
    borough: str = Field(..., min_length=1)
    incident_zip: str = Field(..., min_length=1)
    channel: str = "PHONE"
    hour: int = Field(12, ge=0, le=23)
    dow: int = Field(2, ge=0, le=6)

class ScenarioInput(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    crews_available: int = Field(6, ge=0, le=10000)
    heat_weight: float = Field(100, ge=0, le=300)
    street_weight: float = Field(80, ge=0, le=300)
    noise_weight: float = Field(50, ge=0, le=300)

class SimulationInput(BaseModel):
    crews_available: int = Field(6, ge=0, le=10000)
    heat_weight: float = Field(100, ge=0, le=300)
    street_weight: float = Field(80, ge=0, le=300)
    noise_weight: float = Field(50, ge=0, le=300)

@app.get("/")
def root():
    html_path = locate("html")
    if html_path:
        return FileResponse(html_path, media_type="text/html")
    return {
        "service": "Urban Risk Desk API",
        "docs": "/docs",
        "message": "No dashboard HTML found. Open /docs to test the API.",
    }

@app.get("/nyc_borough_boundaries.geojson")
def borough_geojson():
    path = locate("geojson")
    if path:
        return FileResponse(path, media_type="application/geo+json")
    return {"error": "geojson not found"}

@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)

@app.get("/lib/{file_path:path}")
def serve_lib(file_path: str):
    """Serve local library assets (leaflet, chart.js, marker images) — no CDN/API key needed."""
    try:
        lib_dir = PROJECT_ROOT / "frontend" / "lib"
        if not lib_dir.is_dir():
            return JSONResponse({"error": "lib directory not found"}, status_code=404)
        full = (lib_dir / file_path).resolve()
        if not full.is_file():
            return JSONResponse({"error": "not found"}, status_code=404)
        return FileResponse(full)
    except Exception as exc:
        LOG.exception("serve_lib failed for %s", file_path)
        return JSONResponse({"error": "internal", "detail": str(exc)}, status_code=500)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "dataset_loaded": df is not None,
        "priority_data_loaded": priority_df is not None,
        "model_loaded": model_bundle is not None,
        "model_error": model_load_error,
    }

@app.get("/api/overview")
def overview():
    data = require_dataset()
    total = int(len(data))
    status = data["status"].astype(str).str.upper() if "status" in data else pd.Series([], dtype=str)
    open_mask = status.isin(["OPEN", "IN PROGRESS", "ASSIGNED", "PENDING"])
    slow = pd.to_numeric(data["slow_48h"], errors="coerce").dropna() if "slow_48h" in data else pd.Series(dtype=float)
    return {
        "total_complaints": total,
        "open_complaints": int(open_mask.sum()),
        "at_risk_pct": round(float(slow.mean() * 100), 2) if len(slow) else None,
        "available_crews": None,
        "dataset_note": "Available crews are not contained in the NYC 311 dataset.",
    }

@app.get("/api/borough-summary")
def borough_summary():
    data = require_dataset()
    if data is None or len(data) == 0:
        raise HTTPException(status_code=503, detail="Dataset unavailable.")
    df = data.copy()
    # Ensure borough column is clean
    if "borough" in df.columns:
        df["borough"] = df["borough"].fillna("Unknown").astype(str).str.title()
    else:
        raise HTTPException(status_code=503, detail="Borough column missing from dataset.")

    # Ensure slow_48h is numeric
    if "slow_48h" in df.columns:
        df["slow_48h"] = pd.to_numeric(df["slow_48h"], errors="coerce")

    # Ensure status is clean
    if "status" in df.columns:
        df["status"] = df["status"].fillna("Unknown").astype(str).str.upper()

    # Calculate open/in-progress
    open_statuses = ["OPEN", "IN PROGRESS", "ASSIGNED", "PENDING"]
    open_mask = df["status"].isin(open_statuses)

    summary = {}
    for borough in ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]:
        boro_data = df[df["borough"] == borough]
        total_boro = len(boro_data)
        slow_boro = pd.to_numeric(boro_data["slow_48h"], errors="coerce").dropna()
        slow_count = int(len(slow_boro[slow_boro == 1])) if len(slow_boro) else 0
        slow_rate = round(float(slow_count / len(slow_boro) * 100), 2) if len(slow_boro) else 0.0
        closed_count = int(((~open_mask) & (df["status"].astype(str) == "CLOSED")).sum()) if "status" in df.columns else 0
        # Count actual open/in-progress from this borough
        open_in_progress = int(open_mask[df["borough"] == borough].sum())
        closed = int((((~open_mask) & (df["status"] == "CLOSED")) & (df["borough"] == borough)).sum())

        # Most frequent complaint type
        if "complaint_type" in boro_data.columns:
            top_type = boro_data["complaint_type"].mode().iloc[0] if not boro_data["complaint_type"].mode().empty else "Unknown"
        else:
            top_type = "Unknown"

        # Top ZIP codes by volume
        zip_counts = boro_data["incident_zip"].value_counts().head(3).to_dict() if "incident_zip" in boro_data.columns else {}

        summary[borough] = {
            "total_complaints": total_boro,
            "open_complaints": open_in_progress,
            "closed_complaints": closed,
            "slow_complaints": slow_count,
            "observed_slow_rate_pct": slow_rate,
            "most_frequent_complaint_type": top_type,
            "top_zip_codes": [{"zip": k, "volume": v} for k, v in zip_counts.items()],
        }

    return {"boroughs": summary}

@app.get("/api/map-markers")
def map_markers(
    limit: int = Query(2000, ge=1, le=10000),
    borough: Optional[str] = None,
    complaint_type: Optional[str] = None,
):
    data = require_dataset()
    cols = [c for c in [
        "unique_key", "latitude", "longitude", "complaint_type",
        "borough", "incident_zip", "status", "slow_48h"
    ] if c in data.columns]
    points = data[cols].copy()
    points["latitude"] = pd.to_numeric(points["latitude"], errors="coerce")
    points["longitude"] = pd.to_numeric(points["longitude"], errors="coerce")
    points = points.dropna(subset=["latitude", "longitude"])
    if borough:
        points = points[points["borough"].str.upper() == borough.upper()]
    if complaint_type:
        points = points[points["complaint_type"].str.contains(complaint_type, case=False, na=False)]
    if len(points) > limit:
        points = points.sample(n=limit, random_state=42)
    points = points.replace({np.nan: None})
    return points.to_dict(orient="records")

@app.get("/api/priority")
def priority_queue(
    limit: int = Query(100, ge=1, le=1000),
    zip: Optional[str] = None,
    complaint_type: Optional[str] = None,
    heat_weight: float = Query(1.0, ge=0, le=3),
    street_weight: float = Query(0.8, ge=0, le=3),
    noise_weight: float = Query(0.5, ge=0, le=3),
):
    if priority_df is None:
        raise HTTPException(status_code=503, detail="priority_by_zip_type.csv not found or could not be read.")
    out = priority_df.copy()
    if zip:
        out = out[out["zip"].astype(str) == clean_zip(zip)]
    if complaint_type:
        out = out[out["pt"].astype(str).str.contains(complaint_type, case=False, na=False)]

    def impact_weight(pt: str) -> float:
        text = pt.upper()
        if "HEAT" in text:
            return heat_weight
        if "STREET" in text:
            return street_weight
        if "NOISE" in text:
            return noise_weight
        return 1.0

    out["impact_weight"] = out["pt"].astype(str).map(impact_weight)
    # Transparent policy score: observed volume x saved average delay-risk x user-set impact weight.
    out["base_priority_score"] = out["volume"] * out["avg_risk"]
    out["adjusted_priority_score"] = out["base_priority_score"] * out["impact_weight"]
    # Add borough labels from the complaint dataset where ZIP-to-borough mapping exists.
    data = require_dataset()
    if "incident_zip" in data.columns and "borough" in data.columns:
        zip_borough = data[["incident_zip", "borough"]].copy()
        zip_borough["incident_zip"] = zip_borough["incident_zip"].map(clean_zip)
        zip_borough = zip_borough[zip_borough["incident_zip"] != "Unknown"]
        zip_borough = zip_borough.groupby("incident_zip")["borough"].agg(
            lambda values: values.mode().iloc[0] if not values.mode().empty else "Unknown"
        )
        out["borough"] = out["zip"].map(zip_borough).fillna("Unknown")
    out = out.sort_values("adjusted_priority_score", ascending=False).head(limit)
    out = out.replace({np.nan: None})
    return out.to_dict(orient="records")

@app.get("/api/metrics")
def metrics():
    data = read_json("metrics", {})
    if not data:
        raise HTTPException(status_code=503, detail="Model metrics JSON not found.")
    return data

@app.get("/api/clusters")
def clusters():
    data = read_json("clusters", {})
    if not data:
        raise HTTPException(status_code=503, detail="Cluster results JSON not found.")
    # Keep original keys and provide a compatibility alias for clients expecting it.
    if isinstance(data, dict) and "profiles" in data and "cluster_profiles" not in data:
        data["cluster_profiles"] = data["profiles"]
    return data

@app.get("/api/map-coordinates")
def coordinates():
    return map_coords

@app.post("/api/predict")
def predict(request: PredictionRequest):
    if model_bundle is None:
        raise HTTPException(
            status_code=503,
            detail=f"Prediction model unavailable: {model_load_error or 'model not loaded'}.",
        )
    model = model_bundle["model"]
    features = list(model_bundle["features"])

    # These historical rates are calculated from the cleaned dataset, not fabricated constants.
    pt_rate = get_recent_rate("complaint_type", request.complaint_type, 0.0)
    ag_rate = get_recent_rate("agency", request.agency, 0.0)
    row = {
        "pt": request.complaint_type.upper(),
        "ag": request.agency.upper(),
        "b": request.borough.upper(),
        "ch": request.channel.upper(),
        "zip": clean_zip(request.incident_zip),
        "hour": request.hour,
        "dow": request.dow,
        "pt_rr": pt_rate,
        "ag_rr": ag_rate,
    }
    missing = [feature for feature in features if feature not in row]
    if missing:
        raise HTTPException(status_code=500, detail=f"Model expects unsupported features: {missing}")
    try:
        input_frame = pd.DataFrame([{feature: row[feature] for feature in features}])
        probability = float(model.predict_proba(input_frame)[:, 1][0])
    except Exception as exc:
        LOG.exception("Model prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    tier = "High" if probability >= 0.66 else "Medium" if probability >= 0.45 else "Low"
    return {
        "probability": round(probability, 4),
        "prob": round(probability, 4),  # compatibility with the earlier frontend
        "tier": tier,
        "color": {"High": "magenta", "Medium": "yellow", "Low": "green"}[tier],
        "pt_historical_slow_rate": round(pt_rate, 4),
        "agency_historical_slow_rate": round(ag_rate, 4),
        "target": "Complaint remains unresolved beyond 48 hours",
    }

def scenario_rows():
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, name, crews_available, heat_weight, street_weight, noise_weight, created_at "
            "FROM scenarios ORDER BY id DESC"
        ).fetchall()
    return [dict(row) for row in rows]

@app.get("/api/scenarios")
def list_scenarios():
    return scenario_rows()

@app.post("/api/scenarios")
def create_scenario(scenario: ScenarioInput):
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            "INSERT INTO scenarios (name, crews_available, heat_weight, street_weight, noise_weight) "
            "VALUES (?, ?, ?, ?, ?)",
            (scenario.name, scenario.crews_available, scenario.heat_weight,
             scenario.street_weight, scenario.noise_weight),
        )
        scenario_id = cur.lastrowid
        conn.commit()
    return {"id": scenario_id, "message": "Scenario saved"}

@app.put("/api/scenarios/{scenario_id}")
def update_scenario(scenario_id: int, scenario: ScenarioInput):
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            "UPDATE scenarios SET name=?, crews_available=?, heat_weight=?, street_weight=?, noise_weight=? WHERE id=?",
            (scenario.name, scenario.crews_available, scenario.heat_weight,
             scenario.street_weight, scenario.noise_weight, scenario_id),
        )
        conn.commit()
    if cur.rowcount == 0:
        raise HTTPException(status_code=404, detail="Scenario not found")
    return {"id": scenario_id, "message": "Scenario updated"}

@app.delete("/api/scenarios/{scenario_id}")
def delete_scenario(scenario_id: int):
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute("DELETE FROM scenarios WHERE id=?", (scenario_id,))
        conn.commit()
    if cur.rowcount == 0:
        raise HTTPException(status_code=404, detail="Scenario not found")
    return {"id": scenario_id, "message": "Scenario deleted"}

@app.post("/api/scenarios/simulate")
@app.post("/api/simulate")
def simulate(scenario: SimulationInput):
    if priority_df is None:
        raise HTTPException(status_code=503, detail="Priority data unavailable.")
    # This is a policy-allocation simulation based on the saved priority table,
    # not a prediction of real-world crew productivity or response outcomes.
    rows = priority_queue(
        limit=1000,
        heat_weight=scenario.heat_weight / 100.0,
        street_weight=scenario.street_weight / 100.0,
        noise_weight=scenario.noise_weight / 100.0,
    )
    total_score = sum(max(0.0, float(r.get("adjusted_priority_score") or 0)) for r in rows)
    remaining = scenario.crews_available
    allocations = []
    if total_score > 0 and remaining > 0:
        raw = [scenario.crews_available * max(0.0, float(r.get("adjusted_priority_score") or 0)) / total_score for r in rows]
        base = [math.floor(x) for x in raw]
        left = scenario.crews_available - sum(base)
        order = sorted(range(len(raw)), key=lambda i: raw[i] - base[i], reverse=True)
        for i in order[:left]:
            base[i] += 1
        for row, crews in zip(rows, base):
            if crews:
                allocations.append({
                    "zip": row["zip"],
                    "complaint_type": row["pt"],
                    "priority_score": row["adjusted_priority_score"],
                    "allocated_crews": crews,
                })
    return {
        "crews_available": scenario.crews_available,
        "allocated_crews": sum(x["allocated_crews"] for x in allocations),
        "allocations": allocations,
        "method": "Proportional allocation by transparent weighted priority score",
        "warning": "Illustrative decision-support scenario only; not a validated operational staffing model.",
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app_prepared:app", host="127.0.0.1", port=8000, reload=True)