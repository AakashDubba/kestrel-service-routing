"""
Kestrel Home Appliances — Service Request Routing API
======================================================
app.py — FastAPI application

Loads the trained routing_model.pkl on startup.
Endpoints:
  GET  /health   → health check
  POST /predict  → {"text": "..."} → {"team": "...", "reason": "..."}
  GET  /         → serves the frontend (index.html)
"""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import joblib
import os

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "routing_model.pkl")
INDEX_PATH = os.path.join(BASE_DIR, "index.html")

# ──────────────────────────────────────────────
# App Initialization
# ──────────────────────────────────────────────
app = FastAPI(
    title="Kestrel Service Routing API",
    description="Routes customer service requests to the correct team using a locally-trained ML model.",
    version="1.0.0",
)

# Load model at startup
model = joblib.load(MODEL_PATH)

VALID_TEAMS = [
    "Billing",
    "Filters & Consumables",
    "Installs & Demo",
    "Product Advice",
    "Repairs",
    "Returns & Replacement",
    "Warranty Claims",
]

TEAM_DESCRIPTIONS = {
    "Billing": "Invoices, GST, double charges, refunds, EMI conversion — problems with the payment itself.",
    "Filters & Consumables": "Filters, candles, membranes, jars, brushes, blades, AMC kits — selling and fitting spares.",
    "Installs & Demo": "New-product installation, demo, and wall-mounting visits.",
    "Product Advice": "Pre- and post-purchase usage questions with no fault reported.",
    "Repairs": "Product faults, breakdowns, error codes, noise, leaks — anything needing a technician.",
    "Returns & Replacement": "Damaged, wrong, or incomplete deliveries; returns and exchanges within the return window.",
    "Warranty Claims": "Warranty and Kestrel Shield registration, coverage questions, claim status.",
}


# ──────────────────────────────────────────────
# Request/Response Models
# ──────────────────────────────────────────────
class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    team: str
    reason: str


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the frontend HTML page."""
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "valid_teams": VALID_TEAMS,
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Predict the routing team for a customer service request."""
    text = request.text.strip()
    if not text:
        return PredictResponse(
            team="Unknown",
            reason="No text provided. Please enter a customer service request.",
        )

    predicted_team = model.predict([text])[0]
    team_desc = TEAM_DESCRIPTIONS.get(predicted_team, "")

    reason = (
        f"This request was routed to {predicted_team} based on patterns learned "
        f"from 10,822 historically resolved service requests. "
        f"{team_desc}"
    )

    return PredictResponse(team=predicted_team, reason=reason)
