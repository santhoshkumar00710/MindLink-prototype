"""
MindLink AI – FastAPI Backend
==============================
Endpoints:
  POST /api/decode          – Run the full BCI pipeline
  POST /api/execute         – Execute an intent command
  POST /api/feedback        – Store user correction
  GET  /api/eeg             – Return simulated EEG data (JSON)
  GET  /health              – Liveness check
"""

import sys
import os
from pathlib import Path

# Allow importing sibling packages from project root
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.eeg_simulator import (
    simulate_eeg,
    compute_power_spectrum,
    CHANNEL_LABELS,
    N_CHANNELS,
    SAMPLING_RATE,
)
from backend.thought_decoder import decode_thought, save_feedback
from backend.executor import execute_command
from utils.signal_processing import compute_all_band_powers

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="MindLink AI",
    description="Virtual Brain Computer Interface API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request / Response models ────────────────────────────────────────────────

class ThoughtRequest(BaseModel):
    thought: str

class FeedbackRequest(BaseModel):
    thought: str
    predicted: str
    corrected: str

class ExecuteRequest(BaseModel):
    intent: str


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "MindLink AI"}


@app.post("/api/decode")
def api_decode(req: ThoughtRequest):
    """Full pipeline: thought → EEG simulation → model → intent + confidence."""
    if not req.thought.strip():
        raise HTTPException(status_code=400, detail="thought must not be empty")
    result = decode_thought(req.thought)
    return result


@app.post("/api/execute")
def api_execute(req: ExecuteRequest):
    """Execute an intent command and return status."""
    return execute_command(req.intent)


@app.post("/api/feedback")
def api_feedback(req: FeedbackRequest):
    """Store user correction for adaptive learning."""
    save_feedback(req.thought, req.predicted, req.corrected)
    return {"status": "saved"}


@app.get("/api/eeg")
def api_eeg(thought: str = "open browser"):
    """
    Return simulated EEG signals as JSON-serialisable lists,
    along with power spectrum and band powers.
    """
    eeg, time_axis, pattern_key = simulate_eeg(thought)
    freqs, psd = compute_power_spectrum(eeg)
    band_powers = compute_all_band_powers(eeg)

    return {
        "channels": CHANNEL_LABELS,
        "sampling_rate": SAMPLING_RATE,
        "time": time_axis.tolist(),
        "eeg": eeg.tolist(),           # shape [8, 256]
        "freqs": freqs.tolist(),
        "psd": psd.tolist(),
        "band_powers": band_powers,
        "pattern_key": pattern_key,
    }
