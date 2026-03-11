"""
Initialize Model Weights Script
================================
Run once to save an untrained (random init) model to models/eegnet_model.pth.
The thought_decoder pipeline works correctly with random weights because it
blends model output with a deterministic rule-based prior.

For real training, fine-tune on labelled EEG data using this script as a
starting point.

Usage:
    python init_model.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import torch
from backend.intent_model import MindLinkModel

MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODELS_DIR / "eegnet_model.pth"

if MODEL_PATH.exists():
    print(f"✅ Model already exists at {MODEL_PATH}")
else:
    model = MindLinkModel()
    torch.save(model.state_dict(), str(MODEL_PATH))
    print(f"✅ Saved initialised model to {MODEL_PATH}")

# Quick forward-pass sanity check
model = MindLinkModel()
model.load_state_dict(torch.load(str(MODEL_PATH), map_location="cpu"))
model.eval()
dummy = torch.randn(1, 8, 256)
logits, embed = model(dummy)
print(f"✅ Forward pass OK – logits: {logits.shape}, embedding: {embed.shape}")
