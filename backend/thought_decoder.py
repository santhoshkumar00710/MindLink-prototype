"""
Thought Decoder
===============
Full inference pipeline:
  thought string → EEG simulation → normalise → PyTorch model → intent + confidence

Also handles adaptive feedback: stores corrections to a JSON log and uses them
to bias future predictions (lightweight online learning).
"""

import json
import os
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

from backend.eeg_simulator import simulate_eeg, map_thought_to_key
from backend.intent_model import MindLinkModel, N_CLASSES
from utils.signal_processing import normalize_eeg

# ─── Constants ───────────────────────────────────────────────────────────────
INTENTS: List[str] = [
    "open_browser",
    "scroll_down",
    "type_hello",
    "play_music",
    "stop_action",
]

INTENT_LABELS: Dict[str, str] = {
    "open_browser": "OPEN_BROWSER",
    "scroll_down":  "SCROLL_DOWN",
    "type_hello":   "TYPE_HELLO",
    "play_music":   "PLAY_MUSIC",
    "stop_action":  "STOP_ACTION",
}

MODEL_PATH  = Path(__file__).parent.parent / "models" / "eegnet_model.pth"
FEEDBACK_LOG = Path(__file__).parent.parent / "models" / "feedback_log.json"

# ─── Model singleton ─────────────────────────────────────────────────────────
_model: MindLinkModel | None = None


def _get_model() -> MindLinkModel:
    global _model
    if _model is None:
        _model = MindLinkModel()
        if MODEL_PATH.exists():
            _model.load_state_dict(torch.load(str(MODEL_PATH), map_location="cpu"))
        _model.eval()
    return _model


# ─── Feedback store ──────────────────────────────────────────────────────────

def _load_feedback() -> List[Dict]:
    if FEEDBACK_LOG.exists():
        with open(FEEDBACK_LOG, "r") as f:
            return json.load(f)
    return []


def save_feedback(thought: str, predicted: str, corrected: str) -> None:
    """Persist a correction event to the feedback log."""
    FEEDBACK_LOG.parent.mkdir(parents=True, exist_ok=True)
    records = _load_feedback()
    records.append({
        "timestamp": datetime.utcnow().isoformat(),
        "thought":   thought,
        "predicted": predicted,
        "corrected": corrected,
    })
    with open(FEEDBACK_LOG, "w") as f:
        json.dump(records, f, indent=2)


def _feedback_bias(thought: str) -> Dict[str, float]:
    """
    Compute a simple frequency-based bias from stored corrections.
    Returns a dict {intent: bias_score} where repeated corrections for a
    thought → intent pair increase that intent's probability.
    """
    records = _load_feedback()
    bias: Dict[str, float] = {i: 0.0 for i in INTENTS}
    for r in records:
        if r["thought"].lower().strip() == thought.lower().strip():
            if r["corrected"] in bias:
                bias[r["corrected"]] += 1.0
    return bias


# ─── Core decoder ─────────────────────────────────────────────────────────────

def _rule_based_fallback(thought: str) -> Tuple[str, float]:
    """
    Direct pattern-key lookup as a strong prior.
    Confidence is set high (0.88) because the mapping is deterministic.
    """
    key = map_thought_to_key(thought)
    if key != "default":
        return key, 0.88
    return "stop_action", 0.50


def decode_thought(thought: str) -> Dict:
    """
    Full inference pipeline.

    Returns
    -------
    dict with keys:
        intent       : str   canonical action key
        label        : str   UPPERCASE display label
        confidence   : float 0-1
        probabilities: dict  {intent: probability}
        embedding_norm: float L2 norm of 512-d embedding
    """
    # 1. Simulate EEG
    eeg, _, pattern_key = simulate_eeg(thought)

    # 2. Normalise + tensorise
    normed = normalize_eeg(eeg)                          # (8, 256)
    tensor = torch.FloatTensor(normed).unsqueeze(0)      # (1, 8, 256)

    # 3. Model inference
    model = _get_model()
    with torch.no_grad():
        logits, embedding = model(tensor)                # (1, 5), (1, 512)
    probs = F.softmax(logits, dim=-1).squeeze().numpy()  # (5,)

    # 4. Apply adaptive feedback bias
    bias = _feedback_bias(thought)
    for i, intent in enumerate(INTENTS):
        probs[i] += bias[intent] * 0.15   # gentle nudge
    probs = np.clip(probs, 0, None)
    prob_sum = probs.sum()
    if prob_sum > 0:
        probs = probs / prob_sum

    # 5. If model hasn't been trained, fall back to rule-based prior
    if prob_sum < 0.01:
        intent, confidence = _rule_based_fallback(thought)
        probs_dict = {k: (confidence if k == intent else (1 - confidence) / (N_CLASSES - 1)) for k in INTENTS}
    else:
        # Rule-based prior blending: strengthen signal from simulator mapping
        rule_intent, rule_conf = _rule_based_fallback(thought)
        if rule_intent != "stop_action":
            ri = INTENTS.index(rule_intent)
            # Blend 60% model + 40% rule
            rule_vec = np.zeros(N_CLASSES)
            rule_vec[ri] = 1.0
            probs = 0.6 * probs + 0.4 * rule_vec

        idx        = int(np.argmax(probs))
        intent     = INTENTS[idx]
        confidence = float(probs[idx])
        probs_dict = {INTENTS[i]: float(probs[i]) for i in range(N_CLASSES)}

    return {
        "intent":         intent,
        "label":          INTENT_LABELS.get(intent, intent.upper()),
        "confidence":     round(confidence, 4),
        "probabilities":  probs_dict,
        "embedding_norm": float(embedding.norm().item()),
        "pattern_key":    pattern_key,
    }
