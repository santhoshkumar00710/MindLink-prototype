# 🧠 MindLink AI – Virtual Brain Computer Interface

> A full-stack AI prototype that simulates a Brain-Computer Interface pipeline — no physical EEG headset required.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.33-red?logo=streamlit)](https://streamlit.io)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-orange?logo=pytorch)](https://pytorch.org)

---

## 🚀 Demo

Type a thought → watch the brain simulate → see the intent decoded → command fires.

```
User types:   "open browser"
       ↓
EEG Simulation (8ch · 256Hz · α/β/γ waves)
       ↓
EEGNet Encoder (Conv1D → BN → ReLU → BiLSTM → 512-d embedding)
       ↓
Intent Decoder  →  OPEN_BROWSER  (92% confidence)
       ↓
Browser opens → https://www.google.com
```

---

## 📦 Installation

```bash
# 1. Clone / navigate to project directory
cd "MindLink AI demo"

# 2. Create virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Initialise model weights (run once)
python init_model.py
```

---

## ▶️ Running Locally

### Option A – Streamlit only (recommended for demo)

```bash
streamlit run frontend/streamlit_app.py
```

Open **http://localhost:8501** in your browser.

### Option B – With FastAPI backend

```bash
# Terminal 1 – start the API server
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 – start Streamlit
streamlit run frontend/streamlit_app.py
```

---

## 🏗️ Project Structure

```
MindLink AI demo/
├── frontend/
│   └── streamlit_app.py       # Streamlit dashboard UI
├── backend/
│   ├── main.py                # FastAPI server
│   ├── eeg_simulator.py       # 8-ch EEG signal synthesis
│   ├── thought_decoder.py     # Full inference pipeline
│   ├── intent_model.py        # EEGNet-inspired PyTorch model
│   └── executor.py            # Command executor
├── models/
│   └── eegnet_model.pth       # Saved model weights
├── utils/
│   └── signal_processing.py  # DSP helpers (filter, PSD, spectrogram)
├── demo/                      # Place audio files / screenshots here
├── init_model.py              # One-shot weight initialisation
├── requirements.txt
└── README.md
```

---

## 🧠 Supported Thought Commands

| Typed Command    | Decoded Intent     | Action                            |
|------------------|--------------------|-----------------------------------|
| `open browser`   | OPEN_BROWSER       | Opens Google in your browser      |
| `scroll down`    | SCROLL_DOWN        | Scrolls screen down (pyautogui)   |
| `type hello`     | TYPE_HELLO         | Types "Hello from MindLink AI!"   |
| `play music`     | PLAY_MUSIC         | Plays audio from `demo/` folder   |
| `stop`           | STOP_ACTION        | Halts all active tasks            |

---

## 📊 Visualisations

| Chart                      | Description                                 |
|----------------------------|---------------------------------------------|
| ⚡ Live EEG Waveform       | 8 channels · 1-second window · 256 Hz       |
| 🌊 Brain Spectrogram       | STFT spectrogram for channel Fp1            |
| 📊 Power Spectral Density  | Welch PSD averaged across all channels      |
| 🎯 Intent Probability      | Polar bar chart of all 5 intent probabilities |
| 🧠 Neural Band Activity    | δ · θ · α · β · γ power bars               |

---

## 🔄 Adaptive Learning

If the model decodes the wrong intent, use the **Adaptive Feedback** panel to select the correct intent and click **Submit Correction**. Corrections are saved to `models/feedback_log.json` and used to bias future predictions for the same thought.

---

## 🌐 Deployment

| Target         | Platform          | Notes                           |
|----------------|-------------------|---------------------------------|
| Frontend       | Streamlit Cloud   | `streamlit run frontend/...`    |
| Frontend       | Hugging Face Spaces | Upload as Gradio/Streamlit    |
| Backend API    | Railway / Render  | `uvicorn backend.main:app`     |
| Training       | Google Colab GPU  | Fine-tune EEGNet on real data   |

---

## ⚙️ Model Architecture

```
Input EEG (8 × 256)
  → Conv1D(16, k=25) + BN + ReLU + Dropout
  → Conv1D(32, k=1)  + BN + ReLU + AvgPool(4) + Dropout
  → Conv1D(64, k=15) + BN + ReLU + AvgPool(8) + Dropout
  → BiLSTM(256, layers=2)   →  last-step hidden (512-d)
  → LayerNorm → Linear(512) → Tanh()           →  embedding
  → Linear(128) → ReLU → Dropout → Linear(5)   →  logits
```

---

## 📄 License

MIT License © 2025 MindLink AI
