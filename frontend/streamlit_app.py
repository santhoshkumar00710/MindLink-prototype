"""
MindLink AI – Streamlit Dashboard
===================================
Futuristic neurotech interface for the Virtual BCI prototype.
"""

import sys
import os
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time

# ── Import pipeline ──────────────────────────────────────────────────────────
from backend.eeg_simulator import (
    simulate_eeg, compute_power_spectrum,
    CHANNEL_LABELS, N_CHANNELS, SAMPLING_RATE,
)
from backend.thought_decoder import decode_thought, save_feedback, INTENTS, INTENT_LABELS
from backend.executor import execute_command
from utils.signal_processing import compute_all_band_powers, compute_spectrogram

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & GLOBAL STYLE
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="MindLink AI – Virtual BCI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

NEON_BLUE   = "#00d4ff"
NEON_PURPLE = "#a855f7"
NEON_GREEN  = "#39ff14"
DARK_BG     = "#0a0a1a"
CARD_BG     = "#0f0f2d"
ACCENT      = "#1e1e4a"

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Share Tech Mono', monospace;
        background-color: {DARK_BG};
        color: #c0d0ff;
    }}

    /* App background */
    .stApp {{
        background: radial-gradient(ellipse at top, #0d0d2b 0%, {DARK_BG} 70%);
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #060618 0%, #0b0b22 100%);
        border-right: 1px solid {NEON_BLUE}33;
    }}

    /* Metric card */
    div[data-testid="metric-container"] {{
        background: {CARD_BG};
        border: 1px solid {NEON_BLUE}44;
        border-radius: 8px;
        padding: 12px;
        box-shadow: 0 0 12px {NEON_BLUE}22;
    }}

    /* Input box */
    .stTextInput > div > div > input {{
        background-color: #0b0b2b !important;
        border: 1px solid {NEON_BLUE}88 !important;
        color: {NEON_BLUE} !important;
        font-family: 'Orbitron', sans-serif !important;
        font-size: 1.1rem !important;
        border-radius: 6px !important;
        box-shadow: 0 0 8px {NEON_BLUE}40 !important;
    }}

    /* Buttons */
    .stButton > button {{
        background: linear-gradient(135deg, {NEON_PURPLE}aa, {NEON_BLUE}aa);
        color: white;
        border: 1px solid {NEON_BLUE};
        border-radius: 6px;
        font-family: 'Orbitron', sans-serif;
        font-size: 0.85rem;
        letter-spacing: 1px;
        padding: 0.5rem 1.5rem;
        box-shadow: 0 0 15px {NEON_BLUE}55;
        transition: all 0.2s;
    }}
    .stButton > button:hover {{
        box-shadow: 0 0 25px {NEON_BLUE}aa;
        transform: translateY(-1px);
    }}

    /* Headers */
    h1, h2, h3 {{
        font-family: 'Orbitron', sans-serif !important;
        color: {NEON_BLUE} !important;
        text-shadow: 0 0 10px {NEON_BLUE}88;
    }}

    /* Expander */
    .streamlit-expanderHeader {{
        background: {ACCENT} !important;
        border: 1px solid {NEON_PURPLE}44 !important;
    }}

    /* Select box */
    .stSelectbox > div > div {{
        background-color: #0b0b2b !important;
        border: 1px solid {NEON_PURPLE}88 !important;
        color: {NEON_PURPLE} !important;
    }}

    /* Dividers */
    hr {{
        border-color: {NEON_BLUE}33 !important;
    }}

    /* Scrollbar */
    ::-webkit-scrollbar {{ width: 4px; }}
    ::-webkit-scrollbar-track {{ background: {DARK_BG}; }}
    ::-webkit-scrollbar-thumb {{ background: {NEON_BLUE}66; border-radius: 4px; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS – Plotly chart builders
# ═══════════════════════════════════════════════════════════════════════════════

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(5,5,20,0.8)",
    font=dict(family="Share Tech Mono", color="#8899cc"),
    margin=dict(l=40, r=20, t=40, b=40),
    xaxis=dict(gridcolor="#1a1a3a", linecolor="#2a2a5a"),
    yaxis=dict(gridcolor="#1a1a3a", linecolor="#2a2a5a"),
)

CHANNEL_COLORS = [
    "#00d4ff", "#a855f7", "#39ff14", "#ff6b00",
    "#ff0055", "#ffdd00", "#00ffaa", "#ff00ff",
]


def build_eeg_waveform(eeg: np.ndarray, time_axis: np.ndarray, n_ch: int = 4) -> go.Figure:
    fig = go.Figure()
    offset = 3.0
    for i in range(n_ch):
        fig.add_trace(go.Scatter(
            x=time_axis,
            y=eeg[i] + i * offset,
            name=CHANNEL_LABELS[i],
            line=dict(color=CHANNEL_COLORS[i], width=1.5),
            mode="lines",
        ))
        # Channel label
        fig.add_annotation(
            x=time_axis[-1], y=i * offset,
            text=CHANNEL_LABELS[i],
            showarrow=False,
            font=dict(color=CHANNEL_COLORS[i], size=11),
            xanchor="left",
        )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="⚡ Live EEG Waveform",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude (µV)",
        showlegend=False,
        height=320,
    )
    # Scan-line glow effect via shape
    fig.add_shape(
        type="line", x0=time_axis[-1]*0.9, y0=-1, x1=time_axis[-1]*0.9, y1=n_ch*offset,
        line=dict(color=NEON_BLUE, width=1, dash="dot"),
    )
    return fig


def build_spectrogram(eeg: np.ndarray) -> go.Figure:
    from utils.signal_processing import compute_spectrogram
    t, f, Sxx = compute_spectrogram(eeg[0])    # use channel 0
    mask = f <= 50
    fig = go.Figure(go.Heatmap(
        z=Sxx[mask],
        x=t,
        y=f[mask],
        colorscale=[[0, "#000018"], [0.3, "#2a0066"], [0.6, "#0055ff"], [0.85, "#00d4ff"], [1.0, "#ffffff"]],
        showscale=True,
        colorbar=dict(tickfont=dict(color="#8899cc"), title="dB"),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="🌊 Brain Spectrogram",
        xaxis_title="Time (s)",
        yaxis_title="Frequency (Hz)",
        height=280,
    )
    return fig


def build_psd(freqs: np.ndarray, psd: np.ndarray) -> go.Figure:
    mask = freqs <= 50
    fig = go.Figure(go.Scatter(
        x=freqs[mask], y=psd[mask],
        fill="tozeroy",
        line=dict(color=NEON_BLUE, width=2),
        fillcolor=f"rgba(0,212,255,0.15)",
        mode="lines",
    ))
    # Band shading
    bands = [("δ", 0.5, 4, "#ff0055"), ("θ", 4, 8, "#a855f7"), ("α", 8, 13, "#00d4ff"),
             ("β", 13, 30, "#39ff14"), ("γ", 30, 50, "#ffdd00")]
    for label, lo, hi, color in bands:
        fig.add_vrect(x0=lo, x1=hi, fillcolor=color, opacity=0.07, line_width=0)
        fig.add_annotation(x=(lo+hi)/2, y=psd[mask].max()*0.85, text=label,
                           showarrow=False, font=dict(color=color, size=11))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="📊 Power Spectral Density",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Power",
        height=260,
    )
    return fig


def build_intent_radar(probs: dict) -> go.Figure:
    labels = list(probs.keys())
    values = list(probs.values())
    labels_disp = [INTENT_LABELS.get(l, l) for l in labels]
    values_norm = [v * 100 for v in values]

    fig = go.Figure(go.Barpolar(
        r=values_norm,
        theta=labels_disp,
        marker=dict(
            color=values_norm,
            colorscale=[[0, "#0d0d2b"], [0.5, "#a855f7"], [1.0, "#00d4ff"]],
            showscale=False,
            line=dict(color=NEON_BLUE, width=1),
        ),
        opacity=0.85,
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="🎯 Intent Probability",
        polar=dict(
            bgcolor="rgba(5,5,20,0.8)",
            angularaxis=dict(color="#6677aa", gridcolor="#1a1a3a"),
            radialaxis=dict(color="#6677aa", gridcolor="#1a1a3a", range=[0, 100]),
        ),
        height=320,
    )
    return fig


def build_band_bar(band_powers: dict) -> go.Figure:
    bands = list(band_powers.keys())
    powers = [band_powers[b] for b in bands]
    colors = ["#ff0055", "#a855f7", "#00d4ff", "#39ff14", "#ffdd00"]
    fig = go.Figure(go.Bar(
        x=[b.capitalize() for b in bands],
        y=powers,
        marker_color=colors,
        marker_line_color=NEON_BLUE,
        marker_line_width=1,
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="🧠 Neural Band Activity",
        xaxis_title="Band",
        yaxis_title="Power",
        height=240,
        showlegend=False,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

st.sidebar.markdown(
    f"""
    <div style="text-align:center;padding:16px 0 8px;">
      <div style="font-family:Orbitron,sans-serif;font-size:1.5rem;
                  color:{NEON_BLUE};text-shadow:0 0 14px {NEON_BLUE};">🧠</div>
      <div style="font-family:Orbitron,sans-serif;font-size:1.1rem;
                  color:{NEON_BLUE};font-weight:700;">MindLink AI</div>
      <div style="font-size:0.7rem;color:#6677aa;margin-top:4px;">Virtual BCI · v1.0</div>
    </div>
    <hr>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    f"<div style='color:#8899cc;font-size:0.85rem;'>**Pipeline**</div>",
    unsafe_allow_html=True,
)

pipeline_steps = ["💬 Typed Thought", "〰️ EEG Simulation", "🔬 EEGNet Encoder",
                  "🎯 Intent Decoder", "⚡ Command Execution"]
for step in pipeline_steps:
    st.sidebar.markdown(
        f"<div style='padding:4px 0 4px 8px;color:#99aacc;font-size:0.8rem;'>▸ {step}</div>",
        unsafe_allow_html=True,
    )

st.sidebar.markdown("<hr>", unsafe_allow_html=True)
st.sidebar.markdown(
    f"<div style='color:#8899cc;font-size:0.85rem;'>**Quick Commands**</div>",
    unsafe_allow_html=True,
)
quick_cmds = ["open browser", "scroll down", "type hello", "play music", "stop"]
selected_quick = st.sidebar.selectbox("Select preset →", [""] + quick_cmds, label_visibility="collapsed")

st.sidebar.markdown("<hr>", unsafe_allow_html=True)
st.sidebar.markdown(
    f"<div style='color:#8899cc;font-size:0.85rem;'>**Channels**</div>",
    unsafe_allow_html=True,
)
n_channels_display = st.sidebar.slider("Waveform channels", 2, 8, 4)
auto_execute = st.sidebar.checkbox("Auto-execute on decode", value=True)

st.sidebar.markdown("<hr>", unsafe_allow_html=True)
st.sidebar.markdown(
    f"""<div style='color:#6677aa;font-size:0.72rem;text-align:center;'>
    Sampling Rate: {SAMPLING_RATE} Hz<br>Channels: {N_CHANNELS}<br>
    Freq Range: 1–50 Hz<br>Window: 1.0 s</div>""",
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(
    f"""
    <div style="text-align:center;padding:20px 0 4px;">
      <h1 style="font-family:Orbitron,sans-serif;font-size:2rem;
                 background:linear-gradient(90deg,{NEON_BLUE},{NEON_PURPLE});
                 -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                 margin-bottom:0;">
        🧠 MindLink AI
      </h1>
      <p style="color:#6677aa;font-size:0.85rem;margin-top:4px;">
        Virtual Brain–Computer Interface · Neural Signal Decoding Platform
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Status bar
col_s1, col_s2, col_s3, col_s4 = st.columns(4)
col_s1.metric("Signal Quality", "97%", "+ Optimal")
col_s2.metric("Latency", "< 1s", "Real-time")
col_s3.metric("Channels", "8", "Active")
col_s4.metric("Model", "EEGNet-BCI", "Online")

st.markdown("---")

# ── Thought Input ─────────────────────────────────────────────────────────────
st.markdown("### 💬 Thought Input Panel")
thought_input = st.text_input(
    "Type your thought command:",
    value=selected_quick if selected_quick else "",
    placeholder="e.g. open browser · scroll down · type hello · play music · stop",
    label_visibility="collapsed",
)

col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 4])
transmit_clicked = col_btn1.button("⚡ TRANSMIT", use_container_width=True)
clear_clicked    = col_btn2.button("🔄 RESET",    use_container_width=True)

if clear_clicked:
    st.rerun()

# ── Session state init ────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state["history"] = []
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None
if "last_exec" not in st.session_state:
    st.session_state["last_exec"] = None
if "last_eeg" not in st.session_state:
    st.session_state["last_eeg"] = None

# ── TRANSMIT pipeline ────────────────────────────────────────────────────────
if transmit_clicked and thought_input.strip():
    with st.spinner("🧠 Encoding neural signal…"):
        # Simulate progress
        prog = st.progress(0)
        time.sleep(0.1); prog.progress(20, text="Simulating EEG…")
        eeg, time_axis, pattern_key = simulate_eeg(thought_input)
        freqs, psd = compute_power_spectrum(eeg)
        band_powers = compute_all_band_powers(eeg)

        time.sleep(0.1); prog.progress(50, text="Running EEGNet encoder…")
        result = decode_thought(thought_input)

        time.sleep(0.1); prog.progress(80, text="Decoding intent…")
        exec_result = None
        if auto_execute:
            exec_result = execute_command(result["intent"])
        prog.progress(100, text="Complete ✓")
        time.sleep(0.2); prog.empty()

    st.session_state["last_result"] = result
    st.session_state["last_exec"]   = exec_result
    st.session_state["last_eeg"]    = (eeg, time_axis, freqs, psd, band_powers)
    st.session_state["history"].append({
        "thought": thought_input,
        "intent":  result["intent"],
        "conf":    result["confidence"],
    })

# ── Main visualisation area ────────────────────────────────────────────────────
result     = st.session_state["last_result"]
exec_result = st.session_state["last_exec"]
eeg_pkg    = st.session_state["last_eeg"]

if eeg_pkg is not None:
    eeg, time_axis, freqs, psd, band_powers = eeg_pkg

    # ── Row 1: Decoded thought | Command status ─────────────────────────────
    st.markdown("---")
    col_d, col_e = st.columns([1, 1])

    with col_d:
        st.markdown("### 🎯 Decoded Thought")
        conf_pct = int(result["confidence"] * 100)
        bar_color = NEON_GREEN if conf_pct >= 80 else (NEON_BLUE if conf_pct >= 60 else "#ff6b00")
        st.markdown(
            f"""
            <div style="background:{CARD_BG};border:1px solid {NEON_BLUE}55;
                        border-radius:10px;padding:20px;margin-bottom:12px;">
              <div style="font-size:0.75rem;color:#6677aa;">DETECTED THOUGHT</div>
              <div style="font-family:Orbitron,sans-serif;font-size:1.5rem;
                          color:{NEON_BLUE};margin:6px 0;">{result['label']}</div>
              <div style="font-size:0.75rem;color:#6677aa;margin-top:8px;">CONFIDENCE</div>
              <div style="background:#0a0a1a;border-radius:4px;height:12px;margin:6px 0;">
                <div style="width:{conf_pct}%;height:100%;border-radius:4px;
                             background:linear-gradient(90deg,{NEON_PURPLE},{bar_color});
                             box-shadow:0 0 8px {bar_color}80;"></div>
              </div>
              <div style="font-size:1.2rem;color:{bar_color};font-family:Orbitron,sans-serif;">
                {conf_pct}%
              </div>
              <div style="font-size:0.7rem;color:#6677aa;margin-top:8px;">
                Pattern key: <span style="color:{NEON_PURPLE};">{result['pattern_key']}</span>
                &nbsp;|&nbsp; Embed norm: {result['embedding_norm']:.2f}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_e:
        st.markdown("### ⚡ Command Execution")
        if exec_result:
            status_color = NEON_GREEN if exec_result["status"] == "success" else "#ff0055"
            status_icon  = "✅" if exec_result["status"] == "success" else "❌"
            st.markdown(
                f"""
                <div style="background:{CARD_BG};border:1px solid {status_color}55;
                            border-radius:10px;padding:20px;margin-bottom:12px;">
                  <div style="font-size:0.75rem;color:#6677aa;">EXECUTION STATUS</div>
                  <div style="font-size:1.1rem;color:{status_color};
                               font-family:Orbitron,sans-serif;margin:8px 0;">
                    {status_icon} {exec_result['status'].upper()}
                  </div>
                  <div style="font-size:0.8rem;color:#99aacc;margin-top:6px;">
                    {exec_result['message']}
                  </div>
                  <div style="font-size:0.75rem;color:#6677aa;margin-top:12px;">INTENT FIRED</div>
                  <div style="font-size:0.9rem;color:{NEON_PURPLE};">{result['intent']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif not auto_execute:
            if st.button("▶ Execute Command", use_container_width=True):
                er = execute_command(result["intent"])
                st.session_state["last_exec"] = er
                st.rerun()
        else:
            st.info("No execution result yet.")

    # ── Row 2: EEG Waveform | Spectrogram ────────────────────────────────────
    st.markdown("---")
    col_w, col_s = st.columns([3, 2])
    with col_w:
        st.plotly_chart(build_eeg_waveform(eeg, time_axis, n_channels_display),
                        use_container_width=True, config={"displayModeBar": False})
    with col_s:
        st.plotly_chart(build_spectrogram(eeg),
                        use_container_width=True, config={"displayModeBar": False})

    # ── Row 3: PSD | Intent Radar | Band Bar ─────────────────────────────────
    col_p, col_r, col_b = st.columns([2, 2, 1.5])
    with col_p:
        st.plotly_chart(build_psd(freqs, psd),
                        use_container_width=True, config={"displayModeBar": False})
    with col_r:
        st.plotly_chart(build_intent_radar(result["probabilities"]),
                        use_container_width=True, config={"displayModeBar": False})
    with col_b:
        st.plotly_chart(build_band_bar(band_powers),
                        use_container_width=True, config={"displayModeBar": False})

    # ── Adaptive Feedback ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔄 Adaptive Feedback")
    corr_col1, corr_col2, corr_col3 = st.columns([3, 3, 2])
    with corr_col1:
        st.markdown(f"**Predicted:** `{result['label']}`")
    with corr_col2:
        correction = st.selectbox(
            "Correct to:",
            ["-- No correction --"] + list(INTENT_LABELS.keys()),
            label_visibility="visible",
        )
    with corr_col3:
        if st.button("💾 Submit Correction"):
            if correction != "-- No correction --":
                save_feedback(
                    thought=st.session_state["history"][-1]["thought"],
                    predicted=result["intent"],
                    corrected=correction,
                )
                st.success("✅ Feedback saved – model will adapt!")

else:
    # ── Idle splash ──────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div style="text-align:center;padding:60px 20px;
                    border:1px dashed {NEON_BLUE}44;border-radius:12px;
                    background:{CARD_BG};margin:20px 0;">
          <div style="font-size:3rem;margin-bottom:12px;">🧠</div>
          <div style="font-family:Orbitron,sans-serif;font-size:1.2rem;
                      color:{NEON_BLUE};margin-bottom:8px;">Awaiting Neural Input</div>
          <div style="color:#6677aa;font-size:0.85rem;">
            Type a thought command above and press <strong>⚡ TRANSMIT</strong>
          </div>
          <div style="color:#445566;font-size:0.75rem;margin-top:16px;">
            Try: "open browser" · "scroll down" · "type hello" · "play music" · "stop"
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Session History ───────────────────────────────────────────────────────────
if st.session_state["history"]:
    st.markdown("---")
    with st.expander("📜 Session History", expanded=False):
        for i, h in enumerate(reversed(st.session_state["history"][-10:])):
            conf_pct = int(h["conf"] * 100)
            c = NEON_GREEN if conf_pct >= 80 else (NEON_BLUE if conf_pct >= 60 else "#ff6b00")
            st.markdown(
                f"""<div style="font-size:0.82rem;padding:4px 0;
                    border-bottom:1px solid {NEON_BLUE}22;">
                    <span style="color:#6677aa;">#{len(st.session_state['history'])-i}</span>&nbsp;
                    <span style="color:{NEON_BLUE};">"{h['thought']}"</span>&nbsp;→&nbsp;
                    <span style="color:{NEON_PURPLE};">{INTENT_LABELS.get(h['intent'],h['intent'])}</span>&nbsp;
                    <span style="color:{c};">({conf_pct}%)</span>
                    </div>""",
                unsafe_allow_html=True,
            )
