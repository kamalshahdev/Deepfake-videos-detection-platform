import json
import os

import pandas as pd
import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://localhost:8000")


def inject_styles() -> None:
    st.markdown(
        """
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

          :root {
            --bg-0: #f2f7f7;
            --bg-1: #dbe9e8;
            --ink: #142625;
            --muted: #4f6664;
            --card: rgba(255, 255, 255, 0.72);
            --line: rgba(20, 38, 37, 0.16);
            --ok: #13795b;
            --warn: #b26a00;
            --bad: #a1222f;
            --accent: #0f7b7b;
          }

          html, body, [class*="css"]  {
            font-family: "Space Grotesk", sans-serif;
          }

          .stApp {
            color: var(--ink);
            background:
              radial-gradient(850px 500px at 2% 5%, rgba(15, 123, 123, 0.18), rgba(15, 123, 123, 0) 60%),
              radial-gradient(800px 450px at 98% 8%, rgba(230, 171, 37, 0.14), rgba(230, 171, 37, 0) 58%),
              linear-gradient(180deg, var(--bg-0) 0%, var(--bg-1) 100%);
          }

          [data-testid="stAppViewContainer"] > .main .block-container {
            max-width: 1120px;
            padding-top: 1.8rem;
            padding-bottom: 2.2rem;
          }

          .hero {
            border: 1px solid var(--line);
            background: linear-gradient(145deg, rgba(255,255,255,0.88), rgba(255,255,255,0.56));
            border-radius: 18px;
            padding: 1.2rem 1.35rem;
            box-shadow: 0 12px 34px rgba(17, 41, 39, 0.11);
            animation: hero-enter 420ms ease-out both;
          }

          @keyframes hero-enter {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
          }

          .hero h1 {
            margin: 0 0 .2rem 0;
            font-size: clamp(1.55rem, 3vw, 2.25rem);
            letter-spacing: -0.02em;
            color: #103231;
          }

          .hero p {
            margin: 0;
            color: var(--muted);
            font-size: 0.95rem;
          }

          .badge-row {
            margin-top: .75rem;
            display: flex;
            gap: .5rem;
            flex-wrap: wrap;
          }

          .badge {
            display: inline-flex;
            align-items: center;
            border: 1px solid var(--line);
            border-radius: 999px;
            padding: .24rem .62rem;
            font-size: .76rem;
            color: #1e3a38;
            background: rgba(255,255,255,0.6);
          }

          .badge.ok { border-color: rgba(19,121,91,.45); color: var(--ok); }
          .badge.warn { border-color: rgba(178,106,0,.45); color: var(--warn); }
          .badge.bad { border-color: rgba(161,34,47,.45); color: var(--bad); }

          .metric-card {
            border: 1px solid var(--line);
            background: var(--card);
            border-radius: 14px;
            padding: .78rem .9rem;
            box-shadow: 0 8px 20px rgba(18, 43, 41, 0.08);
          }

          .metric-label {
            margin: 0;
            color: var(--muted);
            font-size: .72rem;
            text-transform: uppercase;
            letter-spacing: .08em;
          }

          .metric-value {
            margin: .12rem 0 0 0;
            font-size: 1.15rem;
            font-weight: 700;
            color: #183c3a;
          }

          .mono {
            font-family: "IBM Plex Mono", monospace;
            font-size: .88rem;
          }

          .stTabs [data-baseweb="tab-list"] {
            gap: .45rem;
          }

          .stTabs [data-baseweb="tab"] {
            border-radius: 12px;
            border: 1px solid var(--line);
            background: rgba(255,255,255,0.52);
            padding: .5rem .95rem;
            font-weight: 600;
          }

          .stButton > button {
            border-radius: 12px;
            border: 1px solid rgba(16, 64, 62, 0.35);
            background: linear-gradient(170deg, #0f7b7b, #0e6a6a);
            color: #f3fbfb;
            font-weight: 600;
            box-shadow: 0 10px 24px rgba(15,123,123,0.25);
          }

          .stButton > button:hover {
            border-color: rgba(16,64,62,0.52);
            background: linear-gradient(170deg, #0e6f6f, #0c6161);
          }

          [data-testid="stSidebar"] > div:first-child {
            background: linear-gradient(180deg, rgba(11,66,64,0.96), rgba(19,92,87,0.94));
          }

          [data-testid="stSidebar"] * {
            color: #edf8f7 !important;
          }

          .block-note {
            border: 1px dashed var(--line);
            border-radius: 12px;
            padding: .62rem .78rem;
            background: rgba(255,255,255,0.52);
            color: var(--muted);
            font-size: .86rem;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def build_report_text(payload: dict, file_name: str) -> str:
    report = payload.get("report", {})
    media_info = report.get("media_info", {})
    signals = report.get("forensic_signals", {})

    lines = [
        "ByteGuard Deepfake Analysis Report",
        "--------------------------------",
        f"File: {file_name}",
        f"Report ID: {report.get('report_id', 'n/a')}",
        f"Prediction: {payload.get('label', 'n/a')}",
        f"Fake Score: {payload.get('score', 0.0):.6f}",
        f"Confidence: {payload.get('confidence', 0.0):.6f}",
        f"Risk Level: {report.get('risk_level', 'n/a')}",
        f"Model Source: {payload.get('model_source', 'n/a')}",
        "",
        "Summary:",
        report.get("summary", "n/a"),
        "",
        "Recommendation:",
        report.get("recommendation", "n/a"),
        "",
        "Media Info:",
    ]
    for key, value in media_info.items():
        lines.append(f"- {key}: {value}")

    lines.append("")
    lines.append("Forensic Signals:")
    for key, value in signals.items():
        lines.append(f"- {key}: {value}")

    lines.append("")
    lines.append("Modality Weights:")
    for key, value in payload.get("modality_weights", {}).items():
        lines.append(f"- {key}: {value}")

    lines.append("")
    lines.append("Extracted Modalities:")
    for key, value in payload.get("extracted_modalities", {}).items():
        lines.append(f"- {key}: {value}")

    return "\n".join(lines)


def render_hero(health_ok: bool, model_info: dict | None) -> None:
    if not health_ok:
        backend_text = "Backend offline"
        backend_class = "bad"
    else:
        backend_text = "Backend healthy"
        backend_class = "ok"

    if model_info and bool(model_info.get("model_loaded", False)):
        model_text = "Trained model loaded"
        model_class = "ok"
    else:
        model_text = "Heuristic fallback"
        model_class = "warn"

    threshold = model_info.get("decision_threshold", 0.5) if model_info else 0.5

    st.markdown(
        f"""
        <section class="hero">
          <h1>ByteGuard Deepfake Analysis Console</h1>
          <p>Inspect uploaded clips or URL videos and generate forensic reports with multimodal signals.</p>
          <div class="badge-row">
            <span class="badge {backend_class}">{backend_text}</span>
            <span class="badge {model_class}">{model_text}</span>
            <span class="badge">Threshold: {threshold}</span>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
          <p class="metric-label">{label}</p>
          <p class="metric-value">{value}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prediction(payload: dict, source_name: str) -> None:
    score = float(payload["score"])
    label = str(payload["label"]).upper()
    confidence = float(payload["confidence"])

    c1, c2, c3 = st.columns(3)
    with c1:
        render_metric_card("Prediction", label)
    with c2:
        render_metric_card("Fake Score", f"{score:.3f}")
    with c3:
        render_metric_card("Confidence", f"{confidence:.3f}")

    st.markdown(f'<p class="mono">Source: {source_name}</p>', unsafe_allow_html=True)
    st.progress(min(max(score, 0.0), 1.0), text="Fake probability")

    st.subheader("Analysis Report")
    report = payload.get("report", {})
    risk = str(report.get("risk_level", "unknown")).upper()
    report_id = report.get("report_id", "n/a")
    st.markdown(
        f"""
        <div class="block-note">
          <strong>Report ID:</strong> {report_id}<br/>
          <strong>Risk:</strong> {risk}<br/>
          <strong>Summary:</strong> {report.get("summary", "")}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info(report.get("recommendation", "No recommendation provided."))

    st.subheader("Forensic Weights")
    weights = payload.get("modality_weights", {})
    if weights:
        df = pd.DataFrame({"modality": list(weights.keys()), "weight": list(weights.values())}).set_index("modality")
        st.bar_chart(df)

    st.subheader("Extraction Status")
    st.json(payload.get("extracted_modalities", {}))

    media_info = report.get("media_info", {})
    if media_info:
        st.subheader("Media Info")
        media_df = pd.DataFrame({"metric": list(media_info.keys()), "value": list(media_info.values())})
        st.dataframe(media_df, use_container_width=True, hide_index=True)

    forensic = report.get("forensic_signals", {})
    if forensic:
        st.subheader("Forensic Signals")
        forensic_df = pd.DataFrame({"signal": list(forensic.keys()), "value": list(forensic.values())}).set_index("signal")
        st.bar_chart(forensic_df)

    report_json = json.dumps(payload, indent=2)
    report_txt = build_report_text(payload, source_name)
    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            label="Download Report JSON",
            data=report_json,
            file_name=f"{report_id}.json",
            mime="application/json",
        )
    with d2:
        st.download_button(
            label="Download Report TXT",
            data=report_txt,
            file_name=f"{report_id}.txt",
            mime="text/plain",
        )


def call_predict_file(api_url: str, uploaded) -> dict:
    files = {"video": (uploaded.name, uploaded.getvalue(), uploaded.type or "video/mp4")}
    response = requests.post(f"{api_url}/predict", files=files, timeout=240)
    if response.status_code != 200:
        raise RuntimeError(f"Request failed ({response.status_code}): {response.text}")
    return response.json()


def call_predict_url(api_url: str, video_url: str) -> dict:
    response = requests.post(f"{api_url}/predict-url", json={"url": video_url.strip()}, timeout=420)
    if response.status_code != 200:
        raise RuntimeError(f"Request failed ({response.status_code}): {response.text}")
    return response.json()


st.set_page_config(page_title="ByteGuard Console", page_icon="BG", layout="wide")
inject_styles()

with st.sidebar:
    st.subheader("Connection")
    st.markdown(f"API endpoint: `{API_URL}`")
    st.markdown("Use a direct video file or paste a public video/post URL.")

health_ok = False
model_info: dict | None = None
try:
    health = requests.get(f"{API_URL}/health", timeout=5)
    health_ok = health.status_code == 200
except requests.RequestException:
    health_ok = False

if health_ok:
    try:
        model_resp = requests.get(f"{API_URL}/model-info", timeout=5)
        if model_resp.status_code == 200:
            model_info = model_resp.json()
    except requests.RequestException:
        model_info = None

render_hero(health_ok=health_ok, model_info=model_info)

if not health_ok:
    st.error("Backend unreachable. Start API before running predictions.")
elif model_info is None:
    st.warning("Backend is reachable, but model details are unavailable.")
else:
    if bool(model_info.get("model_loaded", False)):
        st.success(f"Loaded checkpoint: `{model_info.get('model_source', 'unknown')}`")
    else:
        st.warning("No trained checkpoint loaded. Results may be less reliable.")
    st.caption(f"Model path: `{model_info.get('model_path', 'n/a')}`")

upload_tab, url_tab, notes_tab = st.tabs(["Upload Video", "Analyze URL", "How to Use"])

with upload_tab:
    uploaded = st.file_uploader(
        "Drop video file",
        type=["mp4", "mov", "avi", "mkv", "webm"],
        help="Large files may take longer to process.",
    )
    if uploaded is not None:
        st.video(uploaded)

    run_file = st.button("Run detection on uploaded file", type="primary", use_container_width=True)
    if run_file:
        if uploaded is None:
            st.error("Upload a video first.")
        else:
            try:
                with st.spinner("Analyzing uploaded video..."):
                    payload = call_predict_file(API_URL, uploaded)
                render_prediction(payload, uploaded.name)
            except (requests.RequestException, RuntimeError) as exc:
                st.error(str(exc))

with url_tab:
    st.caption("Works for direct video URLs and many social post URLs (X/Twitter/etc.) when `yt-dlp` is installed.")
    video_url = st.text_input(
        "Paste public video URL",
        placeholder="https://x.com/.../status/... or https://cdn.example.com/video.mp4",
    )
    run_url = st.button("Run detection from URL", use_container_width=True)
    if run_url:
        if not video_url.strip():
            st.error("Enter a valid URL first.")
        else:
            try:
                with st.spinner("Downloading and analyzing URL..."):
                    payload = call_predict_url(API_URL, video_url)
                render_prediction(payload, video_url.strip())
            except (requests.RequestException, RuntimeError) as exc:
                st.error(str(exc))

with notes_tab:
    st.markdown(
        """
        <div class="block-note">
          <strong>Workflow</strong><br/>
          1. Confirm backend is healthy and model is loaded.<br/>
          2. Use Upload Video or Analyze URL.<br/>
          3. Review fake score, confidence, and forensic signals.<br/>
          4. Export JSON/TXT report for your records.
        </div>
        """,
        unsafe_allow_html=True,
    )
