import os
import json

import pandas as pd
import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://localhost:8000")


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

st.set_page_config(page_title="ByteGuard", page_icon="BG", layout="wide")

st.title("ByteGuard: Multimodal Deepfake Video Detection")
st.caption("Upload a video to get a fake-vs-real score from the backend model.")

with st.sidebar:
    st.subheader("Backend")
    st.write(f"API URL: `{API_URL}`")

health_ok = False
try:
    health = requests.get(f"{API_URL}/health", timeout=5)
    health_ok = health.status_code == 200
except requests.RequestException:
    health_ok = False

if health_ok:
    st.success("Backend status: healthy")
else:
    st.warning("Backend unreachable. Start API before running predictions.")

uploaded = st.file_uploader("Upload video", type=["mp4", "mov", "avi", "mkv", "webm"])

if uploaded is not None:
    st.video(uploaded)

    if st.button("Run detection", type="primary"):
        files = {"video": (uploaded.name, uploaded.getvalue(), uploaded.type or "video/mp4")}
        try:
            response = requests.post(f"{API_URL}/predict", files=files, timeout=120)
            if response.status_code != 200:
                st.error(f"Request failed ({response.status_code}): {response.text}")
            else:
                payload = response.json()
                score = float(payload["score"])
                label = payload["label"]
                confidence = float(payload["confidence"])

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Prediction", label.upper())
                with col2:
                    st.metric("Fake Score", f"{score:.3f}")
                with col3:
                    st.metric("Confidence", f"{confidence:.3f}")

                st.progress(min(max(score, 0.0), 1.0), text="Fake probability")

                st.subheader("Modality Weights")
                weights = payload.get("modality_weights", {})
                if weights:
                    df = pd.DataFrame(
                        {
                            "modality": list(weights.keys()),
                            "weight": list(weights.values()),
                        }
                    ).set_index("modality")
                    st.bar_chart(df)

                st.subheader("Extraction Status")
                st.json(payload.get("extracted_modalities", {}))

                st.subheader("Model Source")
                st.code(payload.get("model_source", "unknown"))

                report = payload.get("report", {})
                if report:
                    st.subheader("Analysis Report")
                    risk = str(report.get("risk_level", "unknown")).upper()
                    report_id = report.get("report_id", "n/a")
                    st.write(f"Report ID: `{report_id}`")
                    st.write(f"Risk Level: `{risk}`")
                    st.write(report.get("summary", ""))
                    st.info(report.get("recommendation", ""))

                    media_info = report.get("media_info", {})
                    if media_info:
                        st.markdown("**Media Info**")
                        media_df = pd.DataFrame(
                            {"metric": list(media_info.keys()), "value": list(media_info.values())}
                        )
                        st.dataframe(media_df, use_container_width=True, hide_index=True)

                    forensic = report.get("forensic_signals", {})
                    if forensic:
                        st.markdown("**Forensic Signals**")
                        forensic_df = pd.DataFrame(
                            {"signal": list(forensic.keys()), "value": list(forensic.values())}
                        ).set_index("signal")
                        st.bar_chart(forensic_df)

                    report_json = json.dumps(payload, indent=2)
                    report_txt = build_report_text(payload, uploaded.name)
                    c1, c2 = st.columns(2)
                    with c1:
                        st.download_button(
                            label="Download Report JSON",
                            data=report_json,
                            file_name=f"{report_id}.json",
                            mime="application/json",
                        )
                    with c2:
                        st.download_button(
                            label="Download Report TXT",
                            data=report_txt,
                            file_name=f"{report_id}.txt",
                            mime="text/plain",
                        )
        except requests.RequestException as exc:
            st.error(f"Network error: {exc}")
