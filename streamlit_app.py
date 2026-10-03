"""Streamlit dashboard for inspecting and classifying LTspice waveforms."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from circuit_fault_detector.features import FEATURE_COLUMNS, extract_features
from circuit_fault_detector.model import load_model, predict_features

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "random_forest.joblib"
FULL_MANIFEST_PATH = ROOT / "data" / "ltspice" / "manifest.csv"
DEMO_MANIFEST_PATH = ROOT / "data" / "demo" / "manifest.csv"
MANIFEST_PATH = FULL_MANIFEST_PATH if FULL_MANIFEST_PATH.is_file() else DEMO_MANIFEST_PATH

st.set_page_config(page_title="Circuit Fault Detector", page_icon="⚡", layout="wide")
st.title("⚡ Sallen-Key Circuit Fault Detector")
st.caption("LTspice waveform inspection with an eight-feature Random Forest classifier")
st.warning("Educational research prototype. Do not use predictions for live equipment or safety decisions.")

if not MODEL_PATH.is_file():
    st.info("No trained waveform model was found. Run `simulate.py`, `extract_features.py`, then `train.py` first.")
    st.code("python simulate.py --ltspice 'C:\\path\\to\\LTspice.exe'\npython extract_features.py\npython train.py", language="powershell")
    model = None
else:
    model, _ = load_model(MODEL_PATH)

mode = st.radio("Choose input", ["Demo Mode", "Upload waveform CSV"], horizontal=True)
waveform = None
sample_name = None

if mode == "Demo Mode":
    if not MANIFEST_PATH.is_file():
        st.info("No demo waveforms are available. Generate them with `simulate.py` first.")
    else:
        manifest = pd.read_csv(MANIFEST_PATH)
        if manifest.empty:
            st.info("The LTspice manifest has no completed runs yet.")
        else:
            names = manifest["sample_id"].astype(str).tolist()
            sample_name = st.selectbox("Saved waveform", names)
            row = manifest.loc[manifest["sample_id"].astype(str) == sample_name].iloc[0]
            waveform = pd.read_csv(MANIFEST_PATH.parent / row["waveform_path"])
            st.caption(f"Reference label: {row['fault_label']}")
else:
    uploaded = st.file_uploader("Upload a CSV with `time_s` and `output_v` columns", type=["csv"])
    if uploaded is not None:
        waveform = pd.read_csv(uploaded)
        sample_name = uploaded.name

if waveform is not None:
    required = {"time_s", "output_v"}
    if not required.issubset(waveform.columns):
        st.error("CSV must contain time_s and output_v columns.")
    else:
        fig, ax = plt.subplots(figsize=(10, 3.6))
        ax.plot(waveform["time_s"], waveform["output_v"], linewidth=1)
        ax.set(xlabel="Time (s)", ylabel="Output voltage (V)", title=f"Waveform: {sample_name}")
        ax.grid(True, alpha=0.25)
        st.pyplot(fig, clear_figure=True)
        values = extract_features(waveform["time_s"], waveform["output_v"])
        feature_frame = pd.DataFrame([values], columns=FEATURE_COLUMNS)
        left, right = st.columns([1, 1])
        with left:
            st.subheader("Extracted waveform features")
            st.dataframe(feature_frame, hide_index=True, use_container_width=True)
        with right:
            st.subheader("Fault prediction")
            if model is None:
                st.info("Train the Random Forest model to enable predictions.")
            else:
                label, confidence, probabilities = predict_features(model, feature_frame)
                st.metric("Predicted class", label)
                st.metric("Model confidence", f"{confidence:.1%}")
                st.bar_chart(pd.Series(probabilities, name="Probability").sort_values(ascending=False))

st.sidebar.header("Project pipeline")
st.sidebar.markdown("1. Simulate six fault classes in LTspice\n2. Extract eight waveform features\n3. Train Random Forest and 1D-CNN\n4. Compare held-out metrics")
st.sidebar.caption("A confidence score is not a guarantee of correctness.")
