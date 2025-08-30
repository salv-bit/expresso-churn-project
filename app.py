import json
import joblib
import pandas as pd
import streamlit
# app.py
import json
import numpy as np
import pandas as pd
import streamlit as st
from joblib import load
from pathlib import Path
from scipy.special import expit

st.set_page_config(page_title="Expresso Churn Predictor", page_icon="📶", layout="wide")
st.title("📶 Expresso Churn Predictor")
st.caption("Enter feature values below and click Predict to estimate churn probability.")

# ---- Paths (must match what's in artifacts/) ----
MODEL_PATH = Path("artifacts/churn_model.joblib")         # you renamed to this
META_PATH  = Path("artifacts/columns_meta.json")          # created by training script
METRICS    = Path("artifacts/metrics.json")               # optional

# ---- Load artifacts ----
if not MODEL_PATH.exists():
    st.error(f"Model not found at {MODEL_PATH}. Train first or check filename.")
    st.stop()
if not META_PATH.exists():
    st.error(f"Metadata not found at {META_PATH}. Train first or check filename.")
    st.stop()

pipe = load(MODEL_PATH)
meta = json.loads(META_PATH.read_text(encoding="utf-8"))

num_cols     = meta.get("numeric_columns", [])
cat_cols     = meta.get("categorical_columns", [])
cat_choices  = meta.get("categorical_choices", {})  # may be missing if not saved
target       = meta.get("target", "CHURN")

# ---- Metrics panel (optional) ----
with st.sidebar:
    st.header("Model Metrics")
    if METRICS.exists():
        m = json.loads(METRICS.read_text(encoding="utf-8"))
        st.code({"accuracy": m.get("accuracy"), "roc_auc": m.get("roc_auc")}, language="json")
    else:
        st.write("metrics.json not found")

# ---- Input form ----
st.subheader("Input Features")
with st.form("input_form"):
    c1, c2 = st.columns(2)
    inputs = {}

    # numeric widgets
    for i, col in enumerate(num_cols):
        with (c1 if i % 2 == 0 else c2):
            inputs[col] = st.number_input(col, value=0.0, step=1.0, format="%.4f")

    # categorical widgets
    for i, col in enumerate(cat_cols):
        with (c1 if i % 2 == 0 else c2):
            choices = cat_choices.get(col, [])
            if choices:
                inputs[col] = st.selectbox(col, choices, index=0)
            else:
                inputs[col] = st.text_input(col, value="")

    submitted = st.form_submit_button("Predict")

# ---- Predict ----
if submitted:
    X_input = pd.DataFrame({k: [v] for k, v in inputs.items()})
    try:
        if hasattr(pipe, "predict_proba"):
            prob = float(pipe.predict_proba(X_input)[:, 1][0])
        else:
            score = pipe.decision_function(X_input)
            prob = float(expit(score if np.ndim(score) == 0 else score[0]))
        pred = int(prob >= 0.5)

        st.success(f"Prediction: **{'Churn' if pred==1 else 'Not Churn'}**")
        st.metric("Churn probability", f"{prob*100:.2f}%")
    except Exception as e:
        st.error(f"Failed to predict: {e}")
