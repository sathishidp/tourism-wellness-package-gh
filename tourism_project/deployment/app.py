"""
Streamlit app for Wellness Tourism prediction.
- Loads the trained pipeline (best_pipeline.joblib) from the Hugging Face model repo at startup.
- Provides single-record input form and CSV batch upload for predictions.
- Designed to run as a Streamlit app (suitable for HF Spaces using the Streamlit runtime).
"""

import os
import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

# --------------------------
# Configuration (update if needed)
# --------------------------
HF_MODEL_REPO = "sathishaiuse/wellness-tourism-model-best"
MODEL_FILENAME = "best_pipeline.joblib"
MODEL_CACHE_PATH = "/tmp/best_pipeline.joblib"  # predictable local path

# --------------------------
# Utility: download & load model
# --------------------------
def load_pipeline_from_hf(repo_id: str = HF_MODEL_REPO, filename: str = MODEL_FILENAME, token: str | None = None):
    """
    Download pipeline from HF model repo and load with joblib.
    If token (string) provided, it will be passed to hf_hub_download.
    """
    try:
        # hf_hub_download caches; using token when provided allows access to private repos
        if token:
            model_path = hf_hub_download(repo_id=repo_id, filename=filename, token=token)
        else:
            model_path = hf_hub_download(repo_id=repo_id, filename=filename)
        pipeline = joblib.load(model_path)
        return pipeline
    except Exception as e:
        raise RuntimeError(f"Failed to download/load model from {repo_id}/{filename}: {e}")

# --------------------------
# Load Model at Startup
# --------------------------
st.set_page_config(page_title="Wellness Tourism - Purchase Predictor", layout="wide")
st.title("Wellness Tourism — Purchase Probability Predictor")

# Provide instructions for private models
hf_token = os.getenv("HF_TOKEN", None)  # Spaces can set this as a secret env var

with st.spinner("Loading model from Hugging Face..."):
    try:
        model_pipeline = load_pipeline_from_hf(token=hf_token)
        st.success("Model loaded successfully.")
    except Exception as e:
        st.error(f"Model load failed: {e}")
        model_pipeline = None

# --------------------------
# Define expected features (based on dataset)
# Keep types conservative; adjust defaults as needed.
# --------------------------
FEATURES = {
    "Age": {"type": "number", "default": 35},
    "TypeofContact": {"type": "select", "options": ["Self Enquiry", "Company Invited"], "default": "Self Enquiry"},
    "CityTier": {"type": "select", "options": [1, 2, 3], "default": 1},
    "DurationOfPitch": {"type": "number", "default": 10},
    "Occupation": {"type": "select", "options": ["Salaried", "Free Lancer", "Self Employed", "Business"], "default": "Salaried"},
    "Gender": {"type": "select", "options": ["Male", "Female", "Other"], "default": "Male"},
    "NumberOfPersonVisiting": {"type": "number", "default": 2},
    "NumberOfFollowups": {"type": "number", "default": 3},
    "ProductPitched": {"type": "select", "options": ["Basic", "Standard", "Deluxe", "Premium", "Super Deluxe"], "default": "Basic"},
    "PreferredPropertyStar": {"type": "select", "options": [1,2,3,4,5], "default": 3},
    "MaritalStatus": {"type": "select", "options": ["Single", "Married", "Divorced", "Widowed"], "default": "Married"},
    "NumberOfTrips": {"type": "number", "default": 2},
    "Passport": {"type": "select", "options": [0,1], "default": 0},
    "PitchSatisfactionScore": {"type": "select", "options": [1,2,3,4,5], "default": 3},
    "OwnCar": {"type": "select", "options": [0,1], "default": 0},
    "NumberOfChildrenVisiting": {"type": "number", "default": 0},
    "Designation": {"type": "select", "options": ["Executive","Manager","Senior Manager","Junior","Other"], "default": "Executive"},
    "MonthlyIncome": {"type": "number", "default": 22000}
}

# --------------------------
# Sidebar: batch upload & model info
# --------------------------
st.sidebar.header("Batch / Model Options")
st.sidebar.markdown("You can upload a CSV with columns matching the training schema for batch predictions.")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

show_model_meta = st.sidebar.checkbox("Show model pipeline steps", value=False)
if show_model_meta and model_pipeline is not None:
    st.sidebar.write("Pipeline steps:", list(model_pipeline.named_steps.keys()))

# Option to use HF token if model repo is private (helpful locally)
if hf_token is None:
    token_input = st.sidebar.text_input("Hugging Face token (optional, for private model)", type="password")
    if token_input:
        # attempt reload with token
        with st.spinner("Reloading model with provided token..."):
            try:
                model_pipeline = load_pipeline_from_hf(token=token_input)
                st.sidebar.success("Model reloaded with provided token.")
            except Exception as e:
                st.sidebar.error(f"Reload failed: {e}")

# --------------------------
# Main: Single-record form and batch upload handling
# --------------------------
st.header("Single Record Prediction")
st.markdown("Fill the fields for one customer and click **Predict**.")

# Create form for single record input
with st.form("single_record_form"):
    cols = st.columns(4)
    inputs = {}
    i = 0
    for feat, meta in FEATURES.items():
        col = cols[i % 4]
        if meta["type"] == "number":
            val = col.number_input(feat, value=float(meta["default"]))
        elif meta["type"] == "select":
            val = col.selectbox(feat, meta["options"], index=meta["options"].index(meta["default"]))
        else:
            val = col.text_input(feat, value=str(meta.get("default", "")))
        inputs[feat] = val
        i += 1
    submit = st.form_submit_button("Predict single record")

if submit:
    if model_pipeline is None:
        st.error("Model not loaded. Cannot run prediction.")
    else:
        try:
            df_input = pd.DataFrame([inputs])
            st.write("Input preview:")
            st.dataframe(df_input)

            preds = model_pipeline.predict(df_input)
            probs = None
            if hasattr(model_pipeline.named_steps["model"], "predict_proba"):
                probs = model_pipeline.predict_proba(df_input)[:, 1]

            result_df = pd.DataFrame({"prediction": preds, "probability": probs})
            st.success("Prediction complete")
            st.table(result_df)
        except Exception as e:
            st.exception(f"Prediction failed: {e}")

st.markdown("---")

# --------------------------
# Batch predictions via CSV upload
# --------------------------
st.header("Batch Predictions (CSV upload)")
st.markdown("Upload a CSV with the same columns as used in training. Predictions will be returned as a downloadable CSV.")

if uploaded_file is not None:
    try:
        batch_df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data (first 5 rows):")
        st.dataframe(batch_df.head())

        if model_pipeline is None:
            st.error("Model not loaded. Cannot run batch predictions.")
        else:
            try:
                preds = model_pipeline.predict(batch_df)
                probs = None
                if hasattr(model_pipeline.named_steps["model"], "predict_proba"):
                    probs = model_pipeline.predict_proba(batch_df)[:, 1]
                out = batch_df.copy()
                out["prediction"] = preds
                if probs is not None:
                    out["probability"] = probs
                csv_bytes = out.to_csv(index=False).encode("utf-8")
                st.success(f"Predictions added. {len(out)} rows processed.")
                st.download_button("Download predictions CSV", data=csv_bytes, file_name="predictions.csv", mime="text/csv")
                st.dataframe(out.head())
            except Exception as e:
                st.exception(f"Batch prediction failed: {e}")
    except Exception as e:
        st.exception(f"Failed to read uploaded CSV: {e}")
else:
    st.info("No CSV uploaded yet. Use the sidebar to upload a file.")
    
# --------------------------
# Footer / tips
# --------------------------
st.markdown("---")
st.caption("Ensure uploaded CSV columns exactly match the training schema (same names & types). If your model uses encoded categorical columns internally, pass original categorical values (the pipeline will apply encoding).")
