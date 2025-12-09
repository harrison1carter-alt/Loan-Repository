# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import io
import json

st.set_page_config(page_title="Loan Approval Predictor", layout="centered")
st.title("Loan Approval Predictor")

@st.cache_resource
def load_artifacts():
    # Load model
    try:
        model = joblib.load("my_model.pkl")
    except Exception as e:
        st.error(f"Could not load model 'my_model.pkl'. Make sure it is in the app folder. Error: {e}")
        return None, None

    # Load feature names (expected input columns)
    try:
        with open("feature_names.pkl", "rb") as f:
            feature_names = pickle.load(f)
        if isinstance(feature_names, (set, tuple)):
            feature_names = list(feature_names)
    except FileNotFoundError:
        feature_names = None
    except Exception as e:
        st.error(f"Error loading 'feature_names.pkl': {e}")
        feature_names = None

    return model, feature_names

model, feature_names = load_artifacts()
if model is None:
    st.stop()

st.write("**Note:** This app expects input data to match the features used to train the saved model.")
if feature_names is not None:
    st.write(f"Model expects {len(feature_names)} features.")
    if len(feature_names) <= 50:
        st.write("Feature names (preview):")
        st.write(feature_names[:200])
    else:
        st.write("Feature names list is long. Upload a CSV with the same columns as your training data.")

st.markdown("---")

st.header("Input options (choose one)")
st.markdown(
    "1. Upload a CSV file containing the **processed** features.\n"
    "2. Paste a single-row JSON mapping `feature_name: value`.\n"
)

uploaded_file = st.file_uploader("Upload CSV (processed features)", type=["csv"])
json_input = st.text_area(
    "Or paste a single-row JSON here",
    height=120
)

def prepare_df_from_uploaded(df_raw, expected_cols):
    if expected_cols is not None:
        return df_raw.reindex(columns=expected_cols, fill_value=0)
    return df_raw.copy()

input_df = None
if uploaded_file is not None:
    try:
        df_raw = pd.read_csv(uploaded_file)
        st.write("Uploaded CSV preview:")
        st.dataframe(df_raw.head())
        input_df = prepare_df_from_uploaded(df_raw, feature_names)

        # FIX: Persist input_df
        st.session_state["input_df"] = input_df

    except Exception as e:
        st.error(f"Error reading uploaded CSV: {e}")

elif json_input.strip() != "":
    try:
        obj = json.loads(json_input)
        if isinstance(obj, dict):
            df_raw = pd.DataFrame([obj])
            st.write("Parsed JSON input:")
            st.json(obj)
            input_df = prepare_df_from_uploaded(df_raw, feature_names)

            # FIX: Persist input_df
            st.session_state["input_df"] = input_df

        else:
            st.error("JSON must be a dict.")
    except Exception as e:
        st.error(f"Could not parse JSON: {e}")


else:
    st.info("Upload a processed CSV or paste JSON to make predictions.")

if input_df is not None and feature_names is not None:
    missing = [c for c in feature_names if c not in input_df.columns]
    extra = [c for c in input_df.columns if c not in feature_names]
    if missing:
        st.warning(f"Missing columns filled with 0: {missing[:20]}...")
    if extra:
        st.info(f"Extra columns ignored: {extra[:20]}...")
# Restore saved input_df if it exists
if "input_df" in st.session_state:
    input_df = st.session_state["input_df"]

# ------------------------------
#      PREDICT BUTTON
# ------------------------------
if st.button("Predict") and input_df is not None:
    try:
        if feature_names is not None:
            X = input_df.reindex(columns=feature_names, fill_value=0)
        else:
            X = input_df.copy()

        proba = model.predict_proba(X)[:, 1]
        preds = model.predict(X)

        results = X.copy()
        results["predicted_class"] = preds
        results["probability"] = proba

        # Store in session_state (critical for preventing disappearing results)
        st.session_state["preds"] = preds
        st.session_state["proba"] = proba
        st.session_state["results"] = results

        st.success("Prediction complete. See results below.")
        st.dataframe(results.head())

    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.exception(e)

# ------------------------------
#   SHOW LOOKUP SECTION ANYTIME
# ------------------------------
if "preds" in st.session_state:
    st.markdown("### Look up a specific result by index")

    user_idx = st.number_input(
        "Enter an index (0 to {}):".format(len(st.session_state["preds"]) - 1),
        min_value=0,
        max_value=len(st.session_state["preds"]) - 1,
        step=1,
        key="lookup_idx"
    )

    if st.button("Submit"):
        preds = st.session_state["preds"]
        proba = st.session_state["proba"]
        results = st.session_state["results"]

        st.markdown("---")
        st.subheader(f"Result for index {user_idx}")

        pred_val = preds[user_idx]
        proba_val = proba[user_idx]

        if pred_val == 1:
            st.success(f"APPROVED — Probability: {proba_val:.3f}")
        else:
            st.error(f"DENIED — Probability: {proba_val:.3f}")

st.markdown("---")
st.subheader("Notes & recommendations")
st.markdown(
    """
- This app expects **preprocessed input**, matching the features in `feature_names.pkl`.
- If you need help generating a single-row preprocessed CSV, I can help you export it from your notebook.
"""
)
