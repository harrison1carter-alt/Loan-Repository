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
        # Ensure it's a list
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
st.markdown("1. Upload a CSV file containing the **processed** features (columns must match those in `feature_names.pkl`).  \n"
            "2. Paste a single-row JSON mapping `feature_name: value` for a single prediction.  \n"
            "If you do not have a preprocessed CSV, see the notes below about preprocessing.")

uploaded_file = st.file_uploader("Upload CSV (processed features)", type=["csv"])
json_input = st.text_area("Or paste a single-row JSON here (example: {\"feat1\": 10, \"feat2_catA\": 1})", height=120)

def prepare_df_from_uploaded(df_raw, expected_cols):
    # If feature names available, try to align columns exactly
    if expected_cols is not None:
        # Reindex to expected cols: missing => fill 0, extra => drop
        df = df_raw.reindex(columns=expected_cols, fill_value=0)
    else:
        df = df_raw.copy()
    return df

input_df = None
if uploaded_file is not None:
    try:
        df_raw = pd.read_csv(uploaded_file)
        st.write("Uploaded CSV preview:")
        st.dataframe(df_raw.head())
        input_df = prepare_df_from_uploaded(df_raw, feature_names)
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
        else:
            st.error("JSON must be a single JSON object (a dict mapping feature_name -> value).")
    except Exception as e:
        st.error(f"Could not parse JSON: {e}")

else:
    st.info("Upload a processed CSV or paste a single-row JSON to make predictions.")

# If we have feature names but the uploaded input doesn't match, show guidance
if input_df is not None and feature_names is not None:
    # column mismatch diagnostics
    missing = [c for c in feature_names if c not in input_df.columns]
    extra = [c for c in input_df.columns if c not in feature_names]
    if missing:
        st.warning(f"Missing columns (will be filled with 0): {missing[:20]}{('...' if len(missing)>20 else '')}")
    if extra:
        st.info(f"Extra columns in your input that will be ignored: {extra[:20]}{('...' if len(extra)>20 else '')}")

# Predict button
if st.button("Predict") and input_df is not None:
    try:
        # Ensure column order matches training
        if feature_names is not None:
            X = input_df.reindex(columns=feature_names, fill_value=0)
        else:
            X = input_df.copy()

        # If model expects 1d numpy array, ensure shape
        # Many scikit-learn models accept DataFrame directly.
        proba = model.predict_proba(X)[:, 1]
        preds = model.predict(X)

        results = X.copy()
        results["predicted_class"] = preds
        results["probability"] = proba
        st.success("Prediction complete. See results below.")
        st.dataframe(results.head())
        # show a clearer message for first row
    
        # --- USER INDEX LOOKUP SECTION ---
        st.markdown("### Look up a specific result by index")
        
        user_idx = st.number_input(
            "Enter an index (0 to {}):".format(len(preds) - 1),
            min_value=0,
            max_value=len(preds) - 1,
            step=1
        )
        
        if st.button("Submit"):
            st.markdown("---")
            st.subheader(f"Result for index {user_idx}")
        
            selected_pred = preds[user_idx]
            selected_proba = proba[user_idx]
        
            if selected_pred == 1:
                st.success(f"APPROVED — Probability: {selected_proba:.3f}")
            else:
                st.error(f"DENIED — Probability: {selected_proba:.3f}")
        
            # Optional: show the row of feature values too
            st.write("Feature values:")
            st.dataframe(results.iloc[[user_idx]])


    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.exception(e)

st.markdown("---")
st.subheader("Notes & recommendations")
st.markdown(
    """
- **Important:** You saved your model but **not** the preprocessing pipeline. That means this app *expects* you to upload data that is already processed the **same way** you processed training data (one-hot columns already created, same column names and order).
- If you do NOT have a preprocessed CSV, please re-run your Colab notebook, export a single-row CSV with the **final model input columns** (the columns listed in `feature_names.pkl`), and upload that CSV here.
- In future projects, save your full pipeline as one object, e.g. `joblib.dump(pipeline, 'pipeline.pkl')`, where `pipeline = Pipeline([('preproc', preprocessor), ('clf', model)])`. Then the app can accept raw inputs and perform preprocessing automatically.
- If you need help generating a single-row preprocessed CSV from your original dataset, I can provide a small script to do that in Colab.
"""
)
