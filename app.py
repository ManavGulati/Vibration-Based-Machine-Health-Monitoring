import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Machine Health Monitor",
    page_icon="⚙️",
    layout="wide"
)

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"


# =====================================================
# 2. LOAD MODELS AND FEATURE CONFIGURATIONS
# =====================================================

@st.cache_resource
def load_models():
    from xgboost import XGBClassifier, XGBRegressor

    classifier = XGBClassifier()
    classifier.load_model(MODEL_DIR / "fault_classifier.json")

    regressor = XGBRegressor()
    regressor.load_model(MODEL_DIR / "severity_regressor.json")

    encoder = joblib.load(MODEL_DIR / "label_encoder.pkl")

    with open(MODEL_DIR / "feature_columns.json", "r") as f:
        feature_columns = json.load(f)

    with open(MODEL_DIR / "severity_feature_columns.json", "r") as f:
        severity_feature_columns = json.load(f)

    return (
        classifier,
        regressor,
        encoder,
        feature_columns,
        severity_feature_columns
    )


try:
    (
        fault_model,
        severity_model,
        label_encoder,
        FEATURE_COLUMNS,
        SEVERITY_FEATURE_COLUMNS
    ) = load_models()
    models_loaded = True

except Exception as e:
    models_loaded = False
    st.error(f"Could not load models: {e}")


# =====================================================
# 3. PREDICTION FUNCTION
# =====================================================

def predict_machine_health(row):
    """
    row must contain the same input features used during training.
    It must be a pandas Series or dictionary.
    """

    input_df = pd.DataFrame(
        [{column: row[column] for column in FEATURE_COLUMNS}]
    )

    # Fault classification
    predicted_encoded = int(fault_model.predict(input_df)[0])
    fault_type = label_encoder.inverse_transform(
        [predicted_encoded]
    )[0]

    probabilities = fault_model.predict_proba(input_df)[0]
    confidence = float(np.max(probabilities)) * 100

    # Severity estimation
    if fault_type == "healthy":
        severity = 0
        status = "Healthy"

    else:
        severity_input = input_df.copy()
        severity_input["Condition"] = fault_type

        severity_input = pd.get_dummies(
            severity_input,
            columns=["Condition"],
            dtype=int
        )

        severity_input = severity_input.reindex(
            columns=SEVERITY_FEATURE_COLUMNS,
            fill_value=0
        )

        predicted_severity = float(
            severity_model.predict(severity_input)[0]
        )

        severity = int(
            np.clip(np.rint(predicted_severity), 1, 3)
        )

        if severity == 1:
            status = "Mild Fault"
        elif severity == 2:
            status = "Moderate Fault"
        else:
            status = "Severe Fault"

    return {
        "fault": fault_type,
        "severity": severity,
        "status": status,
        "confidence": confidence,
        "probabilities": probabilities,
        "features": input_df.iloc[0].to_dict()
    }


# =====================================================
# 4. DASHBOARD HEADER
# =====================================================

st.title("⚙️ Machine Health Monitoring System")
st.markdown("### Vibration-Based Predictive Maintenance")
st.caption(
    "AI-powered fault detection and machine degradation estimation"
)
st.divider()


# =====================================================
# 5. SIDEBAR
# =====================================================

st.sidebar.title("⚙️ Control Panel")
st.sidebar.info(
    "Upload a MATLAB simulation feature CSV. "
    "Each row should represent one simulated operating condition."
)

st.sidebar.markdown("### Model Information")
st.sidebar.write("**Fault classification:** XGBoost Classifier")
st.sidebar.write("**Severity estimation:** XGBoost Regressor")
st.sidebar.write("**Training source:** MATLAB rotor fault simulation")
st.sidebar.write("**Input:** Extracted vibration features")

st.sidebar.warning(
    "These models are trained on simulation data. "
    "Their performance on physical machines has not yet been established."
)


# =====================================================
# 6. UPLOAD FEATURE DATASET
# =====================================================

st.subheader("📂 Upload MATLAB Simulation Data")

uploaded_file = st.file_uploader(
    "Upload the rotor_fault_realistic.csv file",
    type=["csv"]
)

if uploaded_file is not None and models_loaded:

    try:
        data = pd.read_csv(uploaded_file)

        st.success("CSV file loaded successfully!")

        missing_columns = [
            col for col in FEATURE_COLUMNS
            if col not in data.columns
        ]

        if missing_columns:
            st.error(
                "The uploaded CSV is missing required model input "
                f"columns: {missing_columns}"
            )
            st.stop()

        # Validate feature values
        numeric_features = data[FEATURE_COLUMNS].apply(
            pd.to_numeric,
            errors="coerce"
        )

        invalid_rows = numeric_features.isnull().any(axis=1)

        if invalid_rows.any():
            st.error(
                f"{int(invalid_rows.sum())} row(s) contain missing or "
                "non-numeric values in required feature columns. "
                "Please correct the CSV before analysis."
            )
            st.stop()

        st.write("### Dataset Preview")
        st.write(f"Number of records: **{len(data)}**")
        st.dataframe(data.head(), use_container_width=True)

        if len(data) == 0:
            st.error("The uploaded CSV contains no records.")
            st.stop()

        if len(data) > 1:
            selected_row = st.number_input(
                "Select simulation record (row number, starting from 0)",
                min_value=0,
                max_value=len(data) - 1,
                value=0,
                step=1
            )
        else:
            selected_row = 0

        selected_record = data.iloc[int(selected_row)]
        selected_features = numeric_features.iloc[int(selected_row)]

        st.caption(
            "The selected row is passed to the models using only the "
            "input features used during training. Condition and Severity "
            "columns, if present, are not used as model inputs."
        )

        # =====================================================
        # 7. DISPLAY OPERATING CONDITIONS
        # =====================================================

        st.subheader("⚙️ Operating Conditions")

        info_cols = st.columns(3)

        if "RPM" in data.columns:
            info_cols[0].metric(
                "Rotational Speed",
                f'{selected_record["RPM"]} RPM'
            )

        if "NominalLoad_Percent" in data.columns:
            info_cols[1].metric(
                "Nominal Load",
                f'{selected_record["NominalLoad_Percent"]}%'
            )

        if "Temperature_C" in data.columns:
            info_cols[2].metric(
                "Temperature",
                f'{selected_record["Temperature_C"]} °C'
            )

        # =====================================================
        # 8. FEATURE VISUALIZATION
        # =====================================================

        st.subheader("📊 Vibration Feature Analysis")

        display_features = [
            col for col in FEATURE_COLUMNS
            if col in selected_features.index
        ]

        feature_display = pd.DataFrame({
            "Feature": display_features,
            "Value": [
                float(selected_features[col])
                for col in display_features
            ]
        })

        st.dataframe(
            feature_display,
            use_container_width=True,
            hide_index=True
        )

        # =====================================================
        # 9. PREDICTION
        # =====================================================

        st.divider()
        st.subheader("🤖 AI Machine Health Analysis")

        if st.button(
            "Analyse Machine Health",
            type="primary",
            use_container_width=True
        ):
            result = predict_machine_health(selected_features)
            st.session_state["analysis_result"] = result
            st.session_state["analysis_row"] = int(selected_row)

        if (
            "analysis_result" in st.session_state
            and st.session_state.get("analysis_row") == int(selected_row)
        ):
            result = st.session_state["analysis_result"]

            st.subheader("Prediction Results")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Detected Fault",
                result["fault"]
            )

            col2.metric(
                "Severity Level",
                f'{result["severity"]} / 3'
            )

            col3.metric(
                "Classification Confidence",
                f'{result["confidence"]:.2f}%'
            )

            if result["status"] == "Healthy":
                st.success("✅ Machine condition: HEALTHY")
            elif result["severity"] == 1:
                st.info("⚠️ Machine condition: MILD FAULT")
            elif result["severity"] == 2:
                st.warning("⚠️ Machine condition: MODERATE FAULT")
            else:
                st.error("🚨 Machine condition: SEVERE FAULT")

            # =================================================
            # 10. CLASS PROBABILITIES
            # =================================================

            st.subheader("Fault Classification Probabilities")

            probability_df = pd.DataFrame({
                "Machine Condition": label_encoder.classes_,
                "Probability": result["probabilities"]
            })

            probability_df = probability_df.sort_values(
                "Probability",
                ascending=False
            )

            st.bar_chart(
                probability_df.set_index("Machine Condition")
            )

            # =================================================
            # 11. ACTUAL LABEL COMPARISON (IF PRESENT)
            # =================================================

            if "Condition" in data.columns:
                st.subheader("Simulation Label Comparison")

                actual_condition = str(selected_record["Condition"])
                st.write(f"**Actual simulated condition:** {actual_condition}")
                st.write(f"**Predicted condition:** {result['fault']}")

                if "Severity" in data.columns:
                    actual_severity = int(selected_record["Severity"])
                    st.write(f"**Actual simulated severity:** {actual_severity}")
                    st.write(f"**Predicted severity:** {result['severity']}")

                st.caption(
                    "Actual labels are shown only for comparison. "
                    "They are not supplied to the fault classifier."
                )

    except Exception as e:
        st.error(f"Error processing uploaded file: {e}")


elif uploaded_file is not None and not models_loaded:
    st.error(
        "Models could not be loaded. Check that the five model/configuration "
        "files are present in the models folder."
    )

else:
    st.info(
        "Upload the MATLAB simulation feature CSV to begin machine health analysis."
    )


# =====================================================
# 12. FOOTER
# =====================================================

st.divider()
st.caption(
    "Vibration-Based Machine Health Detection | "
    "XGBoost Predictive Maintenance System"
)
