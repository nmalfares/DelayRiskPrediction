"""
Construction Project Delay Risk Predictor
------------------------------------------
A Streamlit app that uses two trained models to predict:
  1. Whether a project WILL be delayed (YES / NO)
  2. How SEVERE that delay is likely to be (Low / Medium / High)

Both models are scikit-learn Pipelines (preprocessing + RandomForestClassifier)
that were trained on real project records, so this app just needs to collect
the same input fields the models were trained on, run .predict() /
.predict_proba(), and display the result nicely.

To run this app:
    python -m streamlit run app.py
(On Windows, use "python -m streamlit run app.py" if the "streamlit" command
isn't recognized directly.)
"""

import joblib
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Construction Delay Risk Predictor",
    page_icon="🏗️",
    layout="centered",
)

st.title("🏗️ Construction Project Delay Risk Predictor")
st.write(
    "Enter the details of a project below to predict whether it is likely "
    "to be delayed, and how severe that delay is expected to be."
)


# ---------------------------------------------------------------------------
# Load the trained models (cached so this only happens once per session)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_models():
    binary_model = joblib.load("binary_delay_model.joblib")
    severity_model = joblib.load("severity_delay_model.joblib")
    return binary_model, severity_model


try:
    binary_model, severity_model = load_models()
except FileNotFoundError:
    st.error(
        "Couldn't find binary_delay_model.joblib and/or "
        "severity_delay_model.joblib. Make sure both files are in the same "
        "folder as app.py."
    )
    st.stop()


# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------
st.header("Project Details")

col1, col2 = st.columns(2)

with col1:
    budget_kd = st.number_input(
        "Budget (KD)",
        min_value=0.0,
        value=500_000.0,
        step=1000.0,
        help="Total project budget in Kuwaiti Dinar.",
    )
    phase_type = st.selectbox(
        "Phase Type",
        options=[
            "bidding",
            "construction",
            "construction_recompletion",
            "design",
            "design_build",
            "other",
        ],
        help="The project phase this record refers to.",
    )

with col2:
    planned_duration_days = st.number_input(
        "Planned Duration (days)",
        min_value=1,
        value=365,
        step=1,
        help="How many days the project was originally scheduled to take.",
    )
    contractor_origin = st.selectbox(
        "Contractor Origin",
        options=["Local", "Unknown"],
        help="Whether the contractor is local or unknown/unspecified.",
    )

source_sheet = st.selectbox(
    "Project Category",
    options=["Bidding", "Construction", "Design"],
    help="Which stage/category this project record was sourced from.",
)

st.subheader("Reported Delay Risk Factors")
st.caption("Check any factors that are known or expected to affect this project.")

r1, r2, r3 = st.columns(3)
with r1:
    delay_reason_permits = st.checkbox("Permit issues")
    delay_reason_site = st.checkbox("Site conditions")
with r2:
    delay_reason_contractor = st.checkbox("Contractor issues")
    delay_reason_design = st.checkbox("Design issues")
with r3:
    delay_reason_contractual = st.checkbox("Contractual issues")
    delay_reason_scope_change = st.checkbox("Scope change")


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
SEVERITY_COLORS = {
    "Low": "#2e7d32",     # green
    "Medium": "#f9a825",  # amber
    "High": "#c62828",    # red
}
SEVERITY_RECOMMENDATIONS = {
    "Low": "Delay risk is minor. Continue standard monitoring and reporting.",
    "Medium": "Noticeable delay risk. Review the flagged risk factors with "
    "the project team and consider a mitigation plan.",
    "High": "Significant delay risk. Escalate to project leadership, "
    "reassess timeline and resources, and address flagged risk factors "
    "immediately.",
}

if st.button("Predict Delay Risk", type="primary"):
    input_df = pd.DataFrame(
        [
            {
                "budget_kd": budget_kd,
                "planned_duration_days": planned_duration_days,
                "delay_reason_permits": int(delay_reason_permits),
                "delay_reason_site": int(delay_reason_site),
                "delay_reason_contractor": int(delay_reason_contractor),
                "delay_reason_design": int(delay_reason_design),
                "delay_reason_contractual": int(delay_reason_contractual),
                "delay_reason_scope_change": int(delay_reason_scope_change),
                "phase_type": phase_type,
                "contractor_origin": contractor_origin,
                "source_sheet": source_sheet,
            }
        ]
    )

    # --- Binary prediction (Will it be delayed?) ---
    binary_pred = binary_model.predict(input_df)[0]
    binary_proba = binary_model.predict_proba(input_df)[0]
    delayed = bool(binary_pred == 1)
    delay_confidence = binary_proba[1] if delayed else binary_proba[0]

    # --- Severity prediction (How bad is the delay?) ---
    severity_pred = severity_model.predict(input_df)[0]
    severity_proba = severity_model.predict_proba(input_df)[0]
    severity_classes = severity_model.named_steps["clf"].classes_
    severity_df = pd.DataFrame(
        {"Severity": severity_classes, "Probability": severity_proba}
    ).set_index("Severity").reindex(["Low", "Medium", "High"])

    st.header("Results")

    # Delay YES/NO box
    if delayed:
        st.error(
            f"### ⚠️ Delay Predicted: YES\n"
            f"Confidence: **{delay_confidence:.0%}**"
        )
    else:
        st.success(
            f"### ✅ Delay Predicted: NO\n"
            f"Confidence: **{delay_confidence:.0%}**"
        )

    # Severity box
    sev_color = SEVERITY_COLORS.get(severity_pred, "#666")
    st.markdown(
        f"""
        <div style="background-color:{sev_color}22; border-left: 6px solid {sev_color};
                    padding: 12px 16px; border-radius: 6px; margin-top: 8px;">
            <h4 style="color:{sev_color}; margin:0;">
                Predicted Severity: {severity_pred}
            </h4>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    st.subheader("Severity Probability Distribution")
    st.bar_chart(severity_df)

    st.subheader("Recommendation")
    st.info(SEVERITY_RECOMMENDATIONS.get(severity_pred, "No recommendation available."))

    with st.expander("See raw model inputs"):
        st.dataframe(input_df)


# ---------------------------------------------------------------------------
# Sidebar: about this app
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("About")
    st.write(
        "This app uses two Random Forest models trained on historical "
        "construction project data:\n\n"
        "- **Binary model** — predicts whether a project will be delayed.\n"
        "- **Severity model** — predicts how severe the delay will be "
        "(Low / Medium / High), if one occurs."
    )
    st.caption(
        "Predictions are estimates based on historical patterns, not "
        "guarantees. Use alongside professional project judgment."
    )
