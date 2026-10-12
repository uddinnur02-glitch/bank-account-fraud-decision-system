# --------------------------------------------------
# Imports
# --------------------------------------------------

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from catboost import CatBoostClassifier, Pool


# --------------------------------------------------
# Paths
# --------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"


# --------------------------------------------------
# Load model artifacts
# --------------------------------------------------

@st.cache_resource
def load_artifacts():

    model = CatBoostClassifier()

    model.load_model(
        str(
            MODELS_DIR
            / "catboost_fraud_model.cbm"
        )
    )

    calibrator = joblib.load(
        MODELS_DIR
        / "platt_calibrator.joblib"
    )

    with open(
        MODELS_DIR
        / "risk_thresholds.json",
        "r"
    ) as f:

        thresholds = json.load(f)

    with open(
        MODELS_DIR
        / "model_metadata.json",
        "r"
    ) as f:

        metadata = json.load(f)

    with open(
        MODELS_DIR
        / "feature_schema.json",
        "r"
    ) as f:

        feature_schema = json.load(f)

    return (
        model,
        calibrator,
        thresholds,
        metadata,
        feature_schema
    )


(
    model,
    calibrator,
    thresholds,
    metadata,
    feature_schema
) = load_artifacts()


# --------------------------------------------------
# Feature engineering
# --------------------------------------------------

def engineer_features(application):

    df = application.copy()

    sentinel_features = [
        "prev_address_months_count",
        "current_address_months_count",
        "bank_months_count",
        "session_length_in_minutes",
        "device_distinct_emails_8w"
    ]

    # Missing-value indicator features
    for col in sentinel_features:

        df[
            f"{col}_missing"
        ] = (
            df[col]
            .isna()
            .astype(int)
        )

    missing_flag_cols = [
        f"{col}_missing"
        for col in sentinel_features
    ]

    # Total number of missing information fields
    df["missing_info_count"] = (
        df[
            missing_flag_cols
        ]
        .sum(axis=1)
    )

    # Number of valid phone channels
    df["valid_phone_count"] = (
        df["phone_home_valid"]
        + df["phone_mobile_valid"]
    )

    # Whether both address-history fields exist
    df["address_history_complete"] = (
        df[
            [
                "prev_address_months_count",
                "current_address_months_count"
            ]
        ]
        .notna()
        .all(axis=1)
        .astype(int)
    )

    # Heuristic features used during training
    df["low_identity_consistency"] = (
        df["name_email_similarity"]
        < 0.4
    ).astype(int)

    df["high_credit_request"] = (
        df["proposed_credit_limit"]
        >= 1000
    ).astype(int)

    return df




# --------------------------------------------------
# Inference function
# --------------------------------------------------

def predict_fraud_risk(application_data):

    application = pd.DataFrame(
        [application_data]
    )

    # Reproduce training feature engineering
    application = engineer_features(
        application
    )

    # Exact feature order expected by saved model
    application = application[
        feature_schema[
            "model_features"
        ]
    ].copy()

    # Prepare CatBoost categorical columns
    for col in metadata[
        "categorical_features"
    ]:

        application[col] = (
            application[col]
            .fillna("Missing")
            .astype(str)
        )

    # Raw CatBoost probability
    raw_probability = (
        model
        .predict_proba(
            application
        )[0, 1]
    )

    # Prepare score for Platt calibration
    clipped_probability = np.clip(
        raw_probability,
        1e-6,
        1 - 1e-6
    )

    logit_score = np.log(
        clipped_probability
        / (
            1
            - clipped_probability
        )
    )

    calibrated_probability = (
        calibrator
        .predict_proba(
            np.array(
                [[logit_score]]
            )
        )[0, 1]
    )

    # Risk segmentation
    if (
        calibrated_probability
        >= thresholds[
            "critical_threshold"
        ]
    ):

        risk_band = "Critical"

    elif (
        calibrated_probability
        >= thresholds[
            "high_threshold"
        ]
    ):

        risk_band = "High"

    elif (
        calibrated_probability
        >= thresholds[
            "medium_threshold"
        ]
    ):

        risk_band = "Medium"

    else:

        risk_band = "Low"

    # Manual review policy
    manual_review = (
        calibrated_probability
        >= thresholds[
            "review_threshold"
        ]
    )

    return {
        "raw_model_score":
            float(raw_probability),

        "calibrated_probability":
            float(
                calibrated_probability
            ),

        "risk_band":
            risk_band,

        "manual_review":
            bool(manual_review)
    }


# --------------------------------------------------
# Local SHAP explanation
# --------------------------------------------------

def explain_prediction(application_data):

    application = pd.DataFrame(
        [application_data]
    )

    # Reproduce the same feature engineering
    application = engineer_features(
        application
    )

    # Exact model feature order
    application = application[
        feature_schema[
            "model_features"
        ]
    ].copy()

    # Prepare categorical features
    for col in metadata[
        "categorical_features"
    ]:

        application[col] = (
            application[col]
            .fillna("Missing")
            .astype(str)
        )

    # CatBoost Pool
    application_pool = Pool(
        application,
        cat_features=metadata[
            "categorical_features"
        ]
    )

    # SHAP values
    shap_full = (
        model.get_feature_importance(
            application_pool,
            type="ShapValues"
        )
    )

    # Last value is expected/base value
    shap_values = shap_full[
        0, :-1
    ]

    explanation = pd.DataFrame({
        "Feature":
            application.columns,

        "Value":
            application.iloc[0].values,

        "SHAP_Value":
            shap_values
    })

    explanation[
        "Absolute_SHAP"
    ] = np.abs(
        explanation[
            "SHAP_Value"
        ]
    )

    explanation = (
        explanation
        .sort_values(
            "Absolute_SHAP",
            ascending=False
        )
        .reset_index(drop=True)
    )

    return explanation


# --------------------------------------------------
# App UI
# --------------------------------------------------

st.set_page_config(
    page_title=(
        "Bank Account Fraud Risk System"
    ),
    page_icon="🔎",
    layout="wide"
)

st.title(
    "Bank Account Fraud Detection "
    "& Review Decision System"
)

st.caption(
    "Portfolio demonstration using a "
    "calibrated CatBoost fraud-risk model."
)

st.info(
    "This application is a decision-support "
    "demonstration and should not be used as "
    "an automatic financial approval or "
    "rejection system."
)


# --------------------------------------------------
# Risk band legend
# --------------------------------------------------

st.subheader(
    "Risk Band Guide"
)

legend_col1, legend_col2, legend_col3, legend_col4 = (
    st.columns(4)
)

legend_col1.metric(
    "Low",
    f"< {thresholds['medium_threshold']:.2%}"
)

legend_col2.metric(
    "Medium",
    (
        f"{thresholds['medium_threshold']:.2%} "
        f"to < {thresholds['high_threshold']:.2%}"
    )
)

legend_col3.metric(
    "High",
    (
        f"{thresholds['high_threshold']:.2%} "
        f"to < {thresholds['critical_threshold']:.2%}"
    )
)

legend_col4.metric(
    "Critical",
    f"≥ {thresholds['critical_threshold']:.2%}"
)

st.caption(
    "Applications in the High and Critical bands meet the frozen "
    "manual-review threshold used in this portfolio demonstration."
)


# --------------------------------------------------
# Application input container
# --------------------------------------------------

application_data = {}

defaults = (
    feature_schema[
        "defaults"
    ]
)

categorical_options = (
    feature_schema[
        "categorical_options"
    ]
)


# --------------------------------------------------
# Customer Profile
# --------------------------------------------------

st.subheader(
    "1. Customer Profile"
)

col1, col2, col3 = st.columns(3)

with col1:

    application_data["income"] = (
        st.number_input(
            "Income",
            min_value=0.0,
            max_value=1.0,
            value=float(
                defaults["income"]
            ),
            step=0.1
        )
    )

    application_data[
        "customer_age"
    ] = st.number_input(
        "Customer Age",
        min_value=0,
        max_value=100,
        value=int(
            defaults[
                "customer_age"
            ]
        )
    )

with col2:

    application_data[
        "name_email_similarity"
    ] = st.number_input(
        "Name–Email Similarity",
        min_value=0.0,
        max_value=1.0,
        value=float(
            defaults[
                "name_email_similarity"
            ]
        ),
        step=0.01
    )

    application_data[
        "employment_status"
    ] = st.selectbox(
        "Employment Status",
        categorical_options[
            "employment_status"
        ]
    )

with col3:

    application_data[
        "housing_status"
    ] = st.selectbox(
        "Housing Status",
        categorical_options[
            "housing_status"
        ]
    )

    application_data[
        "email_is_free"
    ] = st.selectbox(
        "Free Email Provider",
        [0, 1]
    )


# --------------------------------------------------
# Contact Information
# --------------------------------------------------

st.subheader(
    "2. Contact Information"
)

col1, col2, col3 = st.columns(3)

with col1:

    application_data[
        "phone_home_valid"
    ] = st.selectbox(
        "Home Phone Valid",
        [0, 1]
    )

with col2:

    application_data[
        "phone_mobile_valid"
    ] = st.selectbox(
        "Mobile Phone Valid",
        [0, 1]
    )

with col3:

    application_data[
        "has_other_cards"
    ] = st.selectbox(
        "Has Other Cards",
        [0, 1]
    )

# --------------------------------------------------
# Address History
# --------------------------------------------------

st.subheader(
    "3. Address History"
)

col1, col2 = st.columns(2)

with col1:

    prev_unknown = st.checkbox(
        "Previous Address Duration Unknown"
    )

    if prev_unknown:

        application_data[
            "prev_address_months_count"
        ] = np.nan

    else:

        application_data[
            "prev_address_months_count"
        ] = st.number_input(
            "Previous Address Months",
            min_value=0.0,
            value=float(
                defaults[
                    "prev_address_months_count"
                ]
            )
        )

with col2:

    current_unknown = st.checkbox(
        "Current Address Duration Unknown"
    )

    if current_unknown:

        application_data[
            "current_address_months_count"
        ] = np.nan

    else:

        application_data[
            "current_address_months_count"
        ] = st.number_input(
            "Current Address Months",
            min_value=0.0,
            value=float(
                defaults[
                    "current_address_months_count"
                ]
            )
        )

# --------------------------------------------------
# Banking & Credit Information
# --------------------------------------------------

st.subheader(
    "4. Banking & Credit Information"
)

col1, col2, col3 = st.columns(3)

with col1:

    application_data[
        "credit_risk_score"
    ] = st.number_input(
        "Credit Risk Score",
        value=int(
            defaults[
                "credit_risk_score"
            ]
        )
    )

    application_data[
        "proposed_credit_limit"
    ] = st.number_input(
        "Proposed Credit Limit",
        min_value=0.0,
        value=float(
            defaults[
                "proposed_credit_limit"
            ]
        )
    )

with col2:

    bank_months_unknown = (
        st.checkbox(
            "Bank Relationship Duration Unknown"
        )
    )

    if bank_months_unknown:

        application_data[
            "bank_months_count"
        ] = np.nan

    else:

        application_data[
            "bank_months_count"
        ] = st.number_input(
            "Bank Relationship Months",
            min_value=0.0,
            value=float(
                defaults[
                    "bank_months_count"
                ]
            )
        )

    application_data[
        "bank_branch_count_8w"
    ] = st.number_input(
        "Bank Branch Count — 8 Weeks",
        min_value=0.0,
        value=float(
            defaults[
                "bank_branch_count_8w"
            ]
        )
    )

with col3:

    application_data[
        "payment_type"
    ] = st.selectbox(
        "Payment Type",
        categorical_options[
            "payment_type"
        ]
    )

    application_data[
        "foreign_request"
    ] = st.selectbox(
        "Foreign Request",
        [0, 1]
    )


# --------------------------------------------------
# Application Information
# --------------------------------------------------

st.subheader(
    "5. Application Information"
)

col1, col2, col3 = st.columns(3)

with col1:

    application_data[
        "days_since_request"
    ] = st.number_input(
        "Days Since Request",
        value=float(
            defaults[
                "days_since_request"
            ]
        )
    )

    application_data[
        "intended_balcon_amount"
    ] = st.number_input(
        "Intended Balance Amount",
        value=float(
            defaults[
                "intended_balcon_amount"
            ]
        )
    )

with col2:

    application_data[
        "zip_count_4w"
    ] = st.number_input(
        "ZIP Activity Count — 4 Weeks",
        min_value=0.0,
        value=float(
            defaults[
                "zip_count_4w"
            ]
        )
    )

    application_data[
        "date_of_birth_distinct_emails_4w"
    ] = st.number_input(
        "Distinct Emails Linked to DOB — 4 Weeks",
        min_value=0.0,
        value=float(
            defaults[
                "date_of_birth_distinct_emails_4w"
            ]
        )
    )

with col3:

    application_data[
        "source"
    ] = st.selectbox(
        "Application Source",
        categorical_options[
            "source"
        ]
    )

    application_data[
        "month"
    ] = st.selectbox(
        "Dataset Month",
        list(range(8))
    )


# --------------------------------------------------
# Transaction / Request Velocity
# --------------------------------------------------

st.subheader(
    "6. Velocity Signals"
)

col1, col2, col3 = st.columns(3)

with col1:

    application_data[
        "velocity_6h"
    ] = st.number_input(
        "Velocity — 6 Hours",
        value=float(
            defaults[
                "velocity_6h"
            ]
        )
    )

with col2:

    application_data[
        "velocity_24h"
    ] = st.number_input(
        "Velocity — 24 Hours",
        value=float(
            defaults[
                "velocity_24h"
            ]
        )
    )

with col3:

    application_data[
        "velocity_4w"
    ] = st.number_input(
        "Velocity — 4 Weeks",
        value=float(
            defaults[
                "velocity_4w"
            ]
        )
    )

# --------------------------------------------------
# Device & Session Information
# --------------------------------------------------

st.subheader(
    "7. Device & Session Information"
)

col1, col2, col3 = st.columns(3)

with col1:

    application_data[
        "device_os"
    ] = st.selectbox(
        "Device Operating System",
        categorical_options[
            "device_os"
        ]
    )

    application_data[
        "keep_alive_session"
    ] = st.selectbox(
        "Keep Alive Session",
        [0, 1]
    )

with col2:

    session_unknown = st.checkbox(
        "Session Length Unknown"
    )

    if session_unknown:

        application_data[
            "session_length_in_minutes"
        ] = np.nan

    else:

        application_data[
            "session_length_in_minutes"
        ] = st.number_input(
            "Session Length (Minutes)",
            min_value=0.0,
            value=float(
                defaults[
                    "session_length_in_minutes"
                ]
            )
        )

with col3:

    device_email_unknown = (
        st.checkbox(
            "Device Email Count Unknown"
        )
    )

    if device_email_unknown:

        application_data[
            "device_distinct_emails_8w"
        ] = np.nan

    else:

        application_data[
            "device_distinct_emails_8w"
        ] = st.number_input(
            "Distinct Device Emails — 8 Weeks",
            min_value=0.0,
            value=float(
                defaults[
                    "device_distinct_emails_8w"
                ]
            )
        )


# --------------------------------------------------
# Prediction
# --------------------------------------------------

st.divider()

if st.button(
    "Evaluate Fraud Risk",
    type="primary"
):

    result = predict_fraud_risk(
        application_data
    )

    explanation = explain_prediction(
        application_data
    )

    st.divider()

    st.subheader(
        "Risk Assessment"
    )

    metric1, metric2, metric3 = (
        st.columns(3)
    )

    metric1.metric(
        "Calibrated Fraud Probability",
        f"{result['calibrated_probability']:.2%}"
    )

    metric2.metric(
        "Risk Band",
        result[
            "risk_band"
        ]
    )

    metric3.metric(
        "Manual Review",
        (
            "Required"
            if result[
                "manual_review"
            ]
            else "Not Required"
        )
    )

    # ----------------------------------------------
    # Decision message
    # ----------------------------------------------

    if result["risk_band"] == "Critical":

        st.error(
            "Critical-risk application: strong priority for manual review."
        )

    elif result["risk_band"] == "High":

        st.warning(
            "High-risk application: manual review is recommended."
        )

    elif result["risk_band"] == "Medium":

        st.info(
            "Medium-risk application: below the primary manual-review "
            "threshold, but risk is elevated relative to the Low band."
        )

    else:

        st.success(
            "Low-risk application: below the primary manual-review threshold."
        )

    st.caption(
        "The review recommendation is based on the frozen month-6 "
        "operating threshold selected during model development. "
        "It is a decision-support rule, not an automatic approval "
        "or rejection decision."
    )

    # ----------------------------------------------
    # Technical details
    # ----------------------------------------------

    with st.expander(
        "Technical Details"
    ):

        st.write(
            {
                "Raw model score":
                    result[
                        "raw_model_score"
                    ],

                "Calibrated probability":
                    result[
                        "calibrated_probability"
                    ],

                "Risk band":
                    result[
                        "risk_band"
                    ],

                "Manual review":
                    result[
                        "manual_review"
                    ]
            }
        )

    # ----------------------------------------------
    # SHAP explanation
    # ----------------------------------------------

    st.divider()

    st.subheader(
        "Why did the model assign this risk?"
    )

    st.caption(
        "Positive SHAP values increase the model's fraud-risk score. "
        "Negative SHAP values decrease it. These values explain model "
        "behavior and should not be interpreted as causal evidence of fraud."
    )

    top_explanation = (
        explanation
        .head(10)
        .copy()
    )

    display_explanation = (
        top_explanation[
            [
                "Feature",
                "Value",
                "SHAP_Value"
            ]
        ]
        .copy()
    )

    # Mixed numeric/categorical values -> string for display
    display_explanation["Value"] = (
        display_explanation["Value"]
        .astype(str)
    )

    display_explanation["SHAP_Value"] = (
        display_explanation["SHAP_Value"]
        .round(4)
    )

    display_explanation["Direction"] = np.where(
        display_explanation["SHAP_Value"] > 0,
        "Increases risk",
        "Decreases risk"
    )

    st.dataframe(
        display_explanation,
        use_container_width=True,
        hide_index=True
    )

    chart_data = (
        top_explanation[
            [
                "Feature",
                "SHAP_Value"
            ]
        ]
        .set_index("Feature")
        .sort_values("SHAP_Value")
    )

    st.bar_chart(
        chart_data
    )

    # ----------------------------------------------
    # Human-readable explanation
    # ----------------------------------------------

    risk_increasing = (
        explanation[
            explanation[
                "SHAP_Value"
            ] > 0
        ]
        .head(3)
    )

    risk_decreasing = (
        explanation[
            explanation[
                "SHAP_Value"
            ] < 0
        ]
        .head(3)
    )

    if not risk_increasing.empty:

        st.write(
            "**Main factors increasing predicted fraud risk:**"
        )

        for _, row in risk_increasing.iterrows():
            st.write(
                f"- {row['Feature']} "
                f"(value: {row['Value']})"
            )

    if not risk_decreasing.empty:

        st.write(
            "**Main factors decreasing predicted fraud risk:**"
        )

        for _, row in risk_decreasing.iterrows():
            st.write(
                f"- {row['Feature']} "
                f"(value: {row['Value']})"
            )


# --------------------------------------------------
# Model performance summary
# --------------------------------------------------

st.divider()

st.subheader(
    "Model Performance"
)

perf1, perf2, perf3, perf4 = (
    st.columns(4)
)

perf1.metric(
    "Final Test ROC-AUC",
    "0.8959"
)

perf2.metric(
    "Final Test PR-AUC",
    "0.2180"
)

perf3.metric(
    "Review Precision",
    "19.89%"
)

perf4.metric(
    "Fraud Recall",
    "46.85%"
)

st.caption(
    "Final out-of-time month-7 evaluation using the frozen "
    "month-6 decision threshold. The model reviewed 3.47% of "
    "applications and captured 46.85% of fraud cases."
)
