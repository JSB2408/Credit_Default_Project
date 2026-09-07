import streamlit as st
import pandas as pd
import joblib


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Credit Risk Prediction System",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

/* Main title */
.main-title {
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 18px;
    color: #888888;
    margin-bottom: 25px;
}

/* Section headers */
.section-title {
    font-size: 25px;
    font-weight: 600;
    margin-top: 15px;
    margin-bottom: 15px;
}

/* Result cards */
.result-card {
    padding: 22px;
    border-radius: 12px;
    border: 1px solid rgba(128,128,128,0.25);
    text-align: center;
    min-height: 130px;
}

.result-title {
    font-size: 15px;
    color: #888888;
    margin-bottom: 8px;
}

.result-value {
    font-size: 28px;
    font-weight: 700;
}

/* Information box */
.info-box {
    padding: 18px;
    border-radius: 10px;
    border: 1px solid rgba(128,128,128,0.25);
    margin-bottom: 20px;
}

/* Footer */
.footer {
    text-align: center;
    color: #888888;
    font-size: 13px;
    margin-top: 40px;
}

</style>
""", unsafe_allow_html=True)


from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


# =========================================================
# LOAD MODEL AND PREPROCESSOR
# =========================================================

@st.cache_resource
def load_resources():

    model_path = BASE_DIR / "final_xgb_model.pkl"
    preprocessor_path = BASE_DIR / "preprocessor.pkl"

    # Check files exist
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    if not preprocessor_path.exists():
        raise FileNotFoundError(
            f"Preprocessor file not found: {preprocessor_path}"
        )

    # Load model
    try:

        model = joblib.load(model_path)

    except Exception as e:

        raise RuntimeError(
            f"FAILED TO LOAD final_xgb_model.pkl: {e}"
        ) from e

    # Load preprocessor
    try:

        preprocessor = joblib.load(preprocessor_path)

    except Exception as e:

        raise RuntimeError(
            f"FAILED TO LOAD preprocessor.pkl: {e}"
        ) from e

    return model, preprocessor


try:

    model, preprocessor = load_resources()

except Exception as e:

    st.error("❌ Model loading failed.")

    st.exception(e)

    st.stop()


# =========================================================
# BUILD REFERENCE INFORMATION FROM PREPROCESSOR
# =========================================================

# The original X_train_reference.pkl was a very large
# ~216 MB file. We do not need it for deployment.
#
# The fitted preprocessor already contains:
# - feature names
# - numerical imputation values
# - categorical imputation values
# - categorical levels
#
# We reconstruct the information required by the UI
# directly from the preprocessor.


X_train_columns = list(
    preprocessor.feature_names_in_
)


# ---------------------------------------------------------
# Extract numerical and categorical transformers
# ---------------------------------------------------------

numeric_pipeline = None
categorical_pipeline = None

for name, transformer, columns in preprocessor.transformers_:

    if name == "num":

        numeric_pipeline = transformer

    elif name == "cat":

        categorical_pipeline = transformer


# ---------------------------------------------------------
# Numerical defaults
# ---------------------------------------------------------

numeric_defaults = {}

if numeric_pipeline is not None:

    numeric_imputer = numeric_pipeline.named_steps.get(
        "imputer"
    )

    if numeric_imputer is not None:

        for feature, value in zip(
            numeric_imputer.feature_names_in_,
            numeric_imputer.statistics_
        ):

            numeric_defaults[feature] = value


# ---------------------------------------------------------
# Categorical defaults
# ---------------------------------------------------------

categorical_defaults = {}
categorical_options = {}

if categorical_pipeline is not None:

    categorical_imputer = categorical_pipeline.named_steps.get(
        "imputer"
    )

    categorical_encoder = categorical_pipeline.named_steps.get(
        "encoder"
    )

    # Most frequent categorical values
    if categorical_imputer is not None:

        for feature, value in zip(
            categorical_imputer.feature_names_in_,
            categorical_imputer.statistics_
        ):

            categorical_defaults[feature] = value

    # Categories learned during training
    # Categories learned during training
# Categories learned during training
if categorical_encoder is not None:

    # Use feature names if they exist in the saved encoder
    if hasattr(categorical_encoder, "feature_names_in_"):

        for feature, categories in zip(
            categorical_encoder.feature_names_in_,
            categorical_encoder.categories_
        ):
            categorical_options[feature] = list(categories)

    else:
        # Fallback for encoders saved without feature names
        if "categorical_features" in locals():

            for feature, categories in zip(
                categorical_features,
                categorical_encoder.categories_
            ):
                categorical_options[feature] = list(categories)
        


# =========================================================
# REPLACEMENT FOR X_train
# =========================================================
class TrainingReference:

    def __init__(self, columns):

        self.columns = columns


X_train = TrainingReference(X_train_columns)
# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="main-title">💳 Credit Risk Prediction System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered loan default prediction and applicant risk assessment'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def get_default_value(feature):

    """
    Return the training-time default value for a feature.

    Numerical features use the value learned by the
    SimpleImputer during training.

    Categorical features use the most frequent value
    learned during training.
    """

    if feature in numeric_defaults:

        value = numeric_defaults[feature]

        if pd.isna(value):
            return 0.0

        return float(value)

    if feature in categorical_defaults:

        return categorical_defaults[feature]

    return 0


def get_options(feature):

    """
    Return categorical options learned during training.
    """

    if feature in categorical_options:

        return categorical_options[feature]

    return []


def risk_band(probability):

    if probability < 0.30:

        return "LOW RISK"

    elif probability < 0.60:

        return "MEDIUM RISK"

    else:

        return "HIGH RISK"


def loan_recommendation(probability):

    if probability >= 0.60:

        return "Additional review required before loan approval."

    elif probability >= 0.30:

        return "Proceed with standard assessment and additional verification."

    else:

        return "Applicant shows relatively low predicted default risk."


# =========================================================
# IMPORTANT FEATURES
# =========================================================

# These are the features that the applicant will actually see.
# Any other features required by the model will be filled
# automatically using training-data values.

important_numerical = [

    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "CNT_CHILDREN",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "CREDIT_GOODS_RATIO",
    "CREDIT_ANNUITY_RATIO"

]

important_categorical = [

    "NAME_CONTRACT_TYPE",
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE"

]


# Only keep features that actually exist in X_train.

important_numerical = [
    f for f in important_numerical
    if f in X_train.columns
]

important_categorical = [
    f for f in important_categorical
    if f in X_train.columns
]


# =========================================================
# FORM
# =========================================================

st.markdown(
    '<div class="section-title">👤 Applicant Information</div>',
    unsafe_allow_html=True
)

st.write(
    "Enter the applicant's details below. "
    "The system will automatically handle additional model features."
)


with st.form("credit_risk_form"):

    # =====================================================
    # PERSONAL INFORMATION
    # =====================================================

    with st.expander(
        "👤 Personal Information",
        expanded=True
    ):

        col1, col2, col3 = st.columns(3)

        # Gender
        if "CODE_GENDER" in important_categorical:

            options = get_options("CODE_GENDER")

            with col1:

                gender = st.selectbox(
                    "Gender",
                    options,
                    key="gender"
                )

        # Children
        if "CNT_CHILDREN" in important_numerical:

            with col2:

                children = st.number_input(
                    "Number of Children",
                    min_value=0,
                    max_value=20,
                    value=int(
                        get_default_value("CNT_CHILDREN")
                    ),
                    step=1
                )

        # Age
        if "DAYS_BIRTH" in important_numerical:

            default_age_days = get_default_value("DAYS_BIRTH")

            default_age = abs(default_age_days) / 365

            with col3:

                age = st.number_input(
                    "Age (years)",
                    min_value=18,
                    max_value=100,
                    value=int(default_age),
                    step=1
                )


    # =====================================================
    # FINANCIAL INFORMATION
    # =====================================================

    with st.expander(
        "💰 Financial Information",
        expanded=True
    ):

        col1, col2, col3 = st.columns(3)

        if "AMT_INCOME_TOTAL" in important_numerical:

            with col1:

                income = st.number_input(
                    "Annual Income",
                    min_value=0.0,
                    value=get_default_value(
                        "AMT_INCOME_TOTAL"
                    ),
                    step=1000.0
                )

        if "AMT_CREDIT" in important_numerical:

            with col2:

                credit = st.number_input(
                    "Credit Amount",
                    min_value=0.0,
                    value=get_default_value(
                        "AMT_CREDIT"
                    ),
                    step=1000.0
                )

        if "AMT_ANNUITY" in important_numerical:

            with col3:

                annuity = st.number_input(
                    "Loan Annuity",
                    min_value=0.0,
                    value=get_default_value(
                        "AMT_ANNUITY"
                    ),
                    step=100.0
                )

        col1, col2, col3 = st.columns(3)

        if "NAME_INCOME_TYPE" in important_categorical:

            options = get_options("NAME_INCOME_TYPE")

            with col1:

                income_type = st.selectbox(
                    "Income Type",
                    options
                )

        if "NAME_EDUCATION_TYPE" in important_categorical:

            options = get_options("NAME_EDUCATION_TYPE")

            with col2:

                education = st.selectbox(
                    "Education",
                    options
                )

        if "NAME_FAMILY_STATUS" in important_categorical:

            options = get_options("NAME_FAMILY_STATUS")

            with col3:

                family_status = st.selectbox(
                    "Family Status",
                    options
                )


    # =====================================================
    # CREDIT AND LOAN INFORMATION
    # =====================================================

    with st.expander(
        "🏦 Credit & Loan Information",
        expanded=True
    ):

        col1, col2, col3 = st.columns(3)

        if "NAME_CONTRACT_TYPE" in important_categorical:

            options = get_options("NAME_CONTRACT_TYPE")

            with col1:

                contract_type = st.selectbox(
                    "Contract Type",
                    options
                )

        if "EXT_SOURCE_1" in important_numerical:

            with col2:

                ext1 = st.number_input(
                    "External Credit Score 1",
                    min_value=0.0,
                    max_value=1.0,
                    value=get_default_value(
                        "EXT_SOURCE_1"
                    ),
                    step=0.01
                )

        if "EXT_SOURCE_2" in important_numerical:

            with col3:

                ext2 = st.number_input(
                    "External Credit Score 2",
                    min_value=0.0,
                    max_value=1.0,
                    value=get_default_value(
                        "EXT_SOURCE_2"
                    ),
                    step=0.01
                )

        col1, col2 = st.columns(2)

        if "EXT_SOURCE_3" in important_numerical:

            with col1:

                ext3 = st.number_input(
                    "External Credit Score 3",
                    min_value=0.0,
                    max_value=1.0,
                    value=get_default_value(
                        "EXT_SOURCE_3"
                    ),
                    step=0.01
                )

        if "CREDIT_GOODS_RATIO" in important_numerical:

            with col2:

                credit_ratio = st.number_input(
                    "Credit / Goods Ratio",
                    min_value=0.0,
                    value=get_default_value(
                        "CREDIT_GOODS_RATIO"
                    ),
                    step=0.01
                )


    # =====================================================
    # PROPERTY AND EMPLOYMENT
    # =====================================================

    with st.expander(
        "🏠 Property & Employment",
        expanded=False
    ):

        col1, col2, col3 = st.columns(3)

        if "FLAG_OWN_CAR" in important_categorical:

            options = get_options("FLAG_OWN_CAR")

            with col1:

                own_car = st.selectbox(
                    "Owns a Car",
                    options
                )

        if "FLAG_OWN_REALTY" in important_categorical:

            options = get_options("FLAG_OWN_REALTY")

            with col2:

                own_realty = st.selectbox(
                    "Owns Property",
                    options
                )

        if "NAME_HOUSING_TYPE" in important_categorical:

            options = get_options("NAME_HOUSING_TYPE")

            with col3:

                housing = st.selectbox(
                    "Housing Type",
                    options
                )

        if "DAYS_EMPLOYED" in important_numerical:

            st.number_input(
                "Employment Duration (days)",
                value=get_default_value(
                    "DAYS_EMPLOYED"
                ),
                step=1.0
            )


    # =====================================================
    # SUBMIT BUTTON
    # =====================================================

    st.divider()

    submitted = st.form_submit_button(
        "🔍  Assess Credit Risk",
        use_container_width=True
    )


# =========================================================
# CREATE APPLICANT DATA
# =========================================================

if submitted:

    # -----------------------------------------------------
    # Start with training-data defaults
    # -----------------------------------------------------

    input_data = {}

    for feature in X_train.columns:

        input_data[feature] = get_default_value(feature)


    # -----------------------------------------------------
    # Replace defaults with applicant inputs
    # -----------------------------------------------------

    # Personal

    if "CODE_GENDER" in X_train.columns:
        input_data["CODE_GENDER"] = gender

    if "CNT_CHILDREN" in X_train.columns:
        input_data["CNT_CHILDREN"] = children

    if "DAYS_BIRTH" in X_train.columns:
        input_data["DAYS_BIRTH"] = -(age * 365)


    # Financial

    if "AMT_INCOME_TOTAL" in X_train.columns:
        input_data["AMT_INCOME_TOTAL"] = income

    if "AMT_CREDIT" in X_train.columns:
        input_data["AMT_CREDIT"] = credit

    if "AMT_ANNUITY" in X_train.columns:
        input_data["AMT_ANNUITY"] = annuity

    if "NAME_INCOME_TYPE" in X_train.columns:
        input_data["NAME_INCOME_TYPE"] = income_type

    if "NAME_EDUCATION_TYPE" in X_train.columns:
        input_data["NAME_EDUCATION_TYPE"] = education

    if "NAME_FAMILY_STATUS" in X_train.columns:
        input_data["NAME_FAMILY_STATUS"] = family_status


    # Credit

    if "NAME_CONTRACT_TYPE" in X_train.columns:
        input_data["NAME_CONTRACT_TYPE"] = contract_type

    if "EXT_SOURCE_1" in X_train.columns:
        input_data["EXT_SOURCE_1"] = ext1

    if "EXT_SOURCE_2" in X_train.columns:
        input_data["EXT_SOURCE_2"] = ext2

    if "EXT_SOURCE_3" in X_train.columns:
        input_data["EXT_SOURCE_3"] = ext3

    if "CREDIT_GOODS_RATIO" in X_train.columns:
        input_data["CREDIT_GOODS_RATIO"] = credit_ratio


    # Property

    if "FLAG_OWN_CAR" in X_train.columns:
        input_data["FLAG_OWN_CAR"] = own_car

    if "FLAG_OWN_REALTY" in X_train.columns:
        input_data["FLAG_OWN_REALTY"] = own_realty

    if "NAME_HOUSING_TYPE" in X_train.columns:
        input_data["NAME_HOUSING_TYPE"] = housing


    # -----------------------------------------------------
    # Employment
    # -----------------------------------------------------

    if "DAYS_EMPLOYED" in X_train.columns:

        # Use the existing default unless the feature
        # is directly collected from the applicant.
        pass


    # =====================================================
    # DATAFRAME
    # =====================================================

    applicant_df = pd.DataFrame(
        [input_data],
        columns=X_train.columns
    )


    # =====================================================
    # PREPROCESSING
    # =====================================================

    try:

        applicant_processed = preprocessor.transform(
            applicant_df
        )

    except Exception as e:

        st.error(
            "There was an error while preprocessing "
            "the applicant information."
        )

        st.exception(e)

        st.stop()


    # =====================================================
    # PREDICTION
    # =====================================================

    try:

        probability = model.predict_proba(
            applicant_processed
        )[0][1]

    except Exception as e:

        st.error(
            "There was an error while generating "
            "the prediction."
        )

        st.exception(e)

        st.stop()


    
    # =====================================================
    # RISK CALCULATION
    # =====================================================

    risk_score = round(float(probability) * 100, 2)

    category = risk_band(probability)

    recommendation = loan_recommendation(
    probability
)


    # =====================================================
    # RESULTS
    # =====================================================

    st.divider()

    st.markdown(
        '<div class="section-title">'
        '📊 Credit Risk Assessment'
        '</div>',
        unsafe_allow_html=True
    )


    # -----------------------------------------------------
    # RESULT CARDS
    # -----------------------------------------------------

    col1, col2, col3 = st.columns(3)


    with col1:

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-title">
                    DEFAULT PROBABILITY
                </div>
                <div class="result-value">
                    {risk_score:.2f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    with col2:

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-title">
                    RISK SCORE
                </div>
                <div class="result-value">
                    {risk_score:.2f}/100
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    with col3:

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-title">
                    RISK BAND
                </div>
                <div class="result-value">
                    {category}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    # =====================================================
    # RISK MESSAGE
    # =====================================================

    st.write("")


    if category == "HIGH RISK":

        st.error(
            f"⚠️ **HIGH RISK** — {recommendation}"
        )

    elif category == "MEDIUM RISK":

        st.warning(
            f"⚠️ **MEDIUM RISK** — {recommendation}"
        )

    else:

        st.success(
            f"✅ **LOW RISK** — {recommendation}"
        )


    # =====================================================
    # PROBABILITY BAR
    # =====================================================

    st.subheader("Default Risk Level")

    st.progress(
    min(max(float(probability), 0.0), 1.0)
    )

    st.caption(
        "The probability represents the model's estimated "
        "likelihood of loan default."
    )


    # =====================================================
    # INTERPRETATION
    # =====================================================

    with st.expander("ℹ️ How to interpret this result"):

        st.write(
            """
            **Low Risk (< 30%)**  
            The model estimates a relatively low probability
            of default.

            **Medium Risk (30%–60%)**  
            The applicant requires normal credit assessment
            with additional verification where appropriate.

            **High Risk (≥ 60%)**  
            The model estimates a relatively high probability
            of default and recommends additional review.
            """
        )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        Credit Risk Prediction System | XGBoost Machine Learning Model
    </div>
    """,
    unsafe_allow_html=True
)
