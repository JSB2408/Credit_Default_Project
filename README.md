# 💳 Credit Risk Prediction System

An end-to-end machine learning project for predicting loan default risk and supporting credit-risk assessment through an interactive Streamlit application.

---
## 🚀 Live Demo

[Open the Live Application](https://creditdefaultproject-jasmeetsb.streamlit.app/)

## 📌 Project Overview

The **Credit Risk Prediction System** is a machine learning-based application designed to estimate the probability of loan default for individual applicants.

The project covers the complete data science lifecycle, including:

- Data understanding and exploration
- Data cleaning
- Missing-value analysis
- Exploratory Data Analysis (EDA)
- Feature engineering
- Numerical and categorical preprocessing
- Class-imbalance handling
- Model experimentation and comparison
- XGBoost model development
- Hyperparameter tuning
- Classification threshold optimization
- Model evaluation
- Model serialization
- Streamlit application development
- GitHub and Git LFS integration
- Cloud deployment preparation

The final application converts an applicant's information into a **default probability, risk score, risk band, and recommendation**.

---

# 🎯 Problem Statement

Loan default prediction is an important problem for financial institutions.

A lender needs to identify applicants who may have a higher probability of default while avoiding unnecessary rejection of applicants who are actually low risk.

This project aims to build a machine learning system that can:

- Predict the probability of loan default
- Handle missing data
- Process numerical and categorical variables
- Address severe class imbalance
- Compare multiple machine learning algorithms
- Optimize the classification threshold
- Generate an interpretable risk score
- Categorize applicants into risk bands
- Provide an interactive credit-risk assessment interface

The system is designed as a **credit-risk screening and decision-support tool**, rather than a fully automated loan approval or rejection system.

---

# 📊 Dataset

The project uses an applicant-level loan dataset containing financial, demographic, employment, housing, and external credit-related information.

### Dataset Description

The dataset contains applicant-level information related to financial status, demographics, employment, housing, and external credit history.

The target variable is a binary classification variable:

- `0` → No Default
- `1` → Default

The objective is to predict the probability that an applicant will default on a loan.

---

# 🔍 Project Workflow

The complete project follows the following machine learning workflow:

1. Data loading and inspection
2. Exploratory Data Analysis (EDA)
3. Data quality assessment
4. Missing-value analysis
5. Feature-type identification
6. Numerical and categorical feature preprocessing
7. Train-validation-test split
8. Handling severe class imbalance
9. Model development
10. Model comparison
11. Hyperparameter tuning
12. Probability prediction
13. Classification-threshold optimization
14. Risk-score generation
15. Risk-band classification
16. Interactive Streamlit deployment

---

## 🔬 What I Did — Detailed Project Walkthrough

The project was built as a full, phased data science workflow — from raw data to a deployed decision-support tool — rather than jumping straight to modeling.

### Phase 1–3: Business Understanding & Exploratory Data Analysis
I started by framing the problem from a lender's perspective: the real cost isn't a wrong prediction in the abstract, it's approving a loan that defaults versus rejecting an applicant who would have repaid. With that framing, I explored `application_train.csv` — checking shape, dtypes, `.describe()` summaries, and duplicate records — and confirmed the target (`TARGET`) is **heavily imbalanced**, with defaulters forming a small minority of applications. This imbalance shaped almost every decision downstream, from which metrics to trust to how the model would eventually be evaluated.

I then ran a dedicated EDA pass to answer specific business questions rather than just producing generic plots:
- **Default rate overall** — established the base rate to judge model lift against.
- **Income vs. default** — income alone turned out to be a weak standalone signal.
- **Loan amount vs. default** — similar medians across defaulters and non-defaulters, so loan size alone isn't a good risk signal; it needed to be considered relative to income.
- **Income type vs. default** — meaningful differences here (e.g. applicants on maternity leave or unemployed showed much higher default rates than working applicants or pensioners), suggesting income *source* carries risk information that raw income doesn't.
- **Age vs. default** — younger borrowers (18–25) showed the highest default rate, hinting age could act as a risk-segmentation factor.
- **Credit-to-income ratio vs. default** — broadly similar across groups, so on its own it wasn't a strong differentiator.
- **External credit scores (`EXT_SOURCE_1/2/3`) vs. default** — by far the clearest separation between defaulters and non-defaulters, especially `EXT_SOURCE_3`. This became the single strongest signal I identified in the whole dataset.

### Phase 4–5: Data Cleaning
I audited missing values feature by feature rather than applying one blanket rule. My strategy was:
- Under 20% missing → generally safe to impute.
- 20–60% missing → evaluated case by case for business relevance before deciding.
- Over 60% missing → investigated individually rather than automatically dropped.

No column was removed purely because of a missing-value percentage — I checked whether it still carried usable signal first. I also checked for and handled duplicate records at this stage.

### Phase 6: Feature Engineering
Driven directly by the EDA findings above, I engineered features that better represent repayment *capacity* and *burden* rather than raw amounts in isolation:
- **`CREDIT_GOODS_RATIO`** (credit-to-goods-price ratio) — flags applicants borrowing well beyond the value of the goods being financed.
- **`CREDIT_ANNUITY_RATIO`** — captures the relationship between the loan and the repayment installment, a proxy for repayment burden.
- **Employment stability groups** — bucketed employment duration into meaningful stability tiers instead of using the raw (and sometimes anomalous) day count.
- **Family burden feature** — combined family-related fields into a single indicator of household financial pressure.

Each engineered feature was validated afterward by comparing its distribution between defaulters and non-defaulters — several (like the credit-to-income style ratios) turned out to be weaker than expected and were kept mainly for completeness, while features tied to external credit scores and income stability showed clearer separation.

### Phase 7: Preprocessing Pipeline (Leakage-Safe)
I split the data into train/test **before** fitting any transformation, then identified numerical and categorical columns separately and built two parallel pipelines combined with a single `ColumnTransformer`:
- **Numerical:** median imputation → standard scaling.
- **Categorical:** most-frequent imputation → one-hot encoding.

The combined preprocessor was **fit only on the training set** and then used to transform both train and test data, specifically to avoid data leakage from the test set into preprocessing statistics (medians, modes, encoding categories).

### Phase 8–9: Model Development & Comparison
Because of the class imbalance, I didn't jump straight to one model — I deliberately compared several, with and without class balancing, to see how each handled the minority (default) class:

| Model | Behavior Observed |
|---|---|
| Logistic Regression (baseline) | High overall accuracy but missed most actual defaulters |
| Logistic Regression (`class_weight="balanced"`) | Recall jumped to ~67% — proved that rebalancing mattered a lot |
| Random Forest (baseline) | Very few false alarms, but recall was almost 0% for defaulters — effectively useless for this goal |
| Random Forest (balanced) | Improved recall but still underperformed the boosted models |
| XGBoost (baseline, default threshold) | Ranked applicants by risk well, but the default 0.5 threshold was too conservative to catch defaulters |
| XGBoost (balanced) | ~68% recall but flagged too many low-risk applicants at default threshold — the strongest candidate for further tuning |

All models were compared on **Precision, Recall, F1, ROC-AUC, and PR-AUC together** — not accuracy alone, since a model predicting "no default" for everyone would already score ~92% accuracy on this dataset while being useless. **XGBoost** was selected as the final model family based on this comparison.

### Phase 9.1–10: Hyperparameter Tuning, Threshold Optimization & Final Evaluation
I carved out a separate **validation set** from the training data (keeping the original test set untouched) so I could tune the model and its decision threshold without ever peeking at test data.

- **Hyperparameter tuning:** used `RandomizedSearchCV` to search XGBoost's hyperparameter space and identify the best-performing configuration on the validation set.
- **Threshold optimization:** rather than accepting the default 0.5 cutoff, I swept thresholds from 0.10 to 0.90 and tracked Precision/Recall/F1 at each one. This is important because in a highly imbalanced problem, 0.5 is rarely the right operating point — it's a modeling artifact, not a business decision.
- **Final threshold selected: 0.60** — chosen because it gave the best F1 balance among tested thresholds (Precision ≈ 21.4%, Recall ≈ 51.2%) while still catching roughly half of all actual defaulters.
- The tuned model and threshold were then evaluated **once** on the held-out test set to get an honest, final read on performance (see the metrics table above).

### Phase 11: Model Explainability
Since a credit-risk model can't just be a black box for stakeholders, I examined **global feature importance** from the final XGBoost model. This confirmed what EDA had already hinted at: external credit-score fields were the most influential predictors of default, reinforcing that the engineered ratio features added some value but the pre-existing credit-bureau-style signals carried the most weight.

### Phase 14: Credit Risk Scoring System
I converted the raw model probability into a business-friendly output:
- **Risk Score** — probability scaled to a 0–100 range.
- **Risk Bands** — LOW (<30%), MEDIUM (30–60%), HIGH (≥60%) — mirroring how a credit team would actually triage applications instead of reading a raw probability.
- Applied this scoring system across test applicants and reviewed the resulting risk distribution to sanity-check that it produced a sensible spread rather than clustering everyone into one band.

### Phase 15: Deployment
Finally, I built a **Streamlit interface** (`app.py`) so the model is usable by someone who isn't reading a notebook:
- Persisted the final model, the fitted preprocessor, and a reference slice of training data with `joblib`.
- Built a form collecting only the features that matter most for a human to reason about (income, credit amount, annuity, age, employment, external scores, family/housing status, etc.), while every other feature the model technically needs gets auto-filled from training-data medians/modes behind the scenes.
- Wired the form submission through preprocessing → prediction → risk scoring → a plain-language recommendation, so the output reads like a decision aid rather than a raw number.

### Business Recommendations (from the notebook's conclusion)
1. Use the model as a **risk-screening tool**, not an auto-approve/auto-reject system.
2. Prioritize applicants using the strongest indicators identified — external credit scores in particular.
3. Route applications scoring above the 0.60 threshold for **additional manual review**.
4. Because precision is low (22.2%), avoid automated rejection — some flagged applicants will be false positives.
5. Use the model **alongside** existing credit policy and human judgment, not as a replacement for it.

---

## 📊 Final Model Performance (Test Set)

| Metric | Score |
|---|---|
| Accuracy | 81.4% |
| Precision | 22.2% |
| Recall | 52.0% |
| F1 Score | 31.1% |
| ROC-AUC | 76.9% |
| PR-AUC | 25.8% |
| Decision Threshold | 0.60 |

**Interpretation:**

- **Accuracy: 81.4%** — Overall prediction performance is good, but accuracy is not the main metric due to class imbalance.
- **Recall: 52%** — The model identifies about half of the actual defaulters, reducing the risk of missing high-risk customers.
- **Precision: 22.2%** — About 1 in 5 customers flagged as defaulters actually default, so some customers will be unnecessarily flagged.
- **F1-score: 31.1%** — Shows a reasonable balance between identifying defaulters and avoiding false alarms, but there is still room for improvement.
- **ROC-AUC: 76.9%** — The model has a good ability to distinguish between defaulters and non-defaulters.
- **PR-AUC: 25.8%** — Indicates useful performance on the minority default class, which is more meaningful than accuracy for this imbalanced dataset.

> Precision is deliberately traded off for higher Recall. In a credit risk context, failing to flag a genuine defaulter (a false negative) is typically far costlier than an unnecessary manual review (a false positive), so the decision threshold was tuned rather than left at the default 0.5.

---

## 🖥️ Web Application

The Streamlit app (`app.py`) provides a simple form-based interface:

- Applicant enters key details — income, credit amount, annuity, age, employment, family/housing status, external credit scores, etc.
- Any additional features the model needs (but the applicant doesn't see) are auto-filled with sensible defaults derived from the training data (median for numeric, mode for categorical).
- The pipeline preprocesses the input, generates a default probability from the XGBoost model, and displays:
  - **Default Probability** and **Risk Score** (0–100)
  - **Risk Band** — LOW / MEDIUM / HIGH RISK
  - A plain-language **recommendation** for the loan reviewer

---
## 📁 Repository Structure

├── app.py                              # Streamlit web application
├── Credit_Risk___Loan_Approval.ipynb   # Full EDA, feature engineering & modeling notebook
├── final_xgb_model.pkl                 # Trained, tuned XGBoost model
├── preprocessor.pkl                    # Fitted ColumnTransformer (imputation, scaling, encoding)
├── X_train_reference.pkl               # Training data reference (used to auto-fill non-form features)
├── requirements.txt                    # Python dependencies
└── README.md


## ⚙️ Installation & Setup

1. **Clone the repository**
```bash
   git clone https://github.com/<your-username>/<your-repo-name>.git
   cd <your-repo-name>
```

2. **Create a virtual environment (recommended)**
```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
   pip install -r requirements.txt
```

4. **Run the app**
```bash
   streamlit run app.py
```

5. Open the local URL Streamlit prints (typically `http://localhost:8501`) in your browser.

---

## 🧰 Tech Stack

- **Language:** Python
- **ML / Data:** scikit-learn, XGBoost, pandas, NumPy, SciPy
- **App:** Streamlit
- **Model Persistence:** joblib

See `requirements.txt` for pinned versions.

---

## 🔮 Future Improvements

- Add SHAP-based explainability so each prediction shows *why* the model flagged an applicant as risky
- Expose batch scoring (CSV upload) alongside the single-applicant form
- Add model monitoring / drift checks for production use
- Package as a Docker container for reproducible deployment

---

## 📄 License

This project is for educational and portfolio purposes.
EOF

echo "README.md created successfully."
