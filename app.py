import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix
)

# -----------------------------
# PAGE TITLE
# -----------------------------
st.title("Breast Cancer Classification App")

# -----------------------------
# DOWNLOAD SAMPLE TEST CSV
# -----------------------------
st.subheader("Download Sample Test CSV")

try:
    sample_test = pd.read_csv("test.csv")
    st.download_button(
        label="Download Sample test.csv",
        data=sample_test.to_csv(index=False),
        file_name="test.csv",
        mime="text/csv"
    )
except:
    st.warning("Sample test.csv not found in repo")

# -----------------------------
# LOAD MODELS + SCALER
# -----------------------------
@st.cache_resource
def load_models():

    models = {
        "Logistic Regression": joblib.load("model/logistic_regression.pkl"),
        "Decision Tree": joblib.load("model/decision_tree.pkl"),
        "KNN": joblib.load("model/knn.pkl"),
        "Naive Bayes": joblib.load("model/naive_bayes.pkl"),
        "Random Forest": joblib.load("model/random_forest.pkl"),
        "XGBoost": joblib.load("model/xgboost.pkl")
    }

    scaler = joblib.load("model/scaler.pkl")

    return models, scaler

models, scaler = load_models()

# -----------------------------
# CLASS LABEL NAMES
# -----------------------------
class_names = ["Benign", "Malignant"]

# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload Test CSV File",
    type=["csv"]
)

# -----------------------------
# MODEL SELECT
# -----------------------------
model_name = st.selectbox(
    "Select Model",
    list(models.keys())
)

# -----------------------------
# PREDICTION SECTION
# -----------------------------
if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    if "Diagnosis" not in data.columns:
        st.error("CSV must contain Diagnosis column")
    else:

        X = data.drop("Diagnosis", axis=1)
        y_true = data["Diagnosis"]

        # Scale if needed
        if model_name in ["Logistic Regression", "KNN", "XGBoost"]:
            X = scaler.transform(X)

        model = models[model_name]

        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]

        # -----------------------------
        # METRICS
        # -----------------------------
        acc = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_prob)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)

        st.subheader("Evaluation Metrics")

        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{acc:.4f}")
        col2.metric("AUC", f"{auc:.4f}")
        col3.metric("Precision", f"{prec:.4f}")

        col4, col5, col6 = st.columns(3)
        col4.metric("Recall", f"{rec:.4f}")
        col5.metric("F1 Score", f"{f1:.4f}")
        col6.metric("MCC", f"{mcc:.4f}")

        # -----------------------------
        # CONFUSION MATRIX
        # -----------------------------
        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y_true, y_pred)

        cm_df = pd.DataFrame(
            cm,
            index=[f"Actual {c}" for c in class_names],
            columns=[f"Predicted {c}" for c in class_names]
        )

        st.dataframe(cm_df)
