import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix,
    classification_report, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Breast Cancer Classification Model Evaluator",
    page_icon="🧬",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

FEATURES = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean",
    "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se",
    "smoothness_se", "compactness_se", "concavity_se",
    "concave_points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
    "smoothness_worst", "compactness_worst", "concavity_worst",
    "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"
]

st.title("Breast Cancer Classification Model Evaluator")
st.write(
    "Evaluate a previously trained classification model using a labelled CSV test file. "
    "The application loads saved estimators; it does not retrain models."
)

with st.sidebar:
    st.header("Evaluation settings")
    selected_model = st.selectbox("Select classification model", list(MODEL_FILES))
    st.caption("Positive class: Malignant (M)")
    st.caption("Expected target column: diagnosis")

uploaded_file = st.file_uploader(
    "Upload test dataset (CSV)",
    type=["csv"],
    help="The CSV must contain all 30 predictor columns and a diagnosis column containing B/M labels."
)

if uploaded_file is None:
    st.info("Upload the provided test_data.csv to begin evaluation.")
    st.subheader("Expected CSV structure")
    st.write("30 numerical feature columns plus one `diagnosis` target column.")
    st.code("radius_mean, texture_mean, ..., fractal_dimension_worst, diagnosis\n"
            "14.2, 20.1, ..., 0.082, B")
    st.stop()

try:
    test_df = pd.read_csv(uploaded_file)
except Exception as exc:
    st.error(f"Could not read the uploaded CSV: {exc}")
    st.stop()

required_columns = FEATURES + ["diagnosis"]
missing_columns = [col for col in required_columns if col not in test_df.columns]
extra_columns = [col for col in test_df.columns if col not in required_columns]

if missing_columns:
    st.error("The uploaded CSV is missing required columns:")
    st.write(missing_columns)
    st.stop()

if test_df.empty:
    st.error("The uploaded CSV contains no records.")
    st.stop()

if test_df[FEATURES].isna().any().any():
    st.error("Missing values were found in predictor features. Please provide a complete test CSV.")
    st.stop()

try:
    X_test = test_df[FEATURES].apply(pd.to_numeric, errors="raise")
except Exception:
    st.error("All 30 predictor columns must contain numerical values.")
    st.stop()

y_labels = test_df["diagnosis"].astype(str).str.strip().str.upper()
if not y_labels.isin(["B", "M"]).all():
    st.error("The diagnosis column must contain only B (benign) or M (malignant).")
    st.stop()

y_true = y_labels.map({"B": 0, "M": 1}).to_numpy()

model_path = MODEL_DIR / MODEL_FILES[selected_model]
if not model_path.exists():
    st.error(f"Saved model file not found: {model_path}")
    st.stop()

try:
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        y_score = None
except Exception as exc:
    st.error(f"Model evaluation failed: {exc}")
    st.stop()

if len(np.unique(y_true)) < 2:
    st.error("AUC cannot be calculated because the uploaded test file must contain both B and M classes.")
    st.stop()

metric_values = {
    "Accuracy": accuracy_score(y_true, y_pred),
    "AUC": roc_auc_score(y_true, y_score) if y_score is not None else np.nan,
    "Precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
    "Recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
    "F1 Score": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
    "MCC": matthews_corrcoef(y_true, y_pred),
}

st.success(f"Evaluation completed using {selected_model}. No model retraining was performed.")
st.subheader("Uploaded test data")
col1, col2, col3 = st.columns(3)
col1.metric("Test records", f"{len(test_df):,}")
col2.metric("Benign cases", int((y_true == 0).sum()))
col3.metric("Malignant cases", int((y_true == 1).sum()))

st.subheader("Evaluation metrics")
metric_cols = st.columns(3)
for idx, (metric_name, metric_value) in enumerate(metric_values.items()):
    metric_cols[idx % 3].metric(metric_name, "N/A" if pd.isna(metric_value) else f"{metric_value:.4f}")

st.subheader("Confusion matrix")
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
fig, ax = plt.subplots(figsize=(5, 4))
disp = ConfusionMatrixDisplay(cm, display_labels=["Benign", "Malignant"])
disp.plot(ax=ax, colorbar=False, values_format="d")
ax.set_title(f"{selected_model} – Confusion Matrix")
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.subheader("Classification report")
report = classification_report(
    y_true, y_pred,
    labels=[0, 1],
    target_names=["Benign", "Malignant"],
    output_dict=True,
    zero_division=0
)
st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)

with st.expander("View prediction results"):
    results_view = test_df[["diagnosis"]].copy()
    results_view["Predicted diagnosis"] = pd.Series(y_pred).map({0: "B", 1: "M"}).values
    results_view["Correct prediction"] = (
        results_view["diagnosis"].values == results_view["Predicted diagnosis"].values
    )
    st.dataframe(results_view, use_container_width=True)

st.caption(
    "Educational machine-learning assignment only. This application is not a clinical "
    "diagnostic tool and must not be used to make medical decisions."
)
