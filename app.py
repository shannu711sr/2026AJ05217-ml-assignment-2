
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# PAGE CONFIGURATION
st.set_page_config(
    page_title="Customer Churn Classification",
    page_icon="🤖",
    layout="wide"
)

st.title("Customer Churn Classification")
st.write(
    "Select a trained classification model to evaluate "
    "customer churn using the existing dataset."
)

# FILE PATHS
data_path = Path("data/IT_customer_churn.csv")

model_paths = {
    "Logistic Regression": Path("models/logistic_regression.joblib"),
    "Decision Tree": Path("models/decision_tree.joblib"),
    "KNN": Path("models/knn.joblib"),
    "Naive Bayes": Path("models/naive_bayes.joblib"),
    "SVM": Path("models/svm.joblib")
}

# CHECK REQUIRED FILES
if not data_path.exists():
    st.error("Dataset not found: data/IT_customer_churn.csv")
    st.stop()

df = pd.read_csv(data_path)

required_columns = {"Churn", "TotalCharges"}

if not required_columns.issubset(df.columns):
    st.error("The dataset is missing required columns.")
    st.stop()


# PREPARE DATA
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

X = df.drop(columns=["Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

if y.isnull().any():
    st.error("Unexpected or missing values found in the Churn column.")
    st.stop()

# Use the same split as train_models.py
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# DATASET OVERVIEW
st.subheader("Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Customers", f"{len(df):,}")
col2.metric("Training Customers", f"{len(X_train):,}")
col3.metric("Test Customers", f"{len(X_test):,}")

with st.expander("Preview the existing dataset"):
    st.dataframe(df.head(10), use_container_width=True)

# MODEL SELECTION
st.subheader("Select Classification Model")

selected_model_name = st.selectbox(
    "Choose a model",
    options=list(model_paths.keys())
)

selected_model_path = model_paths[selected_model_name]

if not selected_model_path.exists():
    st.error(
        f"Saved model not found: {selected_model_path}"
    )
    st.stop()

# Load the selected saved model; do not retrain it
try:
    model = joblib.load(selected_model_path)
except Exception as error:
    st.error(f"Could not load the selected model: {error}")
    st.stop()

st.success(f"{selected_model_name} model loaded successfully.")

# MODEL PREDICTIONS
try:
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = None

except Exception as error:
    st.error(f"Could not generate predictions: {error}")
    st.stop()


# EVALUATION METRICS
st.subheader(f"{selected_model_name} Evaluation")

metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(
        y_test, y_pred, zero_division=0
    ),
    "Recall": recall_score(
        y_test, y_pred, zero_division=0
    ),
    "F1-score": f1_score(
        y_test, y_pred, zero_division=0
    ),
    "MCC": matthews_corrcoef(y_test, y_pred)
}

if y_prob is not None:
    metrics["AUC"] = roc_auc_score(y_test, y_prob)

metric_columns = st.columns(3)

for index, (metric_name, metric_value) in enumerate(metrics.items()):
    metric_columns[index % 3].metric(
        metric_name,
        f"{metric_value:.4f}"
    )

# CONFUSION MATRIX
st.subheader("Confusion Matrix")

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
)

cm_df = pd.DataFrame(
    cm,
    index=["Actual No Churn", "Actual Churn"],
    columns=["Predicted No Churn", "Predicted Churn"]
)

st.dataframe(cm_df, use_container_width=True)

# CLASSIFICATION REPORT
st.subheader("Classification Report")

report = classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["No Churn", "Churn"],
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df, use_container_width=True)
