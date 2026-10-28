
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

st.set_page_config(
    page_title="Customer Churn Classification",
    page_icon="🤖",
    layout="wide"
)

st.title("Customer Churn Classification")
st.write(
    "Evaluate the saved Logistic Regression model "
    "using the test portion of our customer churn dataset."
)

# 1. Locate the existing dataset and saved model
data_path = Path("data/IT_customer_churn.csv")
model_path = Path("models/logistic_regression.joblib")

if not data_path.exists():
    st.error("Dataset not found: data/IT_customer_churn.csv")
    st.stop()

if not model_path.exists():
    st.error("Saved Logistic Regression model not found.")
    st.stop()

# 2. Load the existing dataset
df = pd.read_csv(data_path)

# Match the data preparation used during training
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

X = df.drop(columns=["Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

if y.isnull().any():
    st.error("Unexpected values found in the Churn column.")
    st.stop()

# 3. Recreate the same test split used during training
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 4. Load the already-trained model (no retraining)
model = joblib.load(model_path)

st.success("Saved Logistic Regression model loaded successfully.")

st.subheader("Dataset Overview")

col1, col2, col3 = st.columns(3)
col1.metric("Total customers", f"{len(df):,}")
col2.metric("Training customers", f"{len(X_train):,}")
col3.metric("Test customers", f"{len(X_test):,}")

with st.expander("Preview the existing dataset"):
    st.dataframe(df.head(10), use_container_width=True)

# 5. Evaluate on the held-out test data
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

st.subheader("Logistic Regression Evaluation")

metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "AUC": roc_auc_score(y_test, y_prob),
    "Precision": precision_score(y_test, y_pred, zero_division=0),
    "Recall": recall_score(y_test, y_pred, zero_division=0),
    "F1-score": f1_score(y_test, y_pred, zero_division=0),
    "MCC": matthews_corrcoef(y_test, y_pred)
}

cols = st.columns(3)

for i, (name, value) in enumerate(metrics.items()):
    cols[i % 3].metric(name, f"{value:.4f}")

# 6. Confusion matrix
st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

st.dataframe(
    pd.DataFrame(
        cm,
        index=["Actual No Churn", "Actual Churn"],
        columns=["Predicted No Churn", "Predicted Churn"]
    ),
    use_container_width=True
)

# 7. Classification report
st.subheader("Classification Report")

report = classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["No Churn", "Churn"],
    output_dict=True,
    zero_division=0
)

st.dataframe(
    pd.DataFrame(report).transpose(),
    use_container_width=True
)
