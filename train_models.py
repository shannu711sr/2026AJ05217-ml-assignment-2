
import joblib
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)


# ============================================================
# 1. PROJECT PATHS AND DATASET URL
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent
MODEL_DIR = ROOT_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_DATA_URL = (
    "https://huggingface.co/datasets/drukeroni/"
    "airline-satisfaction-analysis/resolve/main/train.csv"
)

TARGET_COLUMN = "satisfaction"

TARGET_MAPPING = {
    "neutral or dissatisfied": 0,
    "satisfied": 1
}

CLASS_NAMES = [
    "Neutral or Dissatisfied",
    "Satisfied"
]


# ============================================================
# 2. LOAD THE TRAINING DATASET
# ============================================================

print("\nLoading Airline Passenger Satisfaction dataset...")

try:
    df = pd.read_csv(TRAIN_DATA_URL)

except Exception as error:
    raise RuntimeError(
        "Unable to download the training dataset. "
        "Check your internet connection and dataset URL."
    ) from error

print("\nDataset loaded successfully!")
print("Original dataset shape:", df.shape)


# ============================================================
# 3. CLEAN COLUMN NAMES
# ============================================================

df.columns = (
    df.columns
    .astype(str)
    .str.replace("\ufeff", "", regex=False)
    .str.strip()
)


# ============================================================
# 4. VALIDATE AND PREPARE THE DATASET
# ============================================================

if TARGET_COLUMN not in df.columns:
    raise ValueError(
        f"Target column '{TARGET_COLUMN}' not found in dataset."
    )

# Remove identifier columns, if present.
df = df.drop(
    columns=["id", "Unnamed: 0"],
    errors="ignore"
)

# Convert target labels into binary values.
target_values = (
    df[TARGET_COLUMN]
    .astype(str)
    .str.strip()
    .str.lower()
)

y = target_values.map(TARGET_MAPPING)

if y.isna().any():
    unknown_values = target_values[y.isna()].unique().tolist()

    raise ValueError(
        "Unexpected values found in the satisfaction column: "
        f"{unknown_values}"
    )

y = y.astype(int)

# Separate input features and target.
X = df.drop(columns=[TARGET_COLUMN])

print("\nDataset preparation completed!")
print("Total records:", len(X))
print("Number of input features:", X.shape[1])

print("\nTarget distribution:")
print(
    y.value_counts()
    .rename(index={
        0: CLASS_NAMES[0],
        1: CLASS_NAMES[1]
    })
)

print("\nTarget distribution in percentage:")
print(
    (y.value_counts(normalize=True) * 100)
    .round(2)
)


# ============================================================
# 5. SPLIT THE DATASET INTO TRAINING AND TESTING DATA
# ============================================================

# Use 80% for training and 20% for testing.
# Stratification preserves the original class distribution.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nDataset split completed!")

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

print(
    "Training satisfaction rate:",
    round(y_train.mean() * 100, 2),
    "%"
)

print(
    "Testing satisfaction rate:",
    round(y_test.mean() * 100, 2),
    "%"
)


# ============================================================
# 6. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# ============================================================

numeric_features = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("\nNumerical features:", numeric_features)
print("\nCategorical features:", categorical_features)


# ============================================================
# 7. CREATE PREPROCESSING PIPELINE
# ============================================================

def create_preprocessor():

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        ],
        sparse_threshold=0
    )

    return preprocessor


# ============================================================
# 8. COMMON MODEL EVALUATION FUNCTION
# ============================================================

def evaluate_and_save_model(
    classifier,
    model_name,
    file_name
):

    print("\n" + "=" * 65)
    print(f"{model_name.upper()} RESULTS")
    print("=" * 65)

    # Create an independent preprocessing pipeline
    # for each classifier.

    model_pipeline = Pipeline([
        ("preprocessor", create_preprocessor()),
        ("classifier", classifier)
    ])

    # Train using only the training portion.
    model_pipeline.fit(X_train, y_train)

    print(f"\n{model_name} training completed.")

    # Generate predictions on the held-out test portion.
    y_pred = model_pipeline.predict(X_test)

    # Obtain scores for AUC calculation.
    if hasattr(model_pipeline, "predict_proba"):
        y_score = model_pipeline.predict_proba(X_test)[:, 1]

    elif hasattr(model_pipeline, "decision_function"):
        y_score = model_pipeline.decision_function(X_test)

    else:
        y_score = None

    # Calculate evaluation metrics.
    accuracy = accuracy_score(y_test, y_pred)

    auc = (
        roc_auc_score(y_test, y_score)
        if y_score is not None
        else np.nan
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    mcc = matthews_corrcoef(y_test, y_pred)

    metrics = {
        "Accuracy": accuracy,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "MCC": mcc
    }

    print("\nEvaluation Metrics:")

    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

    # Confusion matrix.
    matrix = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    print("\nConfusion Matrix:")
    print(matrix)

    # Classification report.
    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=CLASS_NAMES,
        zero_division=0
    )

    print("\nClassification Report:")
    print(report)

    # Save the complete trained pipeline.
    model_path = MODEL_DIR / file_name

    joblib.dump(
        model_pipeline,
        model_path
    )

    print(f"\nSaved model: {model_path}")

    # Return evaluation results.
    return {
        "Model": model_name,
        **metrics
    }


# ============================================================
# 9. LOGISTIC REGRESSION
# ============================================================

logistic_results = evaluate_and_save_model(
    LogisticRegression(
        max_iter=1000,
        solver="liblinear",
        random_state=42
    ),
    "Logistic Regression",
    "logistic_regression.joblib"
)


# ============================================================
# 10. DECISION TREE
# ============================================================

decision_tree_results = evaluate_and_save_model(
    DecisionTreeClassifier(
        criterion="gini",
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    ),
    "Decision Tree",
    "decision_tree.joblib"
)


# ============================================================
# 11. K-NEAREST NEIGHBORS
# ============================================================

knn_results = evaluate_and_save_model(
    KNeighborsClassifier(
        n_neighbors=5,
        weights="uniform",
        metric="minkowski",
        p=2
    ),
    "K-Nearest Neighbors",
    "k-nearest_neighbors.joblib"
)


# ============================================================
# 12. GAUSSIAN NAIVE BAYES
# ============================================================

naive_bayes_results = evaluate_and_save_model(
    GaussianNB(),
    "Naive Bayes",
    "naive_bayes.joblib"
)


# ============================================================
# 13. SUPPORT VECTOR MACHINE
# ============================================================

svm_results = evaluate_and_save_model(
    LinearSVC(
        C=1.0,
        max_iter=5000,
        random_state=42
    ),
    "Support Vector Machine",
    "support_vector_machine.joblib"
)


# ============================================================
# 14. FINAL MODEL COMPARISON
# ============================================================

results = [
    logistic_results,
    decision_tree_results,
    knn_results,
    naive_bayes_results,
    svm_results
]

results_df = pd.DataFrame(results)

print("\n" + "=" * 90)
print("FINAL COMPARISON OF ALL FIVE CLASSIFICATION MODELS")
print("=" * 90)

print(
    results_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}"
    )
)


# ============================================================
# 15. SAVE MODEL COMPARISON RESULTS
# ============================================================

results_path = MODEL_DIR / "model_comparison_results.csv"

results_df.to_csv(
    results_path,
    index=False
)

print("\nModel comparison saved to:", results_path)


# ============================================================
# 16. COMPLETION MESSAGE
# ============================================================

print("\n" + "=" * 65)
print("ALL FIVE MODELS TRAINED AND SAVED SUCCESSFULLY!")
print("=" * 65)

print("\nTraining dataset split:")
print("80% - Model training")
print("20% - Model evaluation")

print("\nSaved model files:")

for file_path in sorted(MODEL_DIR.glob("*.joblib")):
    print("-", file_path.name)

print("\nYou can now run your Streamlit application.")
