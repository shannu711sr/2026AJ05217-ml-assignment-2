
import os
import warnings
import joblib
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC

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

warnings.filterwarnings("ignore")


# --------------------------------------------------
# 1. PROJECT PATHS AND DATASET URLS
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent
MODEL_DIR = ROOT_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_DATA_URL = (
    "https://huggingface.co/datasets/"
    "drukeroni/airline-satisfaction-analysis/"
    "resolve/main/train.csv"
)

TEST_DATA_URL = (
    "https://huggingface.co/datasets/"
    "drukeroni/airline-satisfaction-analysis/"
    "resolve/main/test.csv"
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


# --------------------------------------------------
# 2. LOAD TRAINING AND TEST DATA
# --------------------------------------------------

print("\nLoading Airline Passenger Satisfaction datasets...")

try:
    train_df = pd.read_csv(TRAIN_DATA_URL)
    test_df = pd.read_csv(TEST_DATA_URL)

except Exception as error:
    raise RuntimeError(
        "Unable to download the Airline datasets. "
        "Check your internet connection and dataset URLs."
    ) from error

print("\nDatasets loaded successfully!")
print("Training dataset shape:", train_df.shape)
print("Testing dataset shape:", test_df.shape)


# --------------------------------------------------
# 3. CLEAN COLUMN NAMES
# --------------------------------------------------

def clean_column_names(df):
    df = df.copy()

    df.columns = (
        df.columns
        .astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )

    return df


train_df = clean_column_names(train_df)
test_df = clean_column_names(test_df)


# --------------------------------------------------
# 4. VALIDATE THE DATASETS
# --------------------------------------------------

if TARGET_COLUMN not in train_df.columns:
    raise ValueError(
        f"Training dataset does not contain '{TARGET_COLUMN}'. "
        f"Available columns: {train_df.columns.tolist()}"
    )

if TARGET_COLUMN not in test_df.columns:
    raise ValueError(
        f"Test dataset does not contain '{TARGET_COLUMN}'. "
        f"Available columns: {test_df.columns.tolist()}"
    )

# Remove columns that are not useful for prediction.
# These are dataset index/identifier columns, not passenger features.

columns_to_drop = ["id", "Unnamed: 0"]

train_df = train_df.drop(
    columns=columns_to_drop,
    errors="ignore"
)

test_df = test_df.drop(
    columns=columns_to_drop,
    errors="ignore"
)


# --------------------------------------------------
# 5. PREPARE THE TARGET COLUMN
# --------------------------------------------------

def encode_target(series):
    cleaned = series.astype(str).str.strip().str.lower()

    encoded = cleaned.map(TARGET_MAPPING)

    if encoded.isna().any():
        unknown_values = cleaned[encoded.isna()].unique().tolist()

        raise ValueError(
            "Unexpected values found in the satisfaction column: "
            f"{unknown_values}"
        )

    return encoded.astype(int)


y_train = encode_target(train_df[TARGET_COLUMN])
y_test = encode_target(test_df[TARGET_COLUMN])


# --------------------------------------------------
# 6. PREPARE THE FEATURE COLUMNS
# --------------------------------------------------

X_train = train_df.drop(columns=[TARGET_COLUMN])
X_test = test_df.drop(columns=[TARGET_COLUMN])

# Make sure both datasets have the same features.
missing_in_test = set(X_train.columns) - set(X_test.columns)
extra_in_test = set(X_test.columns) - set(X_train.columns)

if missing_in_test or extra_in_test:
    raise ValueError(
        "Training and testing feature columns do not match.\n"
        f"Missing in test dataset: {sorted(missing_in_test)}\n"
        f"Extra in test dataset: {sorted(extra_in_test)}"
    )

# Keep the test features in exactly the same order as training.
X_test = X_test[X_train.columns]

print("\nFeature preparation completed.")
print("Number of features:", X_train.shape[1])
print("Training feature shape:", X_train.shape)
print("Testing feature shape:", X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts().rename(index={
    0: CLASS_NAMES[0],
    1: CLASS_NAMES[1]
}))

print("\nTesting target distribution:")
print(y_test.value_counts().rename(index={
    0: CLASS_NAMES[0],
    1: CLASS_NAMES[1]
}))


# --------------------------------------------------
# 7. IDENTIFY NUMERIC AND CATEGORICAL FEATURES
# --------------------------------------------------

numeric_features = X_train.select_dtypes(
    include=["int64", "float64", "int32", "float32"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("\nNumeric features:", len(numeric_features))
print("Categorical features:", len(categorical_features))


# --------------------------------------------------
# 8. BUILD THE PREPROCESSING PIPELINE
# --------------------------------------------------

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_transformer,
            numeric_features
        ),
        (
            "categorical",
            categorical_transformer,
            categorical_features
        )
    ],
    sparse_threshold=0
)


# --------------------------------------------------
# 9. DEFINE THE FIVE CLASSIFIERS
# --------------------------------------------------

classifiers = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        solver="liblinear",
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        criterion="gini",
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    ),

    "K-Nearest Neighbors": KNeighborsClassifier(
        n_neighbors=5,
        weights="uniform",
        metric="minkowski",
        p=2
    ),

    "Naive Bayes": GaussianNB(),

    "Support Vector Machine": LinearSVC(
        C=1.0,
        max_iter=5000,
        random_state=42
    )
}


# --------------------------------------------------
# 10. EVALUATE A TRAINED MODEL
# --------------------------------------------------

def evaluate_model(model, X, y):
    predictions = model.predict(X)

    # Calculate AUC using probabilities when available.
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[:, 1]

    # LinearSVC does not provide predict_proba().
    # Its decision_function can be used to calculate ROC AUC.
    elif hasattr(model, "decision_function"):
        probabilities = model.decision_function(X)

    else:
        probabilities = None

    auc = (
        roc_auc_score(y, probabilities)
        if probabilities is not None
        else np.nan
    )

    metrics = {
        "Accuracy": accuracy_score(y, predictions),
        "AUC": auc,
        "Precision": precision_score(
            y, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y, predictions, zero_division=0
        ),
        "MCC": matthews_corrcoef(y, predictions)
    }

    matrix = confusion_matrix(
        y,
        predictions,
        labels=[0, 1]
    )

    report = classification_report(
        y,
        predictions,
        labels=[0, 1],
        target_names=CLASS_NAMES,
        zero_division=0
    )

    return metrics, matrix, report


# --------------------------------------------------
# 11. TRAIN AND EVALUATE ALL FIVE MODELS
# --------------------------------------------------

results = []

print("\n" + "=" * 65)
print("STARTING MODEL TRAINING")
print("=" * 65)

for model_name, classifier in classifiers.items():

    print(f"\nTraining {model_name}...")

    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ]
    )

    # Train only on the training dataset.
    model_pipeline.fit(X_train, y_train)

    print(f"{model_name} training completed.")

    # Evaluate using the separate test dataset.
    metrics, matrix, report = evaluate_model(
        model_pipeline,
        X_test,
        y_test
    )

    # Save the trained model.
    model_filename = (
        model_name.lower()
        .replace(" ", "_")
        + ".joblib"
    )

    model_path = MODEL_DIR / model_filename

    joblib.dump(model_pipeline, model_path)

    print(f"Saved model: {model_path}")

    print("\nEvaluation metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

    print("\nConfusion Matrix:")
    print(matrix)

    print("\nClassification Report:")
    print(report)

    result_row = {
        "Model": model_name,
        **metrics
    }

    results.append(result_row)


# --------------------------------------------------
# 12. SAVE MODEL COMPARISON RESULTS
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_path = MODEL_DIR / "model_comparison_results.csv"

results_df.to_csv(
    results_path,
    index=False
)

print("\n" + "=" * 65)
print("MODEL COMPARISON")
print("=" * 65)

print(
    results_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}"
    )
)

print("\nModel comparison saved to:", results_path)


# --------------------------------------------------
# 13. COMPLETION MESSAGE
# --------------------------------------------------

print("\n" + "=" * 65)
print("ALL FIVE MODELS TRAINED AND SAVED SUCCESSFULLY!")
print("=" * 65)

print("\nSaved model files:")

for file_path in sorted(MODEL_DIR.glob("*.joblib")):
    print("-", file_path.name)

print("\nYou can now run your Streamlit application.")
