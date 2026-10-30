
import pandas as pd
import joblib

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
# 1. LOAD AIRLINE PASSENGER SATISFACTION DATASET
# ============================================================

train_data_url = (
    "https://huggingface.co/datasets/drukeroni/"
    "airline-satisfaction-analysis/resolve/main/train.csv"
)

test_data_url = (
    "https://huggingface.co/datasets/drukeroni/"
    "airline-satisfaction-analysis/resolve/main/test.csv"
)

print("Loading Airline Passenger Satisfaction datasets...")

train_df = pd.read_csv(train_data_url)
test_df = pd.read_csv(test_data_url)

print("\nDatasets loaded successfully!")
print("Training dataset shape:", train_df.shape)
print("Testing dataset shape:", test_df.shape)


# ============================================================
# 2. DATA PREPARATION
# ============================================================

target_column = "satisfaction"
id_column = "id"

# Check whether target column exists
if target_column not in train_df.columns:
    raise ValueError("Target column 'satisfaction' not found in training data.")

if target_column not in test_df.columns:
    raise ValueError("Target column 'satisfaction' not found in testing data.")

# Separate target and input features
y_train = train_df[target_column].astype(str).str.strip().str.lower()
y_test = test_df[target_column].astype(str).str.strip().str.lower()

# Convert target labels into binary values
target_mapping = {
    "neutral or dissatisfied": 0,
    "satisfied": 1
}

y_train = y_train.map(target_mapping)
y_test = y_test.map(target_mapping)

if y_train.isnull().any() or y_test.isnull().any():
    raise ValueError(
        "Unexpected values found in the satisfaction target column."
    )

# Remove target and ID columns from input features
X_train = train_df.drop(columns=[target_column, id_column], errors="ignore")
X_test = test_df.drop(columns=[target_column, id_column], errors="ignore")

# Make sure training and testing features match
missing_columns = set(X_train.columns) - set(X_test.columns)
extra_columns = set(X_test.columns) - set(X_train.columns)

if missing_columns or extra_columns:
    raise ValueError(
        f"Training/testing feature mismatch.\n"
        f"Missing in test data: {missing_columns}\n"
        f"Extra in test data: {extra_columns}"
    )

# Keep the same feature order in both datasets
X_test = X_test[X_train.columns]

print("\nDataset preparation successful!")
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Number of input features:", X_train.shape[1])

print(
    "Training satisfaction rate:",
    round(y_train.mean() * 100, 2), "%"
)

print(
    "Testing satisfaction rate:",
    round(y_test.mean() * 100, 2), "%"
)


# ============================================================
# 3. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
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
# 4. CREATE PREPROCESSING PIPELINE
# ============================================================

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
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ],
    sparse_threshold=0
)


# ============================================================
# 5. CREATE MODELS FOLDER
# ============================================================

Path("models").mkdir(exist_ok=True)


# ============================================================
# 6. COMMON MODEL EVALUATION FUNCTION
# ============================================================

def evaluate_and_save_model(model, model_name, file_name):

    print("\n" + "=" * 55)
    print(f"{model_name.upper()} RESULTS")
    print("=" * 55)

    # Train the model
    model.fit(X_train, y_train)

    # Generate predictions
    y_pred = model.predict(X_test)

    # Obtain scores for AUC calculation
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = model.decision_function(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_score)
    precision = precision_score(
        y_test, y_pred, zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )
    mcc = matthews_corrcoef(y_test, y_pred)

    print("Accuracy :", round(accuracy, 4))
    print("AUC      :", round(auc, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1-score :", round(f1, 4))
    print("MCC      :", round(mcc, 4))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=[
            "Neutral or Dissatisfied",
            "Satisfied"
        ],
        zero_division=0
    ))

    # Save the trained model
    joblib.dump(model, f"models/{file_name}")

    print(f"\n{model_name} saved successfully to models/{file_name}")

    return {
        "Model": model_name,
        "Accuracy": round(accuracy, 4),
        "AUC": round(auc, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1-score": round(f1, 4),
        "MCC": round(mcc, 4)
    }


# ============================================================
# 7. LOGISTIC REGRESSION
# ============================================================

logistic_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=1000,
        solver="liblinear"
    ))
])

logistic_results = evaluate_and_save_model(
    logistic_model,
    "Logistic Regression",
    "logistic_regression.joblib"
)


# ============================================================
# 8. DECISION TREE
# ============================================================

decision_tree_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(
        criterion="gini",
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    ))
])

decision_tree_results = evaluate_and_save_model(
    decision_tree_model,
    "Decision Tree",
    "decision_tree.joblib"
)


# ============================================================
# 9. K-NEAREST NEIGHBORS
# ============================================================

knn_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", KNeighborsClassifier(
        n_neighbors=5,
        weights="uniform",
        metric="minkowski",
        p=2
    ))
])

knn_results = evaluate_and_save_model(
    knn_model,
    "K-Nearest Neighbors",
    "knn.joblib"
)


# ============================================================
# 10. GAUSSIAN NAIVE BAYES
# ============================================================

naive_bayes_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", GaussianNB())
])

naive_bayes_results = evaluate_and_save_model(
    naive_bayes_model,
    "Gaussian Naive Bayes",
    "naive_bayes.joblib"
)


# ============================================================
# 11. SUPPORT VECTOR MACHINE
# ============================================================

svm_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LinearSVC(
        C=1.0,
        max_iter=5000,
        random_state=42
    ))
])

svm_results = evaluate_and_save_model(
    svm_model,
    "Support Vector Machine (SVM)",
    "svm.joblib"
)


# ============================================================
# 12. FINAL MODEL COMPARISON TABLE
# ============================================================

results = [
    logistic_results,
    decision_tree_results,
    knn_results,
    naive_bayes_results,
    svm_results
]

results_df = pd.DataFrame(results)

print("\n\n")
print("=" * 90)
print("FINAL COMPARISON OF ALL FIVE CLASSIFICATION MODELS")
print("=" * 90)

print(results_df.to_string(index=False))

# Save evaluation results as CSV
results_df.to_csv(
    "models/model_comparison_results.csv",
    index=False
)

print("\nAll five models trained and saved successfully!")
print("Evaluation results saved to models/model_comparison_results.csv")
