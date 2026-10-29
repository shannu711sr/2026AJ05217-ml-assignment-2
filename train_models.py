import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


'''
-------------------------------- LOGISTIC REGRESSION ---------------------------------
'''


# Loading dataset
csv_files = list(Path("data").glob("*.csv"))

if not csv_files:
    raise FileNotFoundError("No CSV file found inside the data folder.")

df = pd.read_csv(csv_files[0])

# Preparing the target and features
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

X = df.drop(columns=["Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

if y.isnull().any():
    raise ValueError("Unexpected values found in the Churn column.")

# Identify numerical / categorical columns
numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("Numerical features:", numeric_features)
print("Categorical features:", categorical_features)

# Preprocess numerical features
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Preprocess categorical features
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Combine both preprocessing pipelines
preprocessor = ColumnTransformer(
    [
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ],
    sparse_threshold=0
)


# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nDataset preparation successful!")
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Training churn rate:", round(y_train.mean() * 100, 2), "%")
print("Testing churn rate:", round(y_test.mean() * 100, 2), "%")
print("\nPreprocessing pipeline created successfully.")

from sklearn.linear_model import LogisticRegression
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

# Create the complete Logistic Regression pipeline
logistic_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=1000,
        solver="liblinear"
    ))
])

# Train the model
logistic_model.fit(X_train, y_train)
y_pred = logistic_model.predict(X_test)
y_prob = logistic_model.predict_proba(X_test)[:, 1]

# Calculate the required evaluation metrics
print("\n=========== LOGISTIC REGRESSION ==========")
print("Accuracy :", round(accuracy_score(y_test, y_pred), 4))
print("AUC      :", round(roc_auc_score(y_test, y_prob), 4))
print("Precision:", round(precision_score(y_test, y_pred, zero_division=0), 4))
print("Recall   :", round(recall_score(y_test, y_pred, zero_division=0), 4))
print("F1-score :", round(f1_score(y_test, y_pred, zero_division=0), 4))
print("MCC      :", round(matthews_corrcoef(y_test, y_pred), 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["No Churn", "Churn"],
    zero_division=0
))

# Save the trained Logistic Regression model
from pathlib import Path

Path("models").mkdir(exist_ok=True)

joblib.dump(
    logistic_model,
    "models/logistic_regression.joblib"
)

print("\nLogistic Regression model saved successfully.")


'''
-------------------------------- DECISION TREE ---------------------------------
'''

from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
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
import joblib
from pathlib import Path

# Create the Decision Tree pipeline
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

# Train the model using the existing training data
decision_tree_model.fit(X_train, y_train)

# Predict on the existing test data
y_pred_dt = decision_tree_model.predict(X_test)
y_prob_dt = decision_tree_model.predict_proba(X_test)[:, 1]

# Calculate evaluation metrics
print("\n" + "=" * 45)
print("DECISION TREE CLASSIFIER RESULTS")
print("=" * 45)

print("Accuracy:", round(accuracy_score(y_test, y_pred_dt), 4))
print("AUC:", round(roc_auc_score(y_test, y_prob_dt), 4))
print("Precision:", round(precision_score(y_test, y_pred_dt, zero_division=0), 4))
print("Recall:", round(recall_score(y_test, y_pred_dt, zero_division=0), 4))
print("F1-score:", round(f1_score(y_test, y_pred_dt, zero_division=0), 4))
print("MCC:", round(matthews_corrcoef(y_test, y_pred_dt), 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_dt))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred_dt,
    target_names=["No Churn", "Churn"],
    zero_division=0
))

# Save the trained model
Path("models").mkdir(exist_ok=True)
joblib.dump(decision_tree_model, "models/decision_tree.joblib")

print("\nDecision Tree model saved to models/decision_tree.joblib")

'''
-------------------------------- KNN - K Nearst Neighbors ---------------------------------
'''


from sklearn.neighbors import KNeighborsClassifier

# Create the KNN pipeline
knn_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", KNeighborsClassifier(
        n_neighbors=5,
        weights="uniform",
        metric="minkowski",
        p=2
    ))
])

# Train the model using the existing training data
knn_model.fit(X_train, y_train)

# Predict on the existing test data
y_pred_knn = knn_model.predict(X_test)
y_prob_knn = knn_model.predict_proba(X_test)[:, 1]

# Calculate evaluation metrics
print("\n" + "=" * 45)
print("K-NEAREST NEIGHBORS (KNN) RESULTS")
print("=" * 45)

print("Accuracy:", round(accuracy_score(y_test, y_pred_knn), 4))
print("AUC:", round(roc_auc_score(y_test, y_prob_knn), 4))
print("Precision:", round(
    precision_score(y_test, y_pred_knn, zero_division=0), 4
))
print("Recall:", round(
    recall_score(y_test, y_pred_knn, zero_division=0), 4
))
print("F1-score:", round(
    f1_score(y_test, y_pred_knn, zero_division=0), 4
))
print("MCC:", round(matthews_corrcoef(y_test, y_pred_knn), 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_knn))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred_knn,
    labels=[0, 1],
    target_names=["No Churn", "Churn"],
    zero_division=0
))

# Save the trained model
Path("models").mkdir(exist_ok=True)
joblib.dump(knn_model, "models/knn.joblib")

print("\nKNN model saved to models/knn.joblib")


'''
-------------------------------- Gaussian Naive Bayes ---------------------------------
'''

from sklearn.naive_bayes import GaussianNB

# Create the Naive Bayes pipeline
naive_bayes_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", GaussianNB())
])

# Train the model using the existing training data
naive_bayes_model.fit(X_train, y_train)

# Predict on the existing test data
y_pred_nb = naive_bayes_model.predict(X_test)
y_prob_nb = naive_bayes_model.predict_proba(X_test)[:, 1]

# Calculate evaluation metrics
print("\n" + "=" * 45)
print("GAUSSIAN NAIVE BAYES RESULTS")
print("=" * 45)

print("Accuracy:", round(accuracy_score(y_test, y_pred_nb), 4))
print("AUC:", round(roc_auc_score(y_test, y_prob_nb), 4))
print("Precision:", round(
    precision_score(y_test, y_pred_nb, zero_division=0), 4
))
print("Recall:", round(
    recall_score(y_test, y_pred_nb, zero_division=0), 4
))
print("F1-score:", round(
    f1_score(y_test, y_pred_nb, zero_division=0), 4
))
print("MCC:", round(matthews_corrcoef(y_test, y_pred_nb), 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_nb))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred_nb,
    labels=[0, 1],
    target_names=["No Churn", "Churn"],
    zero_division=0
))

# Save the trained model
Path("models").mkdir(exist_ok=True)
joblib.dump(naive_bayes_model, "models/naive_bayes.joblib")

print("\nNaive Bayes model saved to models/naive_bayes.joblib")


'''
-------------------------------- Gaussian Naive Bayes ---------------------------------
'''
