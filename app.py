
import streamlit as st
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

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
    page_title="Binary Classifier Studio",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"

TARGET_COLUMN = "satisfaction"
ID_COLUMN = "id"

TARGET_MAPPING = {
    "neutral or dissatisfied": 0,
    "satisfied": 1
}

CLASS_NAMES = [
    "Neutral or Dissatisfied",
    "Satisfied"
]

MODEL_OPTIONS = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "K-Nearest Neighbors (KNN)": "knn",
    "Naive Bayes": "naive_bayes",
    "Support Vector Machine (SVM)": "svm"
}


st.markdown(
    """
    <style>

    /* Main application background */
    .stApp {
        background-color: #17121F;
        color: #F2EDF7;
    }

    /* Main content area */
    section[data-testid="stMain"] {
        background-color: #17121F;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #21192C;
    }

    /* Main title */
    .main-title {
        font-size: 2.5rem;
        font-weight: 750;
        color: #E6D5FF;
        margin-bottom: 0.2rem;
    }

    /* Subtitle */
    .sub-title {
        color: #B9A9C9;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    /* Section headings */
    .section-title {
        font-size: 1.35rem;
        font-weight: 650;
        color: #D9C2F5;
        margin-top: 1rem;
        margin-bottom: 0.8rem;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background-color: #251C31;
        border: 1px solid #49375C;
        padding: 18px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
    }

    div[data-testid="stMetricLabel"] {
        color: #BDAACF;
    }

    div[data-testid="stMetricValue"] {
        color: #E6D5FF;
        font-weight: 700;
    }

    /* Buttons */
    div.stButton > button,
    div.stFormSubmitButton > button {
        background-color: #8056AD;
        color: white;
        border-radius: 9px;
        border: 1px solid #A77BCF;
        padding: 0.65rem 1.2rem;
        font-weight: 600;
    }

    div.stButton > button:hover,
    div.stFormSubmitButton > button:hover {
        background-color: #9B70C6;
        color: white;
        border: 1px solid #C6A5E8;
    }

    /* File uploader */
    div[data-testid="stFileUploader"] {
        background-color: #251C31;
        border: 1px dashed #80649E;
        border-radius: 12px;
        padding: 15px;
    }

    div[data-testid="stFileUploader"] section {
        background-color: transparent;
    }

    /* Select box */
    div[data-baseweb="select"] > div {
        background-color: #251C31;
        border-color: #634B7D;
        color: #F2EDF7;
    }

    /* Dataframes */
    div[data-testid="stDataFrame"] {
        border: 1px solid #49375C;
        border-radius: 8px;
    }

    /* Expanders */
    div[data-testid="stExpander"] {
        background-color: #21192C;
        border: 1px solid #49375C;
        border-radius: 10px;
    }

    /* Info and warning containers */
    div[data-testid="stAlert"] {
        background-color: #2B2038;
        color: #E6D5FF;
        border: 1px solid #634B7D;
        border-radius: 10px;
    }

    /* Horizontal separators */
    hr {
        border-color: #49375C;
    }

    /* General text */
    p, label, li {
        color: #E8E0EF;
    }

    /* Captions */
    div[data-testid="stCaptionContainer"] {
        color: #B9A9C9;
    }

    /* Form container */
    div[data-testid="stForm"] {
        background-color: #21192C;
        border: 1px solid #49375C;
        border-radius: 12px;
        padding: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


@st.cache_resource
def load_saved_model(path_string, modified_time):
    return joblib.load(path_string)


def validate_model_features(model, X_test):

    if not hasattr(model, "named_steps"):
        raise ValueError(
            "The saved model does not have the expected preprocessing pipeline."
        )

    if "preprocessor" not in model.named_steps:
        raise ValueError(
            "The saved model is missing its preprocessing step."
        )

    preprocessor = model.named_steps["preprocessor"]

    if not hasattr(preprocessor, "feature_names_in_"):
        raise ValueError(
            "Unable to determine the features used during model training."
        )

    expected_features = list(preprocessor.feature_names_in_)
    uploaded_features = list(X_test.columns)

    missing_features = [
        feature for feature in expected_features
        if feature not in uploaded_features
    ]

    if missing_features:
        raise ValueError(
            "The uploaded dataset is missing the following required features: "
            + ", ".join(missing_features)
        )

    # Keep only the features used during training, in training order.
    return X_test[expected_features]

def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    # Get scores for AUC
    if hasattr(model, "predict_proba"):

        classes = list(model.classes_)

        if 1 not in classes:
            raise ValueError(
                "The trained model does not contain the positive class."
            )

        positive_index = classes.index(1)

        scores = model.predict_proba(X_test)[:, positive_index]

    elif hasattr(model, "decision_function"):

        scores = model.decision_function(X_test)

    else:
        scores = None

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)

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

    auc = (
        roc_auc_score(y_test, scores)
        if scores is not None
        else None
    )

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "auc": auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
        "confusion_matrix": cm,
        "classification_report": report
    }

def display_results(results, selected_model_name):

    st.divider()

    st.markdown(
        '<div class="section-title">Model Evaluation Results</div>',
        unsafe_allow_html=True
    )

    st.caption(f"Selected classifier: {selected_model_name}")

    # -----------------------------------------------------
    # PERFORMANCE METRICS
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">Performance Metrics</div>',
        unsafe_allow_html=True
    )

    metric_columns = st.columns(3)

    metrics = [
        ("Accuracy", results["accuracy"]),
        ("AUC", results["auc"]),
        ("Precision", results["precision"]),
        ("Recall", results["recall"]),
        ("F1 Score", results["f1"]),
        ("MCC", results["mcc"])
    ]

    for index, (metric_name, metric_value) in enumerate(metrics):

        with metric_columns[index % 3]:

            if metric_value is not None:
                st.metric(
                    label=metric_name,
                    value=f"{metric_value:.4f}"
                )
            else:
                st.metric(
                    label=metric_name,
                    value="N/A"
                )

    st.divider()

    st.markdown(
        '<div class="section-title">Confusion Matrix</div>',
        unsafe_allow_html=True
    )

    cm_df = pd.DataFrame(
        results["confusion_matrix"],
        index=[
            "Actual: Neutral / Dissatisfied",
            "Actual: Satisfied"
        ],
        columns=[
            "Predicted: Neutral / Dissatisfied",
            "Predicted: Satisfied"
        ]
    )

    st.dataframe(
        cm_df,
        use_container_width=True
    )

    st.divider()

    st.markdown(
        '<div class="section-title">Classification Report</div>',
        unsafe_allow_html=True
    )

    report_df = pd.DataFrame(
        results["classification_report"]
    ).transpose()

    st.dataframe(
        report_df.round(4),
        use_container_width=True
    )


st.markdown(
    '<div class="main-title">Binary Classifier Studio</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'Evaluate the performance of trained machine learning classifiers '
    'using your test dataset.'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    The five classifiers have already been trained using the training dataset.
    Upload your test dataset, select a classifier, and evaluate its performance.

    **No model training or retraining takes place in this application.**
    """
)

st.divider()

st.markdown(
    '<div class="section-title">Model Evaluation</div>',
    unsafe_allow_html=True
)

# Both input options appear together
upload_col, model_col = st.columns([1.3, 1])

with upload_col:

    st.markdown("**1. Upload Test Dataset**")

    uploaded_file = st.file_uploader(
        "Choose your test CSV file",
        type=["csv"],
        help="Upload the test.csv file containing the satisfaction column."
    )

with model_col:

    st.markdown("**2. Select Classifier**")

    selected_model_name = st.selectbox(
        "Choose a classifier",
        options=list(MODEL_OPTIONS.keys())
    )


test_df = None

if uploaded_file is not None:

    try:

        test_df = pd.read_csv(uploaded_file, encoding="utf-8-sig")

        # Clean column names
        test_df.columns = (
            test_df.columns
            .astype(str)
            .str.replace("\ufeff", "", regex=False)
            .str.strip()
        )

        # Match the target column irrespective of capitalization
        target_matches = [
            col for col in test_df.columns
            if col.lower() == TARGET_COLUMN.lower()
        ]

        if target_matches:
            test_df = test_df.rename(
                columns={target_matches[0]: TARGET_COLUMN}
            )

        if test_df.empty:
            st.error("The uploaded CSV file is empty.")
            test_df = None

        else:

            st.success(
                f"Dataset uploaded successfully: "
                f"{test_df.shape[0]:,} records and "
                f"{test_df.shape[1]} columns."
            )

            with st.expander("Preview uploaded dataset"):

                st.dataframe(
                    test_df.head(10),
                    use_container_width=True
                )

    except Exception as error:

        st.error(f"Unable to read the uploaded CSV file: {error}")
        test_df = None


st.divider()


evaluate_button = st.button(
    "Evaluate Model",
    type="primary",
    use_container_width=True,
    disabled=(test_df is None)
)


if evaluate_button:

    if TARGET_COLUMN not in test_df.columns:

        st.error(
            f"The uploaded dataset must contain the "
            f"'{TARGET_COLUMN}' target column."
        )
        st.stop()

    # Convert target labels into binary values
    y_test_raw = (
        test_df[TARGET_COLUMN]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    y_test = y_test_raw.map(TARGET_MAPPING)

    if y_test.isna().any():

        invalid_values = y_test_raw[y_test.isna()].unique()

        st.error(
            "Unexpected values found in the target column: "
            + ", ".join(map(str, invalid_values))
        )
        st.stop()

    y_test = y_test.astype(int)

    if len(y_test.unique()) < 2:

        st.error(
            "The test dataset must contain both target classes "
            "to calculate AUC and meaningful classification metrics."
        )
        st.stop()

    # Separate features and target
    X_test = test_df.drop(
        columns=[TARGET_COLUMN, ID_COLUMN],
        errors="ignore"
    )

    # Find the selected saved model
    model_key = MODEL_OPTIONS[selected_model_name]

    model_path = MODEL_DIR / f"{model_key}.joblib"

    if not model_path.exists():

        st.error(
            f"Trained model file not found: {model_path.name}. "
            "Please run train_models.py first."
        )
        st.stop()

    try:

        with st.spinner(
            f"Evaluating {selected_model_name}..."
        ):

            # Load the previously trained model
            model = load_saved_model(
                str(model_path),
                model_path.stat().st_mtime
            )

            # Validate uploaded features
            X_test = validate_model_features(
                model,
                X_test
            )

            # Evaluate selected model
            results = evaluate_model(
                model,
                X_test,
                y_test
            )

        st.success("Model evaluation completed successfully!")

        # Display metrics only after clicking Evaluate Model
        display_results(
            results,
            selected_model_name
        )

    except Exception as error:

        st.error(
            f"Unable to evaluate the selected model: {error}"
        )

st.divider()

st.caption(
    "Binary Classifier Studio | Airline Passenger Satisfaction "
    "| Model Evaluation Application"
)
