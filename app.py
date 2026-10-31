
import os
import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Binary Classifier Studio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# PROJECT PATHS AND MODEL CONFIGURATION
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

RESULTS_FILE = os.path.join(
    MODEL_DIR,
    "model_comparison_results.csv"
)

MODEL_OPTIONS = [
    "Logistic Regression",
    "Decision Tree",
    "K-Nearest Neighbors (KNN)",
    "Naive Bayes",
    "Support Vector Machine (SVM)",
]

# Support both possible KNN filenames to accommodate
# the naming convention used by different training scripts.
MODEL_FILE_CANDIDATES = {
    "Logistic Regression": [
        "logistic_regression.joblib",
    ],
    "Decision Tree": [
        "decision_tree.joblib",
    ],
    "K-Nearest Neighbors (KNN)": [
        "k_nearest_neighbors.joblib",
        "k-nearest_neighbors.joblib",
    ],
    "Naive Bayes": [
        "naive_bayes.joblib",
    ],
    "Support Vector Machine (SVM)": [
        "support_vector_machine.joblib",
    ],
}

TRAINING_RESULT_NAMES = {
    "Logistic Regression": "Logistic Regression",
    "Decision Tree": "Decision Tree",
    "K-Nearest Neighbors (KNN)": "K-Nearest Neighbors",
    "Naive Bayes": "Naive Bayes",
    "Support Vector Machine (SVM)": "Support Vector Machine",
}

TARGET_COLUMN = "satisfaction"
ID_COLUMNS = ["id", "Unnamed: 0"]


# ---------------------------------------------------------
# CUSTOM CSS - PLUM AND LAVENDER THEME
# ---------------------------------------------------------

st.markdown(
    """
    <style>
    /* Main application background */
    .stApp {
        background-color: #180F20;
        color: #E8DDF5;
    }

    /* Main page content */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #24152F;
        width: 19rem !important;
    }

    section[data-testid="stSidebar"] > div {
        width: 19rem !important;
    }

    section[data-testid="stSidebar"] * {
        color: #E8DDF5;
    }

    /* Headings */
    h1, h2, h3 {
        color: #D9B8FF !important;
    }

    /* General text */
    p, label, .stMarkdown {
        color: #E8DDF5;
    }

    /* Buttons */
    div.stButton > button {
        background-color: #8155A6;
        color: white;
        border: 1px solid #A77BCB;
        border-radius: 8px;
        padding: 0.45rem 1rem;
        font-weight: 600;
    }

    div.stButton > button:hover {
        background-color: #A77BCB;
        color: white;
        border-color: #D9B8FF;
    }

    /* Select boxes and uploaders */
    div[data-baseweb="select"] > div {
        background-color: #30213B;
        border-color: #8155A6;
    }

    div[data-testid="stFileUploader"] {
        background-color: #24152F;
        border: 1px dashed #8155A6;
        border-radius: 10px;
        padding: 0.5rem;
    }

    /* Metrics */
    div[data-testid="stMetric"] {
        background-color: #30213B;
        border: 1px solid #684783;
        border-radius: 10px;
        padding: 12px;
    }

    div[data-testid="stMetricLabel"] {
        color: #C9A9E8;
    }

    div[data-testid="stMetricValue"] {
        color: #F0E5FF;
    }

    /* Tables */
    div[data-testid="stDataFrame"] {
        border: 1px solid #684783;
        border-radius: 8px;
    }

    /* Info messages */
    div[data-testid="stAlert"] {
        background-color: #30213B;
        border: 1px solid #8155A6;
        color: #E8DDF5;
    }

    /* Compact evaluation controls when results are visible */
    .compact-note {
        color: #C9A9E8;
        font-size: 0.85rem;
        margin-top: -0.4rem;
        margin-bottom: 0.5rem;
    }

    /* Reduce vertical spacing between elements */
    div[data-testid="stVerticalBlock"] {
        gap: 0.7rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------

if "display_mode" not in st.session_state:
    st.session_state.display_mode = None

if "test_results" not in st.session_state:
    st.session_state.test_results = None

if "training_model_name" not in st.session_state:
    st.session_state.training_model_name = "Logistic Regression"

if "show_training_parameters" not in st.session_state:
    st.session_state.show_training_parameters = False

if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None


# ---------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------

@st.cache_resource
def load_saved_model(model_path, modified_time):
    return joblib.load(model_path)


def get_model_path(model_name):
    candidates = MODEL_FILE_CANDIDATES[model_name]

    for filename in candidates:
        path = os.path.join(MODEL_DIR, filename)

        if os.path.exists(path):
            return path

    return None


def load_model(model_name):
    model_path = get_model_path(model_name)

    if model_path is None:
        expected_files = ", ".join(
            MODEL_FILE_CANDIDATES[model_name]
        )

        raise FileNotFoundError(
            f"Could not find the saved model for {model_name}. "
            f"Expected one of these files in the models folder: "
            f"{expected_files}"
        )

    modified_time = os.path.getmtime(model_path)

    return load_saved_model(model_path, modified_time)


# ---------------------------------------------------------
# DATA CLEANING AND PREPARATION
# ---------------------------------------------------------

def clean_column_names(data):
    data.columns = (
        data.columns
        .astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )

    return data


def prepare_uploaded_data(uploaded_file, model):
    uploaded_file.seek(0)

    data = pd.read_csv(
        uploaded_file,
        encoding="utf-8-sig"
    )

    data = clean_column_names(data)

    # Find the target column without depending on capitalization.
    target_matches = [
        col for col in data.columns
        if col.lower() == TARGET_COLUMN.lower()
    ]

    if not target_matches:
        raise ValueError(
            "The uploaded CSV must contain a "
            "'satisfaction' target column."
        )

    target_column = target_matches[0]

    # Convert target labels to numeric values.
    target_values = (
        data[target_column]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    target_mapping = {
        "neutral or dissatisfied": 0,
        "satisfied": 1,
        "0": 0,
        "1": 1,
    }

    y = target_values.map(target_mapping)

    if y.isna().any():
        invalid_values = target_values[y.isna()].unique().tolist()

        raise ValueError(
            "Unrecognized values found in the satisfaction column: "
            f"{invalid_values}"
        )

    y = y.astype(int)

    # Remove target and ID columns from the input features.
    X = data.drop(columns=[target_column])

    columns_to_drop = [
        col for col in X.columns
        if col.lower() in [item.lower() for item in ID_COLUMNS]
    ]

    X = X.drop(columns=columns_to_drop)

    # Retrieve the feature names expected by the saved pipeline.
    try:
        expected_features = list(
            model.named_steps["preprocessor"].feature_names_in_
        )
    except (AttributeError, KeyError):
        raise ValueError(
            "The saved model does not contain the expected "
            "preprocessing information."
        )

    # Check for missing features.
    missing_features = [
        col for col in expected_features
        if col not in X.columns
    ]

    if missing_features:
        raise ValueError(
            "The uploaded dataset is missing the following "
            f"required features: {missing_features}"
        )

    # Ignore additional columns and arrange columns in the
    # same order used during model training.
    X = X[expected_features]

    return X, y


# ---------------------------------------------------------
# MODEL EVALUATION
# ---------------------------------------------------------

def evaluate_model(model, X, y):
    predictions = model.predict(X)

    # Calculate AUC using probabilities or decision scores.
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, scores)

    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        auc = roc_auc_score(y, scores)

    else:
        auc = np.nan

    results = {
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
        "MCC": matthews_corrcoef(y, predictions),
        "Confusion Matrix": confusion_matrix(y, predictions),
        "Classification Report": classification_report(
            y,
            predictions,
            target_names=[
                "Neutral or Dissatisfied",
                "Satisfied"
            ],
            output_dict=True,
            zero_division=0,
        ),
    }

    return results


# ---------------------------------------------------------
# DISPLAY EVALUATION METRICS
# ---------------------------------------------------------

def display_metric_cards(metrics):
    metric_names = [
        "Accuracy",
        "AUC",
        "Precision",
        "Recall",
        "F1 Score",
        "MCC",
    ]

    columns = st.columns(3)

    for index, metric_name in enumerate(metric_names):
        value = metrics.get(metric_name, np.nan)

        if pd.isna(value):
            display_value = "N/A"
        else:
            display_value = f"{value:.4f}"

        with columns[index % 3]:
            st.metric(
                label=metric_name,
                value=display_value
            )


def display_confusion_matrix(matrix):
    st.subheader("Confusion Matrix")

    matrix_df = pd.DataFrame(
        matrix,
        index=[
            "Actual: Neutral/Dissatisfied",
            "Actual: Satisfied"
        ],
        columns=[
            "Predicted: Neutral/Dissatisfied",
            "Predicted: Satisfied"
        ],
    )

    st.dataframe(
        matrix_df,
        use_container_width=True
    )


def display_classification_report(report):
    st.subheader("Classification Report")

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(
        report_df,
        use_container_width=True
    )


def display_test_results(results):
    st.header("Test Dataset Evaluation Results")

    display_metric_cards(results)

    st.divider()

    display_confusion_matrix(
        results["Confusion Matrix"]
    )

    st.divider()

    display_classification_report(
        results["Classification Report"]
    )


# ---------------------------------------------------------
# TRAINED MODEL METRICS
# ---------------------------------------------------------

def display_training_metrics(model_name):
    if not os.path.exists(RESULTS_FILE):
        st.error(
            "Training evaluation results file was not found. "
            "Please check that model_comparison_results.csv "
            "exists inside the models folder."
        )
        return

    try:
        results_df = pd.read_csv(RESULTS_FILE)
        results_df = clean_column_names(results_df)

        if "Model" not in results_df.columns:
            st.error(
                "The training results CSV does not contain "
                "a 'Model' column."
            )
            return

        result_name = TRAINING_RESULT_NAMES[model_name]

        matching_rows = results_df[
            results_df["Model"].astype(str).str.strip()
            == result_name
        ]

        if matching_rows.empty:
            st.error(
                f"No saved training metrics were found for "
                f"{result_name}."
            )
            return

        row = matching_rows.iloc[0]

        st.header("Trained Model Evaluation Results")
        st.subheader(model_name)

        metric_mapping = {
            "Accuracy": "Accuracy",
            "AUC": "AUC",
            "Precision": "Precision",
            "Recall": "Recall",
            "F1 Score": "F1 Score",
            "MCC": "MCC",
        }

        metrics = {}

        for display_name, csv_column in metric_mapping.items():
            if csv_column in results_df.columns:
                metrics[display_name] = pd.to_numeric(
                    row[csv_column],
                    errors="coerce"
                )

        display_metric_cards(metrics)

        st.caption(
            "These metrics were calculated during model training "
            "and evaluation using the saved test dataset."
        )

    except Exception as error:
        st.error(
            f"Unable to load the trained model metrics: {error}"
        )


# ---------------------------------------------------------
# TRAINING PARAMETERS
# ---------------------------------------------------------

def display_model_parameters(model_name):
    try:
        model = load_model(model_name)

        st.markdown("#### Training Parameters")

        if hasattr(model, "named_steps"):
            classifier = model.named_steps.get("classifier")

            if classifier is not None:
                parameters = classifier.get_params()

                parameter_df = pd.DataFrame(
                    list(parameters.items()),
                    columns=["Parameter", "Value"]
                )

                st.dataframe(
                    parameter_df,
                    use_container_width=True,
                    hide_index=True
                )

            st.caption(
                "The saved model also includes the preprocessing "
                "pipeline used during training."
            )

        else:
            parameters = model.get_params()

            parameter_df = pd.DataFrame(
                list(parameters.items()),
                columns=["Parameter", "Value"]
            )

            st.dataframe(
                parameter_df,
                use_container_width=True,
                hide_index=True
            )

    except Exception as error:
        st.error(
            f"Unable to load training parameters: {error}"
        )


# ---------------------------------------------------------
# SIDEBAR - TRAINED MODEL METRICS SELECTION
# ---------------------------------------------------------

with st.sidebar:
    st.title("Trained Model Metrics")

    st.caption(
        "Select a trained classifier to view its saved "
        "evaluation metrics."
    )

    selected_training_model = st.selectbox(
        "Select Classifier",
        MODEL_OPTIONS,
        key="training_model_selector"
    )

    show_training_metrics_button = st.button(
        "Show Trained Model Evaluation Metrics",
        use_container_width=True
    )

    if show_training_metrics_button:
        st.session_state.display_mode = "training"
        st.session_state.training_model_name = (
            selected_training_model
        )
        st.session_state.test_results = None
        st.session_state.show_training_parameters = False

    st.divider()

    if st.session_state.display_mode == "training":
        if st.button(
            "View Training Parameters",
            use_container_width=True
        ):
            st.session_state.show_training_parameters = (
                not st.session_state.show_training_parameters
            )

        if st.session_state.show_training_parameters:
            display_model_parameters(
                st.session_state.training_model_name
            )


# ---------------------------------------------------------
# MAIN PAGE - TITLE AND DESCRIPTION
# ---------------------------------------------------------

st.title("Binary Classifier Studio")

st.markdown(
    """
    Evaluate the performance of trained machine learning
    classifiers using your test dataset.
    """
)

st.markdown(
    "No model training or retraining takes place in this application. "
    "The application uses previously trained models."
)


# ---------------------------------------------------------
# MAIN PAGE - TEST DATASET EVALUATION CONTROLS
# ---------------------------------------------------------

results_are_visible = (
    st.session_state.display_mode in ["training", "test"]
)

if results_are_visible:
    st.subheader("Test Dataset Evaluation")
    st.markdown(
        '<p class="compact-note">'
        'You can upload another dataset or evaluate a different model. '
        'Scroll down to view the evaluation results.'
        '</p>',
        unsafe_allow_html=True
    )
else:
    st.header("Test Dataset Evaluation")

    st.caption(
        "Upload your test dataset, select a classifier, "
        "and evaluate its performance."
    )


# Use a compact layout once results are visible.
if results_are_visible:
    upload_column, model_column = st.columns([1.5, 1])

    with upload_column:
        uploaded_file = st.file_uploader(
            "Upload Test Dataset (CSV)",
            type=["csv"],
            key="test_dataset_uploader",
            help="Upload a CSV containing the satisfaction target column."
        )

    with model_column:
        selected_test_model = st.selectbox(
            "Select Classifier",
            MODEL_OPTIONS,
            key="test_model_selector"
        )

else:
    uploaded_file = st.file_uploader(
        "Upload Test Dataset (CSV)",
        type=["csv"],
        key="test_dataset_uploader",
        help="Upload a CSV containing the satisfaction target column."
    )

    selected_test_model = st.selectbox(
        "Select Classifier",
        MODEL_OPTIONS,
        key="test_model_selector"
    )


evaluate_button = st.button(
    "Evaluate Model",
    use_container_width=True
)


# ---------------------------------------------------------
# RUN TEST DATASET EVALUATION
# ---------------------------------------------------------

if evaluate_button:
    if uploaded_file is None:
        st.warning(
            "Please upload a test dataset before clicking "
            "Evaluate Model."
        )

    else:
        try:
            with st.spinner("Evaluating the selected model..."):
                model = load_model(selected_test_model)

                X, y = prepare_uploaded_data(
                    uploaded_file,
                    model
                )

                results = evaluate_model(
                    model,
                    X,
                    y
                )

            st.session_state.test_results = {
                "model_name": selected_test_model,
                "metrics": results,
                "file_name": uploaded_file.name,
            }

            # Test dataset results replace previously displayed
            # trained model results.
            st.session_state.display_mode = "test"
            st.session_state.show_training_parameters = False

            st.success("Model evaluation completed successfully.")

        except Exception as error:
            st.error(
                f"An error occurred during model evaluation: {error}"
            )


# ---------------------------------------------------------
# MAIN PAGE - DISPLAY ALL EVALUATION METRICS
# ---------------------------------------------------------

if st.session_state.display_mode == "training":
    st.divider()

    display_training_metrics(
        st.session_state.training_model_name
    )

elif st.session_state.display_mode == "test":
    st.divider()

    test_results = st.session_state.test_results

    if test_results is not None:
        st.caption(
            f"Dataset: {test_results['file_name']} | "
            f"Classifier: {test_results['model_name']}"
        )

        display_test_results(
            test_results["metrics"]
        )
