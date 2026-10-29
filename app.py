from pathlib import Path
import html
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, confusion_matrix, classification_report
)

st.set_page_config(
    page_title="ChurnLab | Customer Churn Analytics",
    page_icon="🪻",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "IT_customer_churn.csv"
MODEL_PATHS = {
    "Logistic Regression": ROOT / "models" / "logistic_regression.joblib",
    "Decision Tree": ROOT / "models" / "decision_tree.joblib",
    "KNN": ROOT / "models" / "knn.joblib",
    "Naive Bayes": ROOT / "models" / "naive_bayes.joblib",
    "SVM": ROOT / "models" / "svm.joblib",
}

# Inter is loaded from Google Fonts; the fallback keeps the app readable offline.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], [data-testid="stAppViewContainer"] {
    font-family: 'Inter', sans-serif;
}
.stApp { background: #211A2E; color: #F5F0FA; }
[data-testid="stHeader"] { background: rgba(33,26,46,.92); }
[data-testid="stSidebar"] { background: #30213F; border-right: 1px solid #574365; }
[data-testid="stSidebar"] p, [data-testid="stSidebar"] label { color: #F5F0FA; }
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: #3D2D50; border-color: #755B8C;
}
.brand { display:flex; align-items:center; gap:12px; margin:4px 0 8px; }
.brand-icon {
    width:42px; height:42px; border-radius:13px; display:flex;
    align-items:center; justify-content:center;
    background:linear-gradient(135deg,#D6BCFA,#8B9CF6);
    color:#211A2E; font-size:22px; font-weight:800;
}
.brand-title { color:#F5F0FA; font-size:20px; font-weight:800; letter-spacing:-.6px; }
.brand-subtitle { color:#C9BBD8; font-size:11px; margin-top:3px; letter-spacing:1.2px; text-transform:uppercase; }
.eyebrow { color:#B794F6; font-size:11px; font-weight:700; letter-spacing:1.5px; margin-bottom:7px; }
.page-title { color:#F5F0FA; font-size:clamp(26px,3vw,36px); font-weight:800; letter-spacing:-1.2px; line-height:1.15; }
.page-subtitle { color:#C9BBD8; font-size:14px; margin:8px 0 22px; }
.section-title { color:#F5F0FA; font-size:18px; font-weight:700; margin:8px 0 12px; }
.muted { color:#C9BBD8; font-size:13px; }
.kpi-card {
    background:linear-gradient(145deg,#302640,#2B2238);
    border:1px solid #4B3B5C; border-radius:16px; padding:17px 18px;
    min-height:126px; box-shadow:0 8px 24px rgba(0,0,0,.10);
}
.kpi-top { display:flex; align-items:center; justify-content:space-between; gap:8px; }
.kpi-label { color:#C9BBD8; font-size:12px; font-weight:600; }
.kpi-icon { display:inline-flex; align-items:center; justify-content:center;
    width:30px; height:30px; border-radius:10px; background:#49365E; font-size:15px; }
.kpi-value { color:#F5F0FA; font-size:29px; font-weight:800; letter-spacing:-1px; margin-top:12px; line-height:1.1; }
.kpi-caption { color:#AFA0C1; font-size:11px; margin-top:7px; }
.info-card { background:#302640; border:1px solid #4B3B5C; border-radius:15px; padding:16px 18px; }
.info-label { color:#C9BBD8; font-size:12px; margin-bottom:7px; }
.info-value { color:#F5F0FA; font-size:23px; font-weight:750; }
.info-note { color:#AFA0C1; font-size:11px; margin-top:4px; }
div[data-testid="stTabs"] button { font-family:'Inter',sans-serif; font-weight:600; }
div[data-testid="stTabs"] button[aria-selected="true"] { color:#D6BCFA; }
div[data-testid="stDataFrame"], div[data-testid="stTable"] {
    border:1px solid #4B3B5C; border-radius:12px; overflow:hidden;
}
.sidebar-note { color:#C9BBD8; font-size:12px; line-height:1.5; }
@media (max-width:800px) { .kpi-card {padding:14px;min-height:115px;} .kpi-value {font-size:24px;} }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_dataset(path_string):
    data = pd.read_csv(path_string)
    if "TotalCharges" in data.columns:
        data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    return data


@st.cache_resource
def load_saved_model(path_string):
    return joblib.load(path_string)


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    scores = None
    if hasattr(model, "predict_proba"):
        try:
            scores = model.predict_proba(X_test)[:, 1]
        except (ValueError, IndexError, AttributeError):
            scores = None
    elif hasattr(model, "decision_function"):
        try:
            scores = model.decision_function(X_test)
        except (ValueError, AttributeError):
            scores = None
    try:
        auc = roc_auc_score(y_test, scores) if scores is not None else None
    except ValueError:
        auc = None
    return {
        "Accuracy": accuracy_score(y_test, predictions),
        "AUC": auc,
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1-score": f1_score(y_test, predictions, zero_division=0),
        "MCC": matthews_corrcoef(y_test, predictions),
        "predictions": predictions,
    }


def render_kpi(label, value, icon, caption, accent="#B794F6"):
    shown = "N/A" if value is None or pd.isna(value) else str(value)
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-top"><div class="kpi-label">{html.escape(label)}</div>
        <div class="kpi-icon" style="color:{accent}">{icon}</div></div>
      <div class="kpi-value">{html.escape(shown)}</div>
      <div class="kpi-caption">{html.escape(caption)}</div>
    </div>""", unsafe_allow_html=True)


def fmt_percent(value):
    return "N/A" if value is None or pd.isna(value) else f"{value:.2%}"


def comparison_frame(results):
    return pd.DataFrame({
        name: {key: value for key, value in metrics.items()
               if key in ["Accuracy", "AUC", "Precision", "Recall", "F1-score", "MCC"]}
        for name, metrics in results.items()
    }).T


# Sidebar
with st.sidebar:
    st.markdown("""
    <div class="brand">
      <div class="brand-icon">✦</div>
      <div><div class="brand-title">ChurnLab</div>
      <div class="brand-subtitle">ML Analytics</div></div>
    </div>""", unsafe_allow_html=True)
    st.divider()
    st.markdown('<div class="eyebrow">MODEL CONFIGURATION</div>', unsafe_allow_html=True)
    available_models = {name: path for name, path in MODEL_PATHS.items() if path.exists()}
    selected_model = st.selectbox("Select classifier", list(available_models)) if available_models else None
    if not available_models:
        st.error("No saved model files found in the models/ folder.")
    st.divider()
    st.markdown('<div class="eyebrow">DATASET</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-note">Customer churn dataset<br>Saved models · Held-out test evaluation</div>', unsafe_allow_html=True)
    st.caption("Models are loaded from disk. This dashboard does not retrain them.")

# Header
st.markdown('<div class="eyebrow">CLASSIFICATION PERFORMANCE</div>', unsafe_allow_html=True)
st.markdown('<div class="page-title">Customer Churn Analytics</div>', unsafe_allow_html=True)
st.markdown('<div class="page-subtitle">Explore model performance, inspect predictions, and compare saved classifiers.</div>', unsafe_allow_html=True)

if not DATA_PATH.exists():
    st.error(f"Dataset file not found: {DATA_PATH}")
    st.info("Confirm data/IT_customer_churn.csv exists relative to app.py.")
    st.stop()

try:
    df = load_dataset(str(DATA_PATH))
except Exception as exc:
    st.error(f"Could not load dataset: {exc}")
    st.stop()

if "Churn" not in df.columns:
    st.error("The dataset must contain a 'Churn' target column.")
    st.stop()

y = df["Churn"].map({"No": 0, "Yes": 1})
if y.isna().any():
    st.error("Churn must contain only 'No' and 'Yes' values.")
    st.stop()
X = df.drop(columns=["Churn"])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Load and evaluate all existing models for the comparison tab. No fitting occurs here.
results, errors = {}, {}
for name, path in available_models.items():
    try:
        results[name] = evaluate_model(load_saved_model(str(path)), X_test, y_test)
    except Exception as exc:
        errors[name] = str(exc)

tabs = st.tabs(["Overview", "Model Evaluation", "Model Comparison"])
overview_tab, evaluation_tab, comparison_tab = tabs

with overview_tab:
    st.markdown('<div class="section-title">Dataset overview</div>', unsafe_allow_html=True)
    overview = [
        ("Total customers", f"{len(df):,}", "Rows in dataset", "#B794F6"),
        ("Features", str(X.shape[1]), "Input variables", "#8B9CF6"),
        ("Training rows", f"{len(X_train):,}", "80% training split", "#67D5C0"),
        ("Test rows", f"{len(X_test):,}", "20% held-out split", "#D96C9D"),
    ]
    cols = st.columns(4, gap="medium")
    for col, (label, value, caption, accent) in zip(cols, overview):
        with col:
            render_kpi(label, value, "●", caption, accent)

    left, right = st.columns([1.1, 1], gap="large")
    with left:
        st.markdown('<div class="section-title">Target distribution</div>', unsafe_allow_html=True)
        counts = df["Churn"].value_counts().reindex(["No", "Yes"], fill_value=0)
        st.bar_chart(counts, color="#B794F6")
        st.caption(f"Churn rate: {(y == 1).mean():.1%}. Consider recall, F1 and AUC as well as accuracy.")
    with right:
        st.markdown('<div class="section-title">Dataset details</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="info-card">
          <div class="info-label">Target column</div><div class="info-value">Churn</div>
          <div class="info-note">No = 0 · Yes = 1</div>
          <hr style="border-color:#4B3B5C;margin:14px 0;">
          <div class="info-label">Saved models found</div><div class="info-value">{len(available_models)}</div>
          <div class="info-note">Model files detected in models/</div>
        </div>""", unsafe_allow_html=True)
        with st.expander("Preview dataset"):
            st.dataframe(df.head(10), use_container_width=True)
    if errors:
        with st.expander("Model evaluation notices"):
            for name, error in errors.items():
                st.warning(f"{name}: {error}")

with evaluation_tab:
    st.markdown('<div class="section-title">Selected model performance</div>', unsafe_allow_html=True)
    if not selected_model or selected_model not in results:
        st.warning("The selected model could not be evaluated. Check the Overview tab for any model errors.")
    else:
        metrics = results[selected_model]
        st.markdown(f'<div class="muted">Currently viewing: <strong>{html.escape(selected_model)}</strong></div>', unsafe_allow_html=True)
        card_specs = [
            ("Accuracy", fmt_percent(metrics["Accuracy"]), "◎", "Overall correct predictions", "#B794F6"),
            ("AUC", fmt_percent(metrics["AUC"]), "⌁", "Ranking / discrimination ability", "#8B9CF6"),
            ("Precision", fmt_percent(metrics["Precision"]), "◈", "Positive predictions that were correct", "#67D5C0"),
            ("Recall", fmt_percent(metrics["Recall"]), "↗", "Actual churners correctly identified", "#D96C9D"),
            ("F1-score", fmt_percent(metrics["F1-score"]), "✳", "Balance of precision and recall", "#F2C879"),
            ("MCC", "N/A" if pd.isna(metrics["MCC"]) else f'{metrics["MCC"]:.3f}', "⊙", "Balanced correlation measure", "#A8A0B8"),
        ]
        for row_start in (0, 3):
            cols = st.columns(3, gap="medium")
            for col, spec in zip(cols, card_specs[row_start:row_start + 3]):
                with col:
                    render_kpi(*spec)
        matrix_col, report_col = st.columns([0.9, 1.1], gap="large")
        with matrix_col:
            st.markdown('<div class="section-title">Confusion matrix</div>', unsafe_allow_html=True)
            cm = confusion_matrix(y_test, metrics["predictions"], labels=[0, 1])
            cm_df = pd.DataFrame(cm, index=["Actual: No churn", "Actual: Churn"],
                                 columns=["Predicted: No churn", "Predicted: Churn"])
            st.dataframe(cm_df.style.format("{:,.0f}").background_gradient(cmap="Purples", axis=None),
                         use_container_width=True)
            st.caption("Rows are actual labels; columns are predicted labels.")
        with report_col:
            st.markdown('<div class="section-title">Classification report</div>', unsafe_allow_html=True)
            report = classification_report(
                y_test, metrics["predictions"], labels=[0, 1],
                target_names=["No churn", "Churn"], output_dict=True, zero_division=0
            )
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.format(precision=3).background_gradient(
                cmap="Purples", subset=["precision", "recall", "f1-score"], axis=None
            ), use_container_width=True)

with comparison_tab:
    st.markdown('<div class="section-title">Compare saved classifiers</div>', unsafe_allow_html=True)
    st.markdown('<div class="muted">Each available saved model is evaluated on the same held-out test split. No models are retrained.</div>', unsafe_allow_html=True)
    if not results:
        st.warning("No saved model could be evaluated. Check files in models/.")
    else:
        comparison = comparison_frame(results)
        metric = st.selectbox("Metric to compare", ["Accuracy", "AUC", "Precision", "Recall", "F1-score", "MCC"])
        chart = comparison[[metric]].dropna().sort_values(metric, ascending=False)
        st.markdown(f'<div class="section-title">{html.escape(metric)} by model</div>', unsafe_allow_html=True)
        st.bar_chart(chart, color="#B794F6", horizontal=True)
        st.markdown('<div class="section-title">All evaluation metrics</div>', unsafe_allow_html=True)
        st.dataframe(comparison.style.format("{:.3f}").background_gradient(cmap="Purples", axis=None),
                     use_container_width=True)
        if comparison["F1-score"].notna().any():
            best = comparison["F1-score"].idxmax()
            best_f1 = comparison.loc[best, "F1-score"]
            st.markdown(f"""
            <div class="info-card">
              <div class="info-label">Highest F1-score among evaluated models</div>
              <div class="info-value">{html.escape(str(best))}</div>
              <div class="info-note">F1-score: {best_f1:.3f}. Also consider recall and the costs of false predictions.</div>
            </div>""", unsafe_allow_html=True)

st.divider()
st.caption("ChurnLab · Evaluation uses a fixed stratified 80/20 split (random_state=42).")
