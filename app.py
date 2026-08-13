import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
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
    page_title="ML Classification Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 ML Model Classification & Evaluation Dashboard")
st.markdown("BITS Pilani WILP - Machine Learning Assignment 2")

# Sidebar - Dataset Upload and Model Selection
st.sidebar.header("1. Upload Test Data")
uploaded_file = st.sidebar.file_uploader("Upload test_data.csv", type=["csv"])

st.sidebar.header("2. Choose Model")
model_choice = st.sidebar.selectbox(
    "Select ML Classifier",
    [
        "Logistic Regression",
        "Decision Tree",
        "kNN",
        "Naive Bayes",
        "Random Forest",
        "Gradient Boosting"
    ]
)

# Model Mapping
model_files = {
    "Logistic Regression": ("models/logistic_regression.pkl", True),
    "Decision Tree": ("models/decision_tree.pkl", False),
    "kNN": ("models/knn.pkl", True),
    "Naive Bayes": ("models/naive_bayes.pkl", True),
    "Random Forest": ("models/random_forest.pkl", False),
    "Gradient Boosting": ("models/gradient_boosting.pkl", False)
}

if uploaded_file is not None:
    test_data = pd.read_csv(uploaded_file)
    st.subheader("📁 Uploaded Test Dataset Preview")
    st.dataframe(test_data.head())

    if "target" not in test_data.columns:
        st.error("Error: The uploaded CSV must contain a 'target' column for ground truth evaluation.")
    else:
        X_test = test_data.drop(columns=["target"])
        y_test = test_data["target"]

        # Load Scaler and Model
        model_path, needs_scaling = model_files[model_choice]
        try:
            model = joblib.load(model_path)
            scaler = joblib.load("models/scaler.pkl")

            # Scale if required
            X_eval = scaler.transform(X_test) if needs_scaling else X_test

            # Predictions
            y_pred = model.predict(X_eval)
            y_proba = model.predict_proba(X_eval)[:, 1] if hasattr(model, "predict_proba") else y_pred

            # Metrics
            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            mcc = matthews_corrcoef(y_test, y_pred)

            st.markdown(f"---")
            st.subheader(f"📈 Performance Metrics: {model_choice}")

            # Display Metrics in Columns
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("Accuracy", f"{acc:.4f}")
            col2.metric("AUC Score", f"{auc:.4f}")
            col3.metric("Precision", f"{prec:.4f}")
            col4.metric("Recall", f"{rec:.4f}")
            col5.metric("F1 Score", f"{f1:.4f}")
            col6.metric("MCC", f"{mcc:.4f}")

            st.markdown("---")

            # Visualizations Section
            col_left, col_right = st.columns(2)

            with col_left:
                st.subheader("📌 Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                            xticklabels=["Malignant (0)", "Benign (1)"],
                            yticklabels=["Malignant (0)", "Benign (1)"])
                plt.ylabel("Actual")
                plt.xlabel("Predicted")
                st.pyplot(fig)

            with col_right:
                st.subheader("📋 Classification Report")
                report = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.style.highlight_max(axis=0))

        except Exception as e:
            st.error(f"Error loading model or generating metrics: {e}")
else:
    st.info("👈 Please upload `test_data.csv` in the sidebar to run the evaluation.")