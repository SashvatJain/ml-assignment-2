import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="ML Classification Dashboard",
    page_icon="logo.png",
    layout="wide"
)

MODEL_FILES = {
    "Logistic Regression": ("models/logistic_regression.pkl", True),
    "Decision Tree": ("models/decision_tree.pkl", False),
    "kNN": ("models/knn.pkl", True),
    "Naive Bayes": ("models/naive_bayes.pkl", True),
    "Random Forest": ("models/random_forest.pkl", False),
    "Gradient Boosting": ("models/gradient_boosting.pkl", False)
}

MODEL_DESCRIPTIONS = {
    "Logistic Regression": "Fast and strong baseline for linearly separable patterns after scaling.",
    "Decision Tree": "Simple to interpret and useful for rule-based split behavior.",
    "kNN": "Instance-based model that benefits from standardized numeric features.",
    "Naive Bayes": "Lightweight probabilistic baseline with low training overhead.",
    "Random Forest": "Robust ensemble model that handles nonlinear patterns well.",
    "Gradient Boosting": "Boosted tree ensemble focused on strong predictive accuracy."
}

st.markdown(
    """
    <style>
    /* Layout */
    [data-testid="stHeader"] {
        background: none !important;
        border: none !important;
        height: 0 !important;
        min-height: 0 !important;
        overflow: visible !important;
    }
    [data-testid="stToolbar"] {
        position: fixed !important;
        top: 0 !important;
        right: 0 !important;
        padding: 0.7rem 1.2rem 0 0;
        height: auto !important;
        align-items: flex-start !important;
    }
    [data-testid="stAppViewContainer"] {
        padding-top: 5.5rem;
    }
    [data-testid="stMainBlockContainer"] {
        padding: 0 1.5rem 6rem;
        max-width: none !important;
    }

    /* Expander cards */
    [data-testid="stExpander"] {
        border-radius: 12px;
        border: 1px solid rgba(128, 128, 128, 0.15);
    }

    /* Banner */
    .top-banner {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        z-index: 998;
        padding: 0.7rem 8rem 0.75rem 2.8rem;
        border-bottom: 1px solid rgba(128, 128, 128, 0.15);
        backdrop-filter: blur(10px);
    }
    .top-banner .top-title {
        margin: 0;
        font-size: 1.25rem;
        font-weight: 700;
        line-height: 1.25;
        white-space: nowrap;
    }
    .top-banner p {
        margin: 0.2rem 0 0;
        opacity: 0.7;
        font-size: 0.85rem;
        line-height: 1.4;
    }

    /* Toolbar z-index */
    [data-testid="stToolbar"] [data-testid="stToolbarActions"],
    [data-testid="stAppDeployButton"],
    [data-testid="stMainMenu"] {
        z-index: 1001;
    }

    /* Metrics subtle bg */
    [data-testid="stMetric"] {
        border-radius: 8px;
        padding: 0.6rem 0.8rem;
    }

    /* Disabled buttons */
    button:disabled {
        opacity: 0.4 !important;
    }
    </style>
    <div class="top-banner">
        <div class="top-title">ML Model Classification and Evaluation Dashboard</div>
        <p>
            Upload a labeled evaluation dataset, choose a trained classifier, and run a structured model review
            with metrics, class-level results, and confusion matrix analysis.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Setup Accordion ---
with st.expander("Setup", expanded=True):
    st.caption("Upload a CSV file and select a model, then click Evaluate.")

    with st.form("evaluation_form"):
        form_left, form_right = st.columns([1.3, 1])

        with form_left:
            uploaded_file = st.file_uploader(
                "Upload evaluation CSV",
                type=["csv"],
                help="Must include a 'target' column for ground-truth comparison.",
            )

        with form_right:
            model_choice = st.selectbox(
                "Select trained model",
                list(MODEL_FILES.keys()),
                help="Choose which saved classifier to use for prediction.",
            )
            st.caption(MODEL_DESCRIPTIONS[model_choice])

        submitted = st.form_submit_button("Evaluate", type="primary", use_container_width=True)

    if submitted and uploaded_file is not None:
        test_data = pd.read_csv(uploaded_file)
        st.session_state["test_data"] = test_data
        st.session_state["model_choice"] = model_choice
        st.session_state["evaluated"] = True
    elif submitted and uploaded_file is None:
        st.warning("Upload a CSV file before evaluating.")

    if st.session_state.get("evaluated"):
        test_data = st.session_state["test_data"]

        with st.container(border=True):
            st.markdown("**Dataset**")
            st.caption(f"{len(test_data)} rows, {test_data.shape[1]} columns")

            page_size = 10
            total_rows = len(test_data)
            total_pages = max(1, (total_rows + page_size - 1) // page_size)

            if "data_page" not in st.session_state:
                st.session_state["data_page"] = 1
            page_num = st.session_state["data_page"]
            start_idx = (page_num - 1) * page_size
            end_idx = min(start_idx + page_size, total_rows)

            st.dataframe(test_data.iloc[start_idx:end_idx], use_container_width=True, hide_index=True)

            pgn_left, pgn_center, pgn_right = st.columns([1, 2, 1])
            with pgn_left:
                if st.button("← Previous", disabled=(page_num <= 1), use_container_width=True, key="data_prev"):
                    st.session_state["data_page"] = max(1, page_num - 1)
                    st.rerun()
            with pgn_center:
                st.markdown(f"<div style='text-align:center; padding:0.5rem 0; color:#555;'>Page {page_num} of {total_pages}</div>", unsafe_allow_html=True)
            with pgn_right:
                if st.button("Next →", disabled=(page_num >= total_pages), use_container_width=True, key="data_next"):
                    st.session_state["data_page"] = min(total_pages, page_num + 1)
                    st.rerun()

# --- Results Accordion ---
if st.session_state.get("evaluated"):
    test_data = st.session_state["test_data"]
    model_choice = st.session_state["model_choice"]

    if "target" not in test_data.columns:
        st.error("The uploaded CSV must contain a 'target' column.")
    else:
        from sklearn.metrics import (
            accuracy_score, roc_auc_score, precision_score,
            recall_score, f1_score, matthews_corrcoef,
            confusion_matrix, classification_report,
        )
        import matplotlib.pyplot as plt
        import seaborn as sns

        X_test = test_data.drop(columns=["target"])
        y_test = test_data["target"]
        model_path, needs_scaling = MODEL_FILES[model_choice]

        try:
            with st.spinner("Loading model and computing predictions..."):
                model = joblib.load(model_path)
                scaler = joblib.load("models/scaler.pkl")

                X_eval = scaler.transform(X_test) if needs_scaling else X_test
                y_pred = model.predict(X_eval)
                y_proba = model.predict_proba(X_eval)[:, 1] if hasattr(model, "predict_proba") else y_pred

            with st.spinner("Calculating metrics..."):
                acc = accuracy_score(y_test, y_pred)
                auc = roc_auc_score(y_test, y_proba)
                prec = precision_score(y_test, y_pred)
                rec = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                mcc = matthews_corrcoef(y_test, y_pred)
                cm = confusion_matrix(y_test, y_pred)
                report_df = pd.DataFrame(
                    classification_report(y_test, y_pred, output_dict=True)
                ).transpose()

            with st.expander("Results", expanded=True):
                st.markdown(f"**Model Loaded:** {model_choice}")

                # KPI cards
                c1, c2, c3, c4, c5, c6 = st.columns(6)
                for col, label, val in [
                    (c1, "Accuracy", acc), (c2, "AUC", auc), (c3, "Precision", prec),
                    (c4, "Recall", rec), (c5, "F1 Score", f1), (c6, "MCC", mcc),
                ]:
                    with col:
                        with st.container(border=True):
                            st.metric(label, f"{val:.4f}")

                # Confusion Matrix & Prediction Breakdown side by side
                chart_col, table_col = st.columns([1, 1.1])

                with chart_col:
                    with st.container(border=True):
                        st.markdown("#### Confusion Matrix")
                        fig, ax = plt.subplots(figsize=(5.5, 4))
                        sns.heatmap(
                            cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                            xticklabels=["Malignant (0)", "Benign (1)"],
                            yticklabels=["Malignant (0)", "Benign (1)"],
                        )
                        ax.set_ylabel("Actual")
                        ax.set_xlabel("Predicted")
                        st.pyplot(fig)

                with table_col:
                    with st.container(border=True):
                        st.markdown("#### Prediction Breakdown")
                        comparison_df = pd.DataFrame({
                            "Actual": y_test.reset_index(drop=True),
                            "Predicted": pd.Series(y_pred).reset_index(drop=True),
                        })
                        comparison_df["Result"] = comparison_df.apply(
                            lambda r: "Correct" if r["Actual"] == r["Predicted"] else "Mismatch", axis=1
                        )

                        pred_page_size = 10
                        pred_total = len(comparison_df)
                        pred_pages = max(1, (pred_total + pred_page_size - 1) // pred_page_size)
                        if "pred_page" not in st.session_state:
                            st.session_state["pred_page"] = 1
                        pred_page = st.session_state["pred_page"]
                        pred_start = (pred_page - 1) * pred_page_size
                        pred_end = min(pred_start + pred_page_size, pred_total)

                        st.dataframe(comparison_df.iloc[pred_start:pred_end], use_container_width=True, hide_index=True)

                        p_left, p_center, p_right = st.columns([1, 2, 1])
                        with p_left:
                            if st.button("← Previous", disabled=(pred_page <= 1), use_container_width=True, key="pred_prev"):
                                st.session_state["pred_page"] = max(1, pred_page - 1)
                                st.rerun()
                        with p_center:
                            st.markdown(f"<div style='text-align:center; padding:0.5rem 0; color:#555;'>Page {pred_page} of {pred_pages}</div>", unsafe_allow_html=True)
                        with p_right:
                            if st.button("Next →", disabled=(pred_page >= pred_pages), use_container_width=True, key="pred_next"):
                                st.session_state["pred_page"] = min(pred_pages, pred_page + 1)
                                st.rerun()

                # Classification Report
                with st.container(border=True):
                    st.markdown("#### Classification Report")
                    st.dataframe(
                        report_df.round(4),
                        use_container_width=True,
                        height=(len(report_df) + 1) * 35 + 20,
                    )

        except Exception as e:
            st.error(f"Error loading model or generating metrics: {e}")
