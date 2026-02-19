import streamlit as st
import pandas as pd
import numpy as np
import time
import tracemalloc
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
)

from models.model_zoo import (
    get_classification_models,
    get_regression_models,
    evaluate_classification,
    evaluate_regression,
)

# ---------------- UI CONFIG ----------------
st.set_page_config(page_title="AutoML Pro Dashboard", layout="wide")

st.markdown("""
    <h1 style='text-align:center; color:#2E86C1;'>🔮 AutoML Pro – Multi‑Model Evaluator</h1>
    <p style='text-align:center; font-size:18px; color:#555;'>
        Upload your dataset, choose the target, and let AutoML Pro train 15+ models with full visual analytics.
    </p>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Configuration Panel")

uploaded_file = st.sidebar.file_uploader("📂 Upload CSV dataset", type=["csv"])

test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.2, 0.05)
random_state = st.sidebar.number_input("Random state", value=42, step=1)

run_button = st.sidebar.button("🚀 Run AutoML")

# ---------------- MAIN LOGIC ----------------
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Dataset Preview")
    st.write(f"Rows: **{df.shape[0]}**, Columns: **{df.shape[1]}**")
    st.dataframe(df.head())

    target_col = st.selectbox("🎯 Select Target Column", df.columns)

    if target_col:
        df = df.dropna(subset=[target_col])
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Detect problem type
        problem_type = "classification" if y.dtype == "object" or y.nunique() <= 20 else "regression"
        st.info(f"Detected Problem Type: **{problem_type.upper()}**")

        # ---------------- ROC FIX: Convert binary text labels to 0/1 ----------------
        if problem_type == "classification" and y.nunique() == 2 and y.dtype == "object":
            y = y.astype("category").cat.codes

        # Encode
        X = pd.get_dummies(X, drop_first=True, dtype="int8").fillna(0)

        # Limit dataset size
        MAX_ROWS = 50000
        if df.shape[0] > MAX_ROWS:
            df = df.sample(MAX_ROWS, random_state=42)
            X = df.drop(columns=[target_col])
            y = df[target_col]

            # Apply ROC fix again after sampling
            if problem_type == "classification" and y.nunique() == 2 and y.dtype == "object":
                y = y.astype("category").cat.codes

            X = pd.get_dummies(X, drop_first=True, dtype="int8").fillna(0)
            st.warning(f"Dataset too large. Using sample of {MAX_ROWS} rows.")

        if run_button:
            with st.spinner("Training 15+ models... please wait ⏳"):
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state,
                    stratify=y if problem_type == "classification" else None
                )

                models = get_classification_models() if problem_type == "classification" else get_regression_models()

                results = []
                trained_models = {}

                progress = st.progress(0)
                total = len(models)

                for i, (name, model) in enumerate(models.items()):
                    try:
                        tracemalloc.start()
                        start = time.time()

                        model.fit(X_train, y_train)
                        preds = model.predict(X_test)

                        current, peak = tracemalloc.get_traced_memory()
                        tracemalloc.stop()

                        metrics = evaluate_classification(y_test, preds) if problem_type == "classification" else evaluate_regression(y_test, preds)

                        results.append({
                            "model": name,
                            "train_time_sec": time.time() - start,
                            "peak_memory_mb": peak / (1024 * 1024),
                            **metrics
                        })

                        trained_models[name] = model

                    except Exception as e:
                        results.append({"model": name, "error": str(e)})

                    progress.progress((i + 1) / total)

            # ---------------- RESULTS TABLE ----------------
            st.subheader("🏁 Model Comparison Table")
            results_df = pd.DataFrame(results)
            st.dataframe(results_df)

            metric_name = "accuracy" if problem_type == "classification" else "r2"

            best_row = results_df.loc[results_df[metric_name].idxmax()]
            best_model_name = best_row["model"]
            best_model = trained_models[best_model_name]

            st.success(f"🏆 Best Model: **{best_model_name}** with **{metric_name} = {best_row[metric_name]:.4f}**")

            # ---------------- PERFORMANCE CHART ----------------
            st.subheader("📊 Performance Comparison")
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(data=results_df, x="model", y=metric_name, ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)

            # ---------------- TRAIN TIME ----------------
            st.subheader("⏱️ Training Time Comparison")
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(data=results_df, x="model", y="train_time_sec", ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)

            # ---------------- MEMORY USAGE ----------------
            st.subheader("💾 Memory Usage Comparison")
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(data=results_df, x="model", y="peak_memory_mb", ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)

            # ---------------- CLASSIFICATION VISUALS ----------------
            if problem_type == "classification":
                preds = best_model.predict(X_test)

                # Confusion Matrix
                st.subheader("🔍 Confusion Matrix")
                cm = confusion_matrix(y_test, preds)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                st.pyplot(fig)

                # ROC Curve
                if len(np.unique(y_test)) == 2 and hasattr(best_model, "predict_proba"):
                    st.subheader("📈 ROC Curve")
                    proba = best_model.predict_proba(X_test)[:, 1]
                    fpr, tpr, _ = roc_curve(y_test, proba)
                    fig, ax = plt.subplots()
                    ax.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.3f}")
                    ax.plot([0, 1], [0, 1], "k--")
                    st.pyplot(fig)

                    # Precision–Recall
                    st.subheader("📉 Precision–Recall Curve")
                    precision, recall, _ = precision_recall_curve(y_test, proba)
                    fig, ax = plt.subplots()
                    ax.plot(recall, precision)
                    st.pyplot(fig)

            # ---------------- REGRESSION VISUALS ----------------
            if problem_type == "regression":
                preds = best_model.predict(X_test)
                residuals = y_test - preds

                col1, col2 = st.columns(2)

                with col1:
                    st.write("📉 Residual Plot")
                    fig, ax = plt.subplots()
                    ax.scatter(preds, residuals, alpha=0.5)
                    ax.axhline(0, color="red", linestyle="--")
                    st.pyplot(fig)

                with col2:
                    st.write("📈 Prediction vs Actual")
                    fig, ax = plt.subplots()
                    ax.scatter(y_test, preds, alpha=0.5)
                    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
                    st.pyplot(fig)

            # ---------------- DOWNLOAD BEST MODEL ----------------
            st.subheader("💾 Download Best Model")
            buffer = pickle.dumps(best_model)
            st.download_button(
                label=f"Download {best_model_name} (.pkl)",
                data=buffer,
                file_name=f"best_model_{best_model_name.replace(' ', '_')}.pkl",
                mime="application/octet-stream",
            )