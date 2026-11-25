import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

st.set_page_config(
    page_title="Credit Card Fraud & Risk Engine",
    page_icon="🛡️",
    layout="wide",
)

# Resolve project root dynamically so app runs from root or /app folder
BASE_DIR = Path(__file__).resolve().parent.parent

MODELS = {
    "European PCA Dataset (V1–V28)": {
        "model_path": BASE_DIR / "model" / "fraud_pipeline.joblib",
        "sample_path": BASE_DIR / "app" / "sample.csv",
        "target_col": "Class",
        "amount_col": "Amount",
        "desc": "Trained on ULB European cardholder transactions (284k rows, 0.17% fraud) using SMOTE + Random Forest.",
    },
    "Banking Behavioral Drift Dataset (2026)": {
        "model_path": BASE_DIR / "model" / "drift_fraud_pipeline.joblib",
        "sample_path": BASE_DIR / "app" / "sample_drift.csv",
        "target_col": "is_fraud",
        "amount_col": "transaction_amount",
        "desc": "Trained on 34 real-time banking & velocity features with a temporal split (Jan–Feb train, Mar–Jun drift test).",
    },
}


@st.cache_resource
def load_pipeline(model_path: str):
    return joblib.load(model_path)


def get_scaler_and_model(pipe):
    scaler = pipe.named_steps.get("scaler") if hasattr(pipe, "named_steps") else None
    model = pipe.steps[-1][1] if hasattr(pipe, "steps") else pipe
    return scaler, model


def get_shap_values(pipe, X_df: pd.DataFrame, max_rows: int = 150):
    scaler, model = get_scaler_and_model(pipe)
    X_sub = X_df.head(max_rows).copy()
    X_scaled = scaler.transform(X_sub) if scaler is not None else X_sub.values

    explainer = shap.TreeExplainer(model)
    raw = explainer.shap_values(X_scaled)

    # Support both older list format and newer 3D array format in SHAP
    if isinstance(raw, list):
        shap_vals = np.asarray(raw[1])
    elif isinstance(raw, np.ndarray) and raw.ndim == 3:
        shap_vals = raw[:, :, 1]
    else:
        shap_vals = np.asarray(raw)

    base = explainer.expected_value
    base_val = float(np.asarray(base).ravel()[-1]) if isinstance(base, (list, np.ndarray)) else float(base)
    return shap_vals, base_val, X_sub


def simulate_thresholds(y_true, probs, avg_loss, inv_cost):
    rows = []
    for t in np.round(np.arange(0.05, 0.96, 0.01), 2):
        preds = (probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
        prevented = tp * avg_loss
        missed = fn * avg_loss
        ops_cost = (tp + fp) * inv_cost
        rows.append({
            "Threshold": t,
            "Net_Savings": prevented - ops_cost,
            "Total_Cost": missed + ops_cost,
        })
    return pd.DataFrame(rows)


# --- SIDEBAR ---
st.sidebar.header("🗂️ Select Model & Dataset")
selected_name = st.sidebar.selectbox("Active Model", list(MODELS.keys()))
cfg = MODELS[selected_name]
pipeline = load_pipeline(str(cfg["model_path"]))
target_col = cfg["target_col"]
amount_col = cfg["amount_col"]

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Decision & Cost Controls")
threshold = st.sidebar.slider(
    "Fraud Probability Threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.50,
    step=0.01,
)
avg_fraud_loss = st.sidebar.number_input("Avg Loss per Missed Fraud (₹)", value=15000, min_value=0, step=1000)
investigation_cost = st.sidebar.number_input("Cost per Alert Investigation (₹)", value=150, min_value=0, step=50)

# --- HEADER ---
st.title("🛡️ Credit Card Fraud Detection & Risk Engine")
st.caption(cfg["desc"])

# --- DATA INPUT ---
col1, col2 = st.columns([3, 1])
with col1:
    uploaded_file = st.file_uploader(f"Upload CSV for {selected_name}", type="csv")
with col2:
    st.write("")
    st.write("")
    use_sample = st.checkbox("Use Built-in Sample Dataset", value=(uploaded_file is None))

data = None
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
elif use_sample and cfg["sample_path"].exists():
    data = pd.read_csv(cfg["sample_path"])
    st.info(f"Loaded sample dataset (`{len(data)}` rows) for **{selected_name}**.")

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs([
    "🚨 Flagged Transactions",
    "💰 Cost-Benefit Optimizer",
    "🧠 SHAP & Feature Importance",
    "🏋️ Train on Custom CSV",
])

if data is not None:
    has_labels = target_col in data.columns
    features = data.drop(columns=[target_col], errors="ignore")

    # Align columns if user uploads a full raw CSV with extra metadata columns
    if hasattr(pipeline, "feature_names_in_"):
        expected_cols = list(pipeline.feature_names_in_)
        missing = [c for c in expected_cols if c not in features.columns]
        if missing:
            st.error(f"Uploaded CSV is missing required columns: {missing[:5]}")
            st.stop()
        features = features[expected_cols]

    fraud_probs = pipeline.predict_proba(features)[:, 1]
    preds = (fraud_probs >= threshold).astype(int)

    # Keep bin edges strictly increasing even when threshold <= 0.25
    watch_cutoff = min(0.25, round(threshold * 0.5, 4))
    results = data.copy()
    results["Fraud_Probability"] = np.round(fraud_probs, 4)
    results["Flagged"] = preds == 1
    results["Risk_Tier"] = pd.cut(
        fraud_probs,
        bins=[-0.01, watch_cutoff, threshold, 1.01],
        labels=["🟢 Low Risk", "🟡 Watchlist", "🔴 High Risk"],
    )

    # KPI Row
    total_txns = len(results)
    flagged_txns = int(results["Flagged"].sum())
    flagged_rate = (flagged_txns / total_txns) * 100 if total_txns else 0.0
    flagged_vol = results.loc[results["Flagged"], amount_col].sum() if amount_col in results.columns else 0.0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Transactions", f"{total_txns:,}")
    k2.metric("Flagged as Fraud", f"{flagged_txns:,}", f"{flagged_rate:.2f}% of batch")
    k3.metric("Flagged Amount", f"{flagged_vol:,.2f}")
    k4.metric("Active Threshold", f"{threshold:.2f}")

    with tab1:
        view = st.radio("Filter:", ["Flagged Only", "All Transactions"], horizontal=True)
        show_df = results[results["Flagged"]] if view == "Flagged Only" else results
        st.dataframe(show_df.sort_values("Fraud_Probability", ascending=False), use_container_width=True)
        st.download_button(
            "⬇️ Download Scored Report (CSV)",
            results.to_csv(index=False).encode("utf-8"),
            "fraud_scored_report.csv",
            "text/csv",
        )

    with tab2:
        if has_labels:
            y_true = data[target_col].astype(int).values
            prec = precision_score(y_true, preds, zero_division=0)
            rec = recall_score(y_true, preds, zero_division=0)
            f1 = f1_score(y_true, preds, zero_division=0)
            tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()

            m1, m2, m3 = st.columns(3)
            m1.metric("Precision", f"{prec:.2%}")
            m2.metric("Recall", f"{rec:.2%}")
            m3.metric("F1-Score", f"{f1:.2%}")

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Confusion Matrix**")
                st.table(pd.DataFrame(
                    [[tn, fp], [fn, tp]],
                    index=["Actual Legit (0)", "Actual Fraud (1)"],
                    columns=["Predicted Legit (0)", "Predicted Fraud (1)"],
                ))
            with c2:
                prevented = tp * avg_fraud_loss
                missed = fn * avg_fraud_loss
                ops_cost = (tp + fp) * investigation_cost
                net_savings = prevented - ops_cost

                st.markdown("**Cost-Benefit Breakdown**")
                st.write(f"- Fraud Loss Prevented (TP): ₹{prevented:,.0f}")
                st.write(f"- Missed Fraud Loss (FN): ₹{missed:,.0f}")
                st.write(f"- Investigation Cost (TP + FP): ₹{ops_cost:,.0f}")
                st.success(f"**Net Savings at {threshold:.2f}:** ₹{net_savings:,.0f}")

            st.markdown("---")
            st.subheader("📈 Optimal Threshold Simulator")
            sweep_df = simulate_thresholds(y_true, fraud_probs, avg_fraud_loss, investigation_cost)
            best = sweep_df.loc[sweep_df["Net_Savings"].idxmax()]

            o1, o2, o3 = st.columns(3)
            o1.metric("Best Threshold", f"{best['Threshold']:.2f}")
            o2.metric("Max Net Savings", f"₹{best['Net_Savings']:,.0f}")
            o3.metric("Min Total Risk Cost", f"₹{best['Total_Cost']:,.0f}")
            st.line_chart(sweep_df.set_index("Threshold")[["Net_Savings", "Total_Cost"]])
        else:
            st.warning(f"No `{target_col}` column found, so ground-truth savings cannot be computed.")
            st.info(f"Estimated Alert Investigation Cost ({flagged_txns} alerts): ₹{flagged_txns * investigation_cost:,.0f}")

    with tab3:
        c_left, c_right = st.columns(2)
        with c_left:
            st.markdown("**Global Feature Importance (Top 12)**")
            _, model = get_scaler_and_model(pipeline)
            if hasattr(model, "feature_importances_"):
                fi_df = pd.DataFrame({
                    "Feature": features.columns[:len(model.feature_importances_)],
                    "Importance": model.feature_importances_,
                }).sort_values("Importance", ascending=False).head(12)
                st.bar_chart(fi_df.set_index("Feature"))

        shap_vals, base_val, X_sub = None, None, None
        with c_right:
            st.markdown("**SHAP Global Impact (Top 12)**")
            if SHAP_AVAILABLE:
                with st.spinner("Computing SHAP values..."):
                    shap_vals, base_val, X_sub = get_shap_values(pipeline, features)
                shap_df = pd.DataFrame({
                    "Feature": X_sub.columns,
                    "Mean |SHAP|": np.abs(shap_vals).mean(axis=0),
                }).sort_values("Mean |SHAP|", ascending=False).head(12)
                st.bar_chart(shap_df.set_index("Feature"))
            else:
                st.info("Install `shap` (`pip install shap`) to enable SHAP charts.")

        if shap_vals is not None and X_sub is not None:
            st.markdown("---")
            st.subheader("🔍 Single-Transaction SHAP Explanation")
            sub_probs = fraud_probs[:len(X_sub)]
            default_idx = int(np.argmax(sub_probs))
            idx = st.selectbox(
                "Pick a transaction to inspect:",
                options=list(range(len(X_sub))),
                index=default_idx,
                format_func=lambda i: f"Row #{i} — Fraud Prob: {sub_probs[i]:.2%}",
            )

            local_df = pd.DataFrame({
                "Feature": X_sub.columns,
                "Value": X_sub.iloc[idx].values,
                "SHAP": shap_vals[idx],
            })
            local_df["Abs"] = local_df["SHAP"].abs()
            top10 = local_df.sort_values("Abs", ascending=False).head(10).sort_values("SHAP")

            fig, ax = plt.subplots(figsize=(8, 4.2))
            colors = ["#d9534f" if v > 0 else "#337ab7" for v in top10["SHAP"]]
            ax.barh(top10["Feature"], top10["SHAP"], color=colors)
            ax.axvline(0, color="#666", linewidth=0.8, linestyle="--")
            ax.set_xlabel("SHAP Impact (+ pushes toward Fraud, - pushes toward Legit)")
            ax.set_title(f"Row #{idx} | Base: {base_val:.3f} → Predicted Prob: {sub_probs[idx]:.2%}")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

with tab4:
    st.subheader("🏋️ Train a Fresh Model on Any Labeled CSV")
    st.write("Upload any fraud/classification CSV with numeric columns and a binary target (0/1) to train and evaluate a live Random Forest pipeline.")
    custom_file = st.file_uploader("Upload Training CSV", type="csv", key="custom_train_csv")

    if custom_file is not None:
        raw_df = pd.read_csv(custom_file)
        st.write(f"Loaded `{raw_df.shape[0]:,}` rows and `{raw_df.shape[1]}` columns.")
        target_pick = st.selectbox("Select Binary Target Column (0/1):", list(raw_df.columns), index=len(raw_df.columns) - 1)
        n_trees = st.slider("Number of Trees (Random Forest)", 50, 200, 100, 25)

        if st.button("🚀 Train & Evaluate Model"):
            num_df = raw_df.select_dtypes(include=[np.number]).dropna()
            if target_pick not in num_df.columns or num_df[target_pick].nunique() != 2:
                st.error("Target column must be numeric with exactly 2 classes (0 and 1).")
            else:
                X_c = num_df.drop(columns=[target_pick])
                y_c = num_df[target_pick].astype(int)
                X_tr, X_te, y_tr, y_te = train_test_split(X_c, y_c, test_size=0.2, stratify=y_c, random_state=42)

                custom_pipe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", RandomForestClassifier(
                        n_estimators=n_trees,
                        max_depth=12,
                        class_weight="balanced",
                        random_state=42,
                    )),
                ])
                with st.spinner("Training Random Forest pipeline..."):
                    custom_pipe.fit(X_tr, y_tr)

                c_probs = custom_pipe.predict_proba(X_te)[:, 1]
                c_preds = (c_probs >= 0.5).astype(int)

                r1, r2, r3, r4 = st.columns(4)
                r1.metric("ROC-AUC", f"{roc_auc_score(y_te, c_probs):.4f}")
                r2.metric("PR-AUC", f"{average_precision_score(y_te, c_probs):.4f}")
                r3.metric("Precision", f"{precision_score(y_te, c_preds, zero_division=0):.2%}")
                r4.metric("Recall", f"{recall_score(y_te, c_preds, zero_division=0):.2%}")

                fi = pd.DataFrame({
                    "Feature": X_c.columns,
                    "Importance": custom_pipe.named_steps["model"].feature_importances_,
                }).sort_values("Importance", ascending=False).head(10)
                st.markdown("**Top 10 Learned Features**")
                st.bar_chart(fi.set_index("Feature"))
