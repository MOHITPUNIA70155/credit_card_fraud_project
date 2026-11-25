import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Credit Card Fraud & Risk Engine",
    page_icon="🛡️",
    layout="wide"
)

# Resolve paths dynamically so app runs cleanly from root or /app directory
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "fraud_pipeline.joblib"
SAMPLE_DATA_PATH = BASE_DIR / "app" / "sample.csv"


@st.cache_resource
def load_pipeline():
    return joblib.load(MODEL_PATH)


def extract_scaler_and_estimator(pipe):
    """Safely extract preprocessing scaler and final estimator from pipeline."""
    scaler = None
    estimator = pipe
    if hasattr(pipe, "named_steps"):
        scaler = pipe.named_steps.get("scaler")
        estimator = pipe.named_steps.get("model", pipe.steps[-1][1])
    elif hasattr(pipe, "steps"):
        estimator = pipe.steps[-1][1]
    return scaler, estimator


def compute_shap_values(_pipe, X_df: pd.DataFrame, max_rows: int = 200):
    """Compute class-1 SHAP values for tree-based or linear estimators."""
    if not SHAP_AVAILABLE:
        return None, None, None

    scaler, estimator = extract_scaler_and_estimator(_pipe)
    X_sample = X_df.head(max_rows).copy()
    X_transformed = scaler.transform(X_sample) if scaler is not None else X_sample.values

    explainer = shap.TreeExplainer(estimator)
    raw_shap = explainer.shap_values(X_transformed)

    # Handle both shap < 0.45 (list of [class0, class1]) and shap >= 0.45 (3D ndarray)
    if isinstance(raw_shap, list):
        shap_fraud = np.asarray(raw_shap[1])
    elif isinstance(raw_shap, np.ndarray) and raw_shap.ndim == 3:
        shap_fraud = raw_shap[:, :, 1]
    else:
        shap_fraud = np.asarray(raw_shap)

    expected = explainer.expected_value
    if isinstance(expected, (list, np.ndarray)):
        base_value = float(np.asarray(expected).ravel()[-1])
    else:
        base_value = float(expected)

    return shap_fraud, base_value, X_sample


def run_cost_benefit_sweep(y_true: np.ndarray, fraud_probs: np.ndarray, avg_loss: float, inv_cost: float):
    """Evaluate net savings and total risk cost across thresholds 0.05 to 0.95."""
    thresholds = np.round(np.arange(0.05, 0.96, 0.01), 2)
    rows = []
    for t in thresholds:
        p = (fraud_probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, p, labels=[0, 1]).ravel()
        prevented = tp * avg_loss
        missed = fn * avg_loss
        ops_cost = (tp + fp) * inv_cost
        net_savings = prevented - ops_cost
        total_cost = missed + ops_cost
        rows.append({
            "Threshold": t,
            "Net_Savings": net_savings,
            "Total_Cost": total_cost,
            "Prevented_Loss": prevented,
            "Missed_Loss": missed,
            "Investigation_Cost": ops_cost,
            "TP": int(tp),
            "FP": int(fp),
            "FN": int(fn),
            "TN": int(tn),
        })
    return pd.DataFrame(rows)


pipeline = load_pipeline()

# --- SIDEBAR: RISK & BUSINESS CONTROLS ---
st.sidebar.header("⚙️ Decision & Cost Controls")
threshold = st.sidebar.slider(
    "Fraud Probability Threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.50,
    step=0.01,
    help="Lower threshold catches more frauds (higher Recall) but increases false alerts."
)

st.sidebar.markdown("---")
st.sidebar.subheader("💰 Business Cost Parameters (₹)")
avg_fraud_loss = st.sidebar.number_input("Avg Loss per Missed Fraud (FN)", value=15000, min_value=0, step=1000)
investigation_cost = st.sidebar.number_input("Cost per Alert Investigation (FP/TP)", value=150, min_value=0, step=50)

# --- MAIN HEADER ---
st.title("🛡️ Credit Card Fraud Detection & Risk Engine")
st.markdown(
    "End-to-end imbalanced classification pipeline (**StandardScaler + SMOTE + Random Forest**) "
    "with dynamic threshold tuning, **SHAP explainability**, and **cost-benefit risk optimization**."
)

# --- DATA INPUT SECTION ---
col_up1, col_up2 = st.columns([3, 1])
with col_up1:
    uploaded_file = st.file_uploader("Upload Transactions CSV (V1–V28, Time, Amount)", type="csv")
with col_up2:
    st.write("")
    st.write("")
    use_sample = st.checkbox("Use Built-in Sample Dataset", value=(uploaded_file is None))

data = None
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
elif use_sample and SAMPLE_DATA_PATH.exists():
    data = pd.read_csv(SAMPLE_DATA_PATH)
    st.info(f"Loaded built-in sample dataset (`{len(data)}` transactions) for instant analysis.")

if data is not None:
    has_ground_truth = "Class" in data.columns
    features = data.drop(columns=["Class"], errors="ignore")

    # Predict fraud probabilities
    fraud_probs = pipeline.predict_proba(features)[:, 1]
    preds = (fraud_probs >= threshold).astype(int)

    # Dynamic strictly monotonic bins so threshold <= 0.25 never crashes
    watchlist_cutoff = min(0.25, round(threshold * 0.5, 4))

    results = data.copy()
    results["Fraud_Probability"] = np.round(fraud_probs, 4)
    results["Flagged"] = preds == 1
    results["Risk_Tier"] = pd.cut(
        fraud_probs,
        bins=[-0.01, watchlist_cutoff, threshold, 1.01],
        labels=["🟢 Low Risk", "🟡 Watchlist", "🔴 High Risk"]
    )

    # --- KPI METRICS ROW ---
    st.markdown("### 📊 Transaction Risk Summary")
    total_txns = len(results)
    flagged_txns = int(results["Flagged"].sum())
    flagged_rate = (flagged_txns / total_txns) * 100 if total_txns else 0.0
    high_risk_amount = results.loc[results["Flagged"], "Amount"].sum() if "Amount" in results.columns else 0.0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Transactions", f"{total_txns:,}")
    k2.metric("Flagged as Fraud", f"{flagged_txns:,}", f"{flagged_rate:.2f}% of batch")
    k3.metric("Flagged Volume (₹/$)", f"{high_risk_amount:,.2f}")
    k4.metric("Active Threshold", f"{threshold:.2f}")

    # --- TABS FOR DEEP ANALYSIS ---
    tab1, tab2, tab3 = st.tabs([
        "🚨 Flagged Transactions",
        "💰 Cost-Benefit & Threshold Optimizer",
        "🧠 SHAP & Feature Importance"
    ])

    with tab1:
        filter_option = st.radio("Filter View:", ["Flagged Only", "All Transactions"], horizontal=True)
        display_df = results[results["Flagged"]] if filter_option == "Flagged Only" else results
        st.dataframe(
            display_df.sort_values("Fraud_Probability", ascending=False),
            use_container_width=True
        )
        csv_bytes = results.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Full Scored Report (CSV)",
            csv_bytes,
            "fraud_scored_report.csv",
            "text/csv"
        )

    with tab2:
        if has_ground_truth:
            y_true = data["Class"].astype(int).values
            prec = precision_score(y_true, preds, zero_division=0)
            rec = recall_score(y_true, preds, zero_division=0)
            f1 = f1_score(y_true, preds, zero_division=0)
            tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()

            st.subheader("Live Threshold Performance (Ground Truth Available)")
            m1, m2, m3 = st.columns(3)
            m1.metric("Precision (Alert Accuracy)", f"{prec:.2%}")
            m2.metric("Recall (Frauds Caught)", f"{rec:.2%}")
            m3.metric("F1-Score", f"{f1:.2%}")

            c_left, c_right = st.columns(2)
            with c_left:
                st.markdown("**Confusion Matrix Breakdown**")
                cm_df = pd.DataFrame(
                    [[tn, fp], [fn, tp]],
                    index=["Actual Legit (0)", "Actual Fraud (1)"],
                    columns=["Predicted Legit (0)", "Predicted Fraud (1)"]
                )
                st.table(cm_df)

            with c_right:
                st.markdown("**Financial Cost-Benefit Calculator**")
                prevented_loss = tp * avg_fraud_loss
                missed_loss = fn * avg_fraud_loss
                alert_ops_cost = (tp + fp) * investigation_cost
                net_savings = prevented_loss - alert_ops_cost
                total_risk_cost = missed_loss + alert_ops_cost

                st.write(f"- **Fraud Loss Prevented (TP):** ₹{prevented_loss:,.0f}")
                st.write(f"- **Loss from Missed Frauds (FN):** ₹{missed_loss:,.0f}")
                st.write(f"- **Investigation Cost (TP + FP):** ₹{alert_ops_cost:,.0f}")
                st.write(f"- **Total Residual Cost (FN + Ops):** ₹{total_risk_cost:,.0f}")
                st.success(f"**Net Financial Savings at {threshold:.2f} threshold:** ₹{net_savings:,.0f}")

            # --- THRESHOLD OPTIMIZATION CURVE ---
            st.markdown("---")
            st.subheader("📈 Optimal Threshold Simulator")
            sweep_df = run_cost_benefit_sweep(y_true, fraud_probs, avg_fraud_loss, investigation_cost)
            best_row = sweep_df.loc[sweep_df["Net_Savings"].idxmax()]
            opt_thresh = float(best_row["Threshold"])
            opt_savings = float(best_row["Net_Savings"])
            delta_savings = opt_savings - net_savings

            o1, o2, o3 = st.columns(3)
            o1.metric("Recommended Optimal Threshold", f"{opt_thresh:.2f}")
            o2.metric("Max Net Savings", f"₹{opt_savings:,.0f}", f"+₹{delta_savings:,.0f} vs current" if delta_savings > 0 else "At optimum")
            o3.metric("Min Total Risk Cost", f"₹{best_row['Total_Cost']:,.0f}")

            chart_df = sweep_df.set_index("Threshold")[["Net_Savings", "Total_Cost"]]
            st.line_chart(chart_df)
            st.caption("Simulated Net Savings (prevented loss minus investigation cost) and Total Cost across decision thresholds.")
        else:
            st.warning(
                "Uploaded CSV does not contain a `Class` ground-truth column, so Confusion Matrix & "
                "Ground-Truth Savings cannot be computed. Showing estimated investigation cost below:"
            )
            est_ops_cost = flagged_txns * investigation_cost
            st.info(f"**Estimated Alert Investigation Cost ({flagged_txns} flagged alerts):** ₹{est_ops_cost:,.0f}")

    with tab3:
        st.subheader("🧠 Model Explainability: Feature Importance & SHAP")

        col_fi, col_shap_global = st.columns(2)
        with col_fi:
            st.markdown("**Global Model Feature Importance (Top 12)**")
            try:
                _, estimator = extract_scaler_and_estimator(pipeline)
                if hasattr(estimator, "feature_importances_"):
                    importances = estimator.feature_importances_
                elif hasattr(estimator, "coef_"):
                    importances = np.abs(estimator.coef_[0])
                else:
                    importances = None

                if importances is not None:
                    feat_df = pd.DataFrame({
                        "Feature": features.columns[:len(importances)],
                        "Importance": importances
                    }).sort_values("Importance", ascending=False).head(12)
                    st.bar_chart(feat_df.set_index("Feature"))
                    st.caption("Built-in Random Forest impurity-based feature importance.")
                else:
                    st.info("Feature importance attribute not exposed by the active pipeline estimator.")
            except Exception as e:
                st.error(f"Could not extract feature importances: {e}")

        with col_shap_global:
            st.markdown("**SHAP Global Mean |Impact| (Top 12)**")
            shap_fraud = None
            base_value = None
            X_shap_sample = None

            if SHAP_AVAILABLE:
                try:
                    with st.spinner("Computing SHAP values..."):
                        shap_fraud, base_value, X_shap_sample = compute_shap_values(pipeline, features, max_rows=200)

                    mean_abs_shap = np.abs(shap_fraud).mean(axis=0)
                    shap_global_df = pd.DataFrame({
                        "Feature": X_shap_sample.columns,
                        "Mean |SHAP Value|": mean_abs_shap
                    }).sort_values("Mean |SHAP Value|", ascending=False).head(12)
                    st.bar_chart(shap_global_df.set_index("Feature"))
                    st.caption(f"Average absolute SHAP impact on fraud probability (computed on {len(X_shap_sample)} transactions).")
                except Exception as e:
                    st.warning(f"Could not compute SHAP values: {e}")
            else:
                st.info("Install `shap` (`pip install shap`) to enable SHAP global and local transaction explanations.")

        # --- LOCAL PER-TRANSACTION SHAP EXPLANATION ---
        if shap_fraud is not None and X_shap_sample is not None:
            st.markdown("---")
            st.subheader("🔍 Individual Transaction SHAP Explanation")
            sample_probs = fraud_probs[:len(X_shap_sample)]
            default_idx = int(np.argmax(sample_probs))

            txn_labels = [
                f"Row #{i} — Fraud Prob: {sample_probs[i]:.2%} | Amount: {X_shap_sample.iloc[i]['Amount']:.2f}"
                if "Amount" in X_shap_sample.columns
                else f"Row #{i} — Fraud Prob: {sample_probs[i]:.2%}"
                for i in range(len(X_shap_sample))
            ]
            selected_idx = st.selectbox(
                "Select a transaction to inspect why the model scored it this way:",
                options=list(range(len(X_shap_sample))),
                index=default_idx,
                format_func=lambda i: txn_labels[i]
            )

            row_shap = shap_fraud[selected_idx]
            local_df = pd.DataFrame({
                "Feature": X_shap_sample.columns,
                "Feature Value": X_shap_sample.iloc[selected_idx].values,
                "SHAP Contribution": row_shap,
                "Abs Impact": np.abs(row_shap)
            }).sort_values("Abs Impact", ascending=False).head(10).sort_values("SHAP Contribution")

            fig, ax = plt.subplots(figsize=(8, 4.5))
            colors = ["#d9534f" if v > 0 else "#337ab7" for v in local_df["SHAP Contribution"]]
            ax.barh(local_df["Feature"], local_df["SHAP Contribution"], color=colors)
            ax.axvline(0, color="#666666", linewidth=0.9, linestyle="--")
            ax.set_xlabel("SHAP Contribution to Fraud Probability (+ increases risk, - decreases risk)")
            ax.set_title(
                f"Row #{selected_idx} | Base Value: {base_value:.3f} → Predicted Fraud Prob: {sample_probs[selected_idx]:.2%}"
            )
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            with st.expander("View Top 10 Feature Contributions Table"):
                st.dataframe(
                    local_df[["Feature", "Feature Value", "SHAP Contribution"]].sort_values(
                        "SHAP Contribution", ascending=False
                    ),
                    use_container_width=True
                )
