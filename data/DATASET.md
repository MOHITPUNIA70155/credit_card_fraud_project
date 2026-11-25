# 📂 Datasets

This project trains and benchmarks fraud detection models across **two complementary credit card fraud datasets**:

---

## 1. European Cardholder Transactions (ULB PCA Dataset)

Anonymized real-world credit card transactions made by European cardholders over two days.

| Property | Value |
|----------|-------|
| File | `data/creditcard.csv` (gitignored, ~151 MB) |
| Source | [Kaggle (ULB MLG)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| Total Transactions | 284,807 |
| Fraudulent Transactions | 492 (0.172%) |
| Features | 30 (`Time`, `V1`–`V28` PCA components, `Amount`) |
| Target Variable | `Class` (`0` = Legitimate, `1` = Fraud) |

---

## 2. Credit Card Fraud Detection with Concept Drift (2026)

Real-world-style behavioral banking transactions across 4 temporal drift phases (January–June 2026), enabling **time-based validation** and **human-readable SHAP explanations**.

| Property | Value |
|----------|-------|
| File | `data/credit_card_fraud_drift_2026.csv` (gitignored, ~42 MB) |
| Source | [Kaggle (Credit Card Fraud Detection with Drift 2026)](https://www.kaggle.com/datasets/mohankrishnathalla/credit-card-fraud-detection-with-drift-2026) |
| Total Transactions | 120,000 |
| Fraudulent Transactions | 11,387 (9.49%) |
| Selected Real-Time Features | 34 leakage-free numeric signals (e.g., `amount_last_1h`, `transactions_last_24h`, `prior_dispute_ratio_90d`, `merchant_velocity_24h`, `pin_retry_count_24h`, `billing_shipping_mismatch`) |
| Target Variable | `is_fraud` (`0` = Legitimate, `1` = Fraud) |

### Temporal Split Protocol
- **Phase 1 (Jan–Feb, Stable):** Training (`39,133` rows)
- **Phase 2 (Mar, Pre-Drift):** Validation (`20,445` rows)
- **Phase 3 (Apr–May, Peak Drift):** Unseen Drift Test (`40,586` rows)
- **Phase 4 (Jun, Partial Stabilization):** Unseen Recovery Test (`19,836` rows)

---

## ⚠️ Note on Large Files
Both full CSV files are excluded via `.gitignore` to keep the repository lightweight. Ready-to-test sample datasets are included in `app/sample.csv` and `app/sample_drift.csv`.
