"""
Model comparison for HV prediction from OM image features, under leave-one-specimen-out (LOSO) CV.

Pipeline:
- Input: data/b_hv_with_features.csv (one row per image; SPECIMEN, FILE_NAME, HV, image features).
  Missing feature values are filled with 0.
- Outer CV: each specimen is held out in turn (K folds = number of specimens), so images from the
  same specimen never appear in both training and validation.
- Preprocessing, fitted on the training fold only:
    1) Collinearity screening: for each feature pair with |r| >= 0.95, the feature with the weaker
       correlation to HV is dropped.
    2) Standardisation (StandardScaler), applied to the validation fold using training statistics.
- Models (fixed hyperparameters, no search): Ridge, Lasso, SVR, RandomForest, GradientBoosting, XGBoost.
- Metrics: R2, RMSE and MAE on the pooled out-of-fold predictions, plus the per-fold mean and SD.
- Statistical test: the held-out specimen is the unit of analysis. For each model, the per-fold MAE
  difference against the best model (highest pooled R2) is tested with a two-sided paired t-test
  (df = K - 1). Holm correction is applied across the pairwise comparisons, and the mean MAE
  difference is reported with its 95% CI.

Outputs (data/, _v2 suffix):
  c_model_comparison_results_v2.csv  over significance per model
  c_per_fold_metrics_v2.csv          R2 / RMSE / MAE per model and fold
  c_specimen_level_test_v2.csv       paired t-test details (per-fold dMAE, CI, p_raw, p_holm)
  c_predictions_all_models_v2.csv    out-dels
  c_preprocessing_audit_log_v2.csv   per-fold train/val sizes and removed features
  c_feature_removal_pairs_v2.csv     feature pairs removed by the collinearity screening
No figures are produced.
"""
import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats

from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

# =============================================================================
# 1. Paths & Configuration
# =============================================================================
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "data")

input_csv = os.path.join(data_dir, "b_hv_with_features.csv")
results_csv = os.path.join(data_dir, "c_model_comparison_results.csv")
per_fold_csv = os.path.join(data_dir, "c_per_fold_metrics.csv")
specimen_test_csv = os.path.join(data_dir, "c_specimen_level_test.csv")
predictions_csv = os.path.join(data_dir, "c_predictions_all_models.csv")
preproc_log_csv = os.path.join(data_dir, "c_preprocessing_audit_log.csv")
feature_pairs_csv = os.path.join(data_dir, "c_feature_removal_pairs.csv")

TARGET = "HV"
GROUP_COL = "SPECIMEN"
CORR_THRESHOLD = 0.95
ALPHA = 0.05

# =============================================================================
# 2. Load Dataset & Define Outer Group Splits (Leave-One-Specimen-Out)
# =============================================================================
df = pd.read_csv(input_csv)
meta_cols = ["SPECIMEN", "FILE_NAME", TARGET]
raw_feature_cols = [c for c in df.columns if c not in meta_cols]
df[raw_feature_cols] = df[raw_feature_cols].fillna(0)

specimens = sorted(df[GROUP_COL].unique())
outer_splits = [
    (np.where(df[GROUP_COL].values != s)[0], np.where(df[GROUP_COL].values == s)[0])
    for s in specimens
]
fold_labels = [f"SPC{s}" for s in specimens]
n_splits = len(outer_splits)

print(f"[INFO] Leave-One-Specimen-Out Group CV | {n_splits} outer folds | group key = '{GROUP_COL}'")
print(f"[INFO] {len(df)} samples | {len(raw_feature_cols)} candidate features")

# Hyperparameters are fixed; no search is performed.
models = {
    "Ridge": lambda: Ridge(alpha=1.0),
    "Lasso": lambda: Lasso(alpha=0.1, random_state=42, max_iter=3000),
    "SVR": lambda: SVR(kernel="rbf", C=10),
    "RandomForest": lambda: RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "GradientBoosting": lambda: GradientBoostingRegressor(n_estimators=200, random_state=42),
    "XGBoost": lambda: XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42, n_jobs=-1),
}
HYPERPARAMS = {
    "Ridge": "alpha=1.0",
    "Lasso": "alpha=0.1",
    "SVR": "kernel=rbf, C=10",
    "RandomForest": "n_estimators=200",
    "GradientBoosting": "n_estimators=200",
    "XGBoost": "n_estimators=200, learning_rate=0.05",
}

# =============================================================================
# 3. Group CV
# =============================================================================
predictions = {m: np.full(len(df), np.nan) for m in models}
per_fold_rows, preproc_rows, pair_rows = [], [], []

print("\n" + "=" * 88)
print(" LEAVE-ONE-SPECIMEN-OUT GROUP CV")
print("=" * 88)

for fold_idx, (train_idx, val_idx) in enumerate(outer_splits):
    df_train = df.iloc[train_idx].copy()
    df_val = df.iloc[val_idx].copy()
    held_out = fold_labels[fold_idx]

    print(f"\n>>> [Fold {fold_idx + 1}/{n_splits}] Held-out: {held_out} | "
          f"Train: {len(df_train)} | Val: {len(df_val)}")

    # --- 3.1 Collinearity screening, fitted on the training fold only ---------
    #     For each feature pair above the threshold, the feature with the weaker
    #     correlation to HV is dropped.
    corr_matrix = df_train[raw_feature_cols].corr().abs()
    target_corr = df_train[raw_feature_cols].apply(lambda x: x.corr(df_train[TARGET])).abs()

    removed_features = set()
    for i in range(len(raw_feature_cols)):
        for j in range(i + 1, len(raw_feature_cols)):
            c1, c2 = raw_feature_cols[i], raw_feature_cols[j]
            if c1 in removed_features or c2 in removed_features:
                continue
            r_val = corr_matrix.loc[c1, c2]
            if r_val >= CORR_THRESHOLD:
                if target_corr[c1] >= target_corr[c2]:
                    kept, dropped = c1, c2
                else:
                    kept, dropped = c2, c1
                removed_features.add(dropped)
                pair_rows.append({
                    "Fold": fold_idx + 1,
                    "Held_Out": held_out,
                    "Retained Feature": kept,
                    "Removed Feature": dropped,
                    "Inter-feature Correlation": round(float(r_val), 4),
                    "Target Correlation (Retained)": round(float(target_corr[kept]), 4),
                    "Target Correlation (Removed)": round(float(target_corr[dropped]), 4),
                })

    selected_features = [c for c in raw_feature_cols if c not in removed_features]
    print(f"    - [Collinearity |r|<{CORR_THRESHOLD}] retained {len(selected_features)}"
          f" / {len(raw_feature_cols)} features ({len(removed_features)} removed)")

    # --- 3.2 Standardisation: fit on train, transform only on validation ------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_train[selected_features].values)
    y_train = df_train[TARGET].values
    X_val = scaler.transform(df_val[selected_features].values)
    y_val = df_val[TARGET].values

    preproc_rows.append({
        "Fold": fold_idx + 1,
        "Held_Out": held_out,
        "N_Train": int(len(df_train)),
        "N_Val": int(len(df_val)),
        "N_Features_Retained": int(len(selected_features)),
        "N_Features_Removed": int(len(removed_features)),
        "Features_Removed": "; ".join(sorted(removed_features)),
        "Scaler_Fitted_On": "training fold only",
    })

    for m_name, make_model in models.items():
        est = make_model()
        est.fit(X_train, y_train)
        y_val_pred = est.predict(X_val)
        predictions[m_name][val_idx] = y_val_pred

        per_fold_rows.append({
            "Model": m_name,
            "Fold": fold_idx + 1,
            "Held_Out": held_out,
            "N_Val": int(len(y_val)),
            "R2": r2_score(y_val, y_val_pred),
            "RMSE": float(np.sqrt(mean_squared_error(y_val, y_val_pred))),
            "MAE": float(mean_absolute_error(y_val, y_val_pred)),
        })

for m_name, p in predictions.items():
    assert not np.isnan(p).any(), f"{m_name}: {int(np.isnan(p).sum())} samples without a prediction"

pd.DataFrame(preproc_rows).to_csv(preproc_log_csv, index=False, encoding="utf-8-sig")
df_pairs = pd.DataFrame(pair_rows)
df_pairs.to_csv(feature_pairs_csv, index=False, encoding="utf-8-sig")

# =============================================================================
# 4. Metrics and specimen-level paired t-test (per-fold MAE, Holm-corrected)
# =============================================================================
y_true = df[TARGET].values
df_per_fold = pd.DataFrame(per_fold_rows)
df_per_fold.to_csv(per_fold_csv, index=False, encoding="utf-8-sig")

rows = []
for m_name in models:
    y_pred = predictions[m_name]
    sub = df_per_fold[df_per_fold["Model"] == m_name]
    rows.append({
        "Model": m_name,
        "Hyperparameters": HYPERPARAMS[m_name],
        "R2": r2_score(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2_fold_mean": sub["R2"].mean(),
        "R2_fold_std": sub["R2"].std(ddof=1),
        "RMSE_fold_mean": sub["RMSE"].mean(),
        "RMSE_fold_std": sub["RMSE"].std(ddof=1),
        "MAE_fold_mean": sub["MAE"].mean(),
        "MAE_fold_std": sub["MAE"].std(ddof=1),
    })

df_results = pd.DataFrame(rows).sort_values("R2", ascending=False).reset_index(drop=True)
best_model_name = df_results.iloc[0]["Model"]

# Per-fold MAE matrix: rows = held-out specimen, columns = model.
mae_fold = df_per_fold.pivot(index="Held_Out", columns="Model", values="MAE").loc[fold_labels]
K = len(fold_labels)
t_crit = stats.t.ppf(1 - ALPHA / 2, df=K - 1)

# d_k = MAE_k(other) - MAE_k(best); d_k > 0 means the best model had the lower error on specimen k.
test_rows = []
for m_name in df_results["Model"]:
    if m_name == best_model_name:
        continue
    d = (mae_fold[m_name] - mae_fold[best_model_name]).values
    d_mean, d_sd = d.mean(), d.std(ddof=1)
    se = d_sd / np.sqrt(K)
    t_stat = d_mean / se
    row = {
        "Model": m_name,
        "Reference": best_model_name,
        "Folds_Ref_Better": f"{int((d > 0).sum())}/{K}",
        "Mean_dMAE": d_mean,
        "SD_dMAE": d_sd,
        "CI95_Low": d_mean - t_crit * se,
        "CI95_High": d_mean + t_crit * se,
        "t": t_stat,
        "df": K - 1,
        "p_raw": float(2 * stats.t.sf(abs(t_stat), df=K - 1)),
    }
    row.update({f"dMAE_{lbl}": v for lbl, v in zip(fold_labels, d)})
    test_rows.append(row)

# Holm step-down correction over the pairwise comparisons against the best model.
df_test = pd.DataFrame(test_rows).sort_values("p_raw").reset_index(drop=True)
m_tests = len(df_test)
df_test["p_holm"] = np.maximum.accumulate(
    np.minimum(1.0, (m_tests - np.arange(m_tests)) * df_test["p_raw"].values))
df_test["Significance"] = np.where(df_test["p_holm"] < ALPHA,
                                   f"Significant (Holm p<{ALPHA})", "Not Significant")
df_test.to_csv(specimen_test_csv, index=False, encoding="utf-8-sig")

p_holm = dict(zip(df_test["Model"], df_test["p_holm"]))
p_holm[best_model_name] = np.nan
df_results["Specimen_t_vs_Best_p_Holm"] = df_results["Model"].map(p_holm)
df_results["Significance"] = df_results["Model"].map(
    dict(zip(df_test["Model"], df_test["Significance"]), **{best_model_name: "Best Model (Ref)"}))
df_results.to_csv(results_csv, index=False, encoding="utf-8-sig")

df_preds = pd.DataFrame({"SPECIMEN": df["SPECIMEN"], "FILE_NAME": df["FILE_NAME"], "HV_Actual": y_true})
for m_name in models:
    df_preds[f"Pred_{m_name}"] = predictions[m_name]
df_preds.to_csv(predictions_csv, index=False, encoding="utf-8-sig")

print("\n" + "=" * 88)
print(f" BENCHMARK (pooled out-of-fold metrics; test unit = held-out specimen, K = {K})")
print("=" * 88)
print(df_results[["Model", "R2", "RMSE", "MAE", "MAE_fold_mean", "MAE_fold_std",
                  "Specimen_t_vs_Best_p_Holm", "Significance"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print(f"\nSpecimen-level paired t-test on per-fold MAE vs {best_model_name} (two-sided, df = {K - 1}, Holm):")
print(df_test[["Model", "Folds_Ref_Better", "Mean_dMAE", "CI95_Low", "CI95_High", "t", "p_raw", "p_holm",
               "Significance"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\nPer-fold MAE:")
print(mae_fold.to_string(float_format=lambda x: f"{x:.4f}"))

print(f"\n[INFO] Results           -> {results_csv}")
print(f"[INFO] Specimen test     -> {specimen_test_csv}")
print(f"[INFO] Per-fold          -> {per_fold_csv}")
print(f"[INFO] Predictions       -> {predictions_csv}")
print(f"[INFO] Preprocessing log -> {preproc_log_csv}")
print(f"[INFO] Feature removal   -> {feature_pairs_csv}")
