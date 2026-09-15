import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "DejaVu Sans"
fontsize = 35
PLOT_COLOR = "#2b5c8f"

# =============================================================================
# 1. Paths & Configuration
# =============================================================================
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "data")
figure_dir = os.path.join(data_dir, "figure")
os.makedirs(figure_dir, exist_ok=True)

input_csv = os.path.join(data_dir, "b_hv_with_features.csv")
results_csv = os.path.join(data_dir, "c_model_comparison_results.csv")
per_fold_csv = os.path.join(data_dir, "c_per_fold_metrics.csv")
predictions_csv = os.path.join(data_dir, "c_predictions_all_models.csv")
preproc_log_csv = os.path.join(data_dir, "c_preprocessing_audit_log.csv")
feature_pairs_csv = os.path.join(data_dir, "c_feature_removal_pairs.csv")

TARGET = "HV"
GROUP_COL = "SPECIMEN"
CORR_THRESHOLD = 0.95

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
# 4. Metrics and Wilcoxon signed-rank test
# =============================================================================
y_true = df[TARGET].values
df_per_fold = pd.DataFrame(per_fold_rows)
df_per_fold.to_csv(per_fold_csv, index=False, encoding="utf-8-sig")

rows, model_errors = [], {}
for m_name in models:
    y_pred = predictions[m_name]
    model_errors[m_name] = np.abs(y_true - y_pred)
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
    })

df_results = pd.DataFrame(rows).sort_values("R2", ascending=False).reset_index(drop=True)
best_model_name = df_results.iloc[0]["Model"]
err_best = model_errors[best_model_name]

p_vals, sig = {}, {}
for m_name in df_results["Model"]:
    if m_name == best_model_name:
        p_vals[m_name], sig[m_name] = 1.0, "Best Model (Ref)"
        continue
    try:
        p_vals[m_name] = float(wilcoxon(err_best, model_errors[m_name], zero_method="wilcox")[1])
    except Exception:
        p_vals[m_name] = np.nan
    sig[m_name] = ("Significant (p<0.05)" if p_vals[m_name] == p_vals[m_name] and p_vals[m_name] < 0.05
                   else "Not Significant")

df_results["Wilcoxon_vs_Best_p"] = df_results["Model"].map(p_vals)
df_results["Significance"] = df_results["Model"].map(sig)
df_results.to_csv(results_csv, index=False, encoding="utf-8-sig")

df_preds = pd.DataFrame({"SPECIMEN": df["SPECIMEN"], "FILE_NAME": df["FILE_NAME"], "HV_Actual": y_true})
for m_name in models:
    df_preds[f"Pred_{m_name}"] = predictions[m_name]
df_preds.to_csv(predictions_csv, index=False, encoding="utf-8-sig")

print("\n" + "=" * 88)
print(f" BENCHMARK (pooled out-of-fold, Wilcoxon on all {len(df)} samples vs {best_model_name})")
print("=" * 88)
print(df_results[["Model", "R2", "RMSE", "MAE", "R2_fold_mean", "R2_fold_std",
                  "Wilcoxon_vs_Best_p", "Significance"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\nPer-fold detail:")
print(df_per_fold.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# =============================================================================
# 5. Figures
# =============================================================================
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(22, 14))
axes = axes.flatten()

# Shared axis range across every model, so the panels are directly comparable.
all_pred = np.concatenate([predictions[m] for m in models])
data_min = min(y_true.min(), all_pred.min())
data_max = max(y_true.max(), all_pred.max())
pad = (data_max - data_min) * 0.04
min_val, max_val = data_min - pad, data_max + pad

for i, m_name in enumerate(models):
    y_pred = predictions[m_name]
    r2 = df_results[df_results["Model"] == m_name].iloc[0]["R2"]

    fig_single, ax_single = plt.subplots(figsize=(10, 10))
    ax_single.scatter(y_true, y_pred, alpha=0.6, edgecolors="w", color=PLOT_COLOR, s=50)
    ax_single.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label=f"$R^2 = {r2:.3f}$")
    ax_single.set_xlim(min_val, max_val)
    ax_single.set_ylim(min_val, max_val)
    ax_single.set_xlabel("Actual HV", fontsize=fontsize)
    ax_single.set_ylabel("Predicted HV", fontsize=fontsize)
    ax_single.tick_params(axis="both", labelsize=fontsize)
    ax_single.legend(loc="upper left", fontsize=fontsize, frameon=True)
    ax_single.grid(True, linestyle=":", alpha=0.6)
    fig_single.tight_layout()
    fig_single.savefig(os.path.join(figure_dir, f"{m_name.lower()}_actual_vs_predicted.png"),
                       dpi=300, bbox_inches="tight")
    plt.close(fig_single)

    ax = axes[i]
    ax.scatter(y_true, y_pred, alpha=0.5, edgecolors="w", color=PLOT_COLOR)
    ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label=f"{m_name}: $R^2 = {r2:.3f}$")
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_xlabel("Actual HV", fontsize=fontsize)
    ax.set_ylabel("Predicted HV", fontsize=fontsize)
    ax.tick_params(axis="both", labelsize=fontsize)
    ax.legend(loc="upper left", fontsize=fontsize, frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)

for j in range(len(models), len(axes)):
    fig.delaxes(axes[j])
fig.tight_layout()
fig.savefig(os.path.join(figure_dir, "cv_actual_vs_predicted_all.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"\n[INFO] Figures           -> {figure_dir}")
print(f"[INFO] Results           -> {results_csv}")
print(f"[INFO] Per-fold          -> {per_fold_csv}")
print(f"[INFO] Predictions       -> {predictions_csv}")
print(f"[INFO] Preprocessing log -> {preproc_log_csv}")
print(f"[INFO] Feature removal   -> {feature_pairs_csv}")
