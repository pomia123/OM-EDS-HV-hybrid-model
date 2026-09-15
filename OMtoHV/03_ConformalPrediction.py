"""
03_ConformalPrediction_CrossConformal.py - 95% conformal prediction intervals
using a cross-conformal procedure based on out-of-fold residuals.

Uses the same four-fold leave-one-specimen-out (LOSO) group cross-validation
scheme as the model benchmark. For each fold, one specimen is held out while
the model is trained on the remaining three specimens. Pearson correlation-
based feature screening (|r| >= 0.95) and StandardScaler are fitted using the
training fold only and then applied to the held-out fold.

Point predictions therefore follow the same evaluation protocol as the
benchmark, yielding identical R2, RMSE, and MAE values.

For each held-out specimen, the prediction interval half-width (q_hat) is
determined from the pooled out-of-fold absolute residuals of the other three
folds. Thus, the held-out specimen does not contribute to the calibration
residuals used to determine its own interval width. Empirical coverage is
then evaluated separately for each held-out fold and across all four folds.

Models use fixed hyperparameters. Prediction intervals and evaluation results
are saved as CSV files.
"""

import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "DejaVu Sans"
fontsize = 40
PLOT_COLOR = "#2b5c8f"

# =============================================================================
# 1. Paths & Configuration
# =============================================================================
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "data")
figure_dir = os.path.join(data_dir, "figure_conformal")
os.makedirs(figure_dir, exist_ok=True)

input_csv = os.path.join(data_dir, "b_hv_with_features.csv")
conformal_csv = os.path.join(data_dir, "d_conformal_results.csv")
conformal_fold_csv = os.path.join(data_dir, "d_conformal_per_fold.csv")
intervals_csv = os.path.join(data_dir, "d_prediction_intervals.csv")

TARGET = "HV"
GROUP_COL = "SPECIMEN"
CORR_THRESHOLD = 0.95
ALPHA = 0.05                       # 95% nominal coverage

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

models = {
    "GradientBoosting": lambda: GradientBoostingRegressor(n_estimators=200, random_state=42),
    "XGBoost": lambda: XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42, n_jobs=-1),
    "RandomForest": lambda: RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
}

print(f"[INFO] CROSS-CONFORMAL (CV+) PREDICTION INTERVALS | alpha={ALPHA} | {n_splits} folds | "
      f"nominal coverage {(1 - ALPHA) * 100:.0f}%")
print(f"[INFO] {len(df)} samples | q_hat for each fold comes from the other folds' residuals")

# =============================================================================
# 3. Out-of-fold predictions (identical protocol to the benchmark)
# =============================================================================
predictions = {m: np.full(len(df), np.nan) for m in models}
fold_of_sample = np.full(len(df), -1)

for fold_idx, (train_idx, val_idx) in enumerate(outer_splits):
    df_train = df.iloc[train_idx].copy()
    df_val = df.iloc[val_idx].copy()
    fold_of_sample[val_idx] = fold_idx

    # Collinearity screening on the training fold only
    corr_matrix = df_train[raw_feature_cols].corr().abs()
    target_corr = df_train[raw_feature_cols].apply(lambda x: x.corr(df_train[TARGET])).abs()

    removed_features = set()
    for i in range(len(raw_feature_cols)):
        for j in range(i + 1, len(raw_feature_cols)):
            c1, c2 = raw_feature_cols[i], raw_feature_cols[j]
            if c1 in removed_features or c2 in removed_features:
                continue
            if corr_matrix.loc[c1, c2] >= CORR_THRESHOLD:
                removed_features.add(c2 if target_corr[c1] >= target_corr[c2] else c1)
    selected_features = [c for c in raw_feature_cols if c not in removed_features]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_train[selected_features].values)
    y_train = df_train[TARGET].values
    X_val = scaler.transform(df_val[selected_features].values)

    print(f"\n>>> [Fold {fold_idx + 1}/{n_splits}] Held-out: {fold_labels[fold_idx]} | "
          f"Train: {len(df_train)} | Val: {len(df_val)} | {len(selected_features)} features")

    for m_name, make_model in models.items():
        est = make_model()
        est.fit(X_train, y_train)
        predictions[m_name][val_idx] = est.predict(X_val)

for m_name, p in predictions.items():
    assert not np.isnan(p).any(), f"{m_name}: {int(np.isnan(p).sum())} samples without a prediction"

# =============================================================================
# 4. CV+ interval width and empirical coverage
# =============================================================================
y_true = df[TARGET].values
fold_rows, summary_rows = [], []
qhat_per_sample = {m: np.full(len(df), np.nan) for m in models}

for m_name in models:
    y_pred = predictions[m_name]
    abs_err = np.abs(y_true - y_pred)

    for fold_idx in range(n_splits):
        val_mask = fold_of_sample == fold_idx
        calib_mask = ~val_mask                          # residuals from the other folds only

        calib_err = abs_err[calib_mask]
        n_cal = len(calib_err)
        q_level = min(1.0, np.ceil((n_cal + 1) * (1 - ALPHA)) / n_cal)
        q_hat = float(np.quantile(calib_err, q_level))
        qhat_per_sample[m_name][val_mask] = q_hat

        covered = (y_true[val_mask] >= y_pred[val_mask] - q_hat) & \
                  (y_true[val_mask] <= y_pred[val_mask] + q_hat)

        fold_rows.append({
            "Model": m_name,
            "Fold": fold_idx + 1,
            "Held_Out": fold_labels[fold_idx],
            "N_Calib": int(n_cal),
            "N_Val": int(val_mask.sum()),
            "CP_Margin_HV": round(q_hat, 4),
            "Coverage_Pct": round(float(covered.mean() * 100), 2),
            "RMSE": round(float(np.sqrt(mean_squared_error(y_true[val_mask], y_pred[val_mask]))), 4),
        })

    q_vec = qhat_per_sample[m_name]
    covered_all = (y_true >= y_pred - q_vec) & (y_true <= y_pred + q_vec)
    sub = pd.DataFrame([r for r in fold_rows if r["Model"] == m_name])

    summary_rows.append({
        "Model": m_name,
        "R2": r2_score(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "CP_Margin_HV_mean": sub["CP_Margin_HV"].mean(),
        "CP_Margin_HV_std": sub["CP_Margin_HV"].std(ddof=1),
        "Empirical_Coverage_Pct": float(covered_all.mean() * 100),
        "Coverage_fold_mean": sub["Coverage_Pct"].mean(),
        "Coverage_fold_std": sub["Coverage_Pct"].std(ddof=1),
        "Nominal_Coverage_Pct": (1 - ALPHA) * 100,
    })

df_fold = pd.DataFrame(fold_rows)
df_fold.to_csv(conformal_fold_csv, index=False, encoding="utf-8-sig")

df_summary = pd.DataFrame(summary_rows).sort_values("R2", ascending=False).reset_index(drop=True)
df_summary.to_csv(conformal_csv, index=False, encoding="utf-8-sig")

df_int = pd.DataFrame({
    "SPECIMEN": df["SPECIMEN"],
    "FILE_NAME": df["FILE_NAME"],
    "HV_Actual": y_true,
    "Fold": fold_of_sample + 1,
})
for m_name in models:
    df_int[f"Pred_{m_name}"] = predictions[m_name]
    df_int[f"Lower_{m_name}"] = predictions[m_name] - qhat_per_sample[m_name]
    df_int[f"Upper_{m_name}"] = predictions[m_name] + qhat_per_sample[m_name]
df_int.to_csv(intervals_csv, index=False, encoding="utf-8-sig")

print("\n" + "=" * 92)
print(f" CROSS-CONFORMAL (CV+) PREDICTION INTERVALS - {(1 - ALPHA) * 100:.0f}% NOMINAL")
print("=" * 92)
print(df_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\nPer-fold detail:")
print(df_fold.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# =============================================================================
# 5. Figures
# =============================================================================
fig, axes = plt.subplots(nrows=1, ncols=len(models), figsize=(11 * len(models), 10))
axes = np.atleast_1d(axes)

# Shared axis range across every model, wide enough to contain the error bars.
lo_candidates = [y_true.min()] + [(predictions[m] - qhat_per_sample[m]).min() for m in models]
hi_candidates = [y_true.max()] + [(predictions[m] + qhat_per_sample[m]).max() for m in models]
data_min, data_max = min(lo_candidates), max(hi_candidates)
pad = (data_max - data_min) * 0.04
min_val, max_val = data_min - pad, data_max + pad

for i, m_name in enumerate(models):
    y_pred = predictions[m_name]
    q_vec = qhat_per_sample[m_name]
    row = df_summary[df_summary["Model"] == m_name].iloc[0]
    r2, q_mean, cov = row["R2"], row["CP_Margin_HV_mean"], row["Empirical_Coverage_Pct"]

    fig_single, ax_s = plt.subplots(figsize=(10, 10))
    ax_s.errorbar(y_true, y_pred, yerr=q_vec, fmt="o", color=PLOT_COLOR, ecolor="#9ebcda",
                  elinewidth=0.8, capsize=2, alpha=0.6,
                  label=f"$\\pm${q_mean:.2f} HV ({cov:.1f}%)")
    ax_s.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label=f"$R^2 = {r2:.3f}$")
    ax_s.set_xlim(min_val, max_val)
    ax_s.set_ylim(min_val, max_val)
    ax_s.set_xlabel("Actual HV", fontsize=fontsize)
    ax_s.set_ylabel("Predicted HV", fontsize=fontsize)
    ax_s.tick_params(axis="both", labelsize=fontsize)
    ax_s.legend(loc="upper left", fontsize=fontsize, frameon=True)
    ax_s.grid(True, linestyle=":", alpha=0.6)
    fig_single.tight_layout()
    fig_single.savefig(os.path.join(figure_dir, f"{m_name.lower()}_conformal.png"),
                       dpi=300, bbox_inches="tight")
    plt.close(fig_single)

    ax = axes[i]
    ax.errorbar(y_true, y_pred, yerr=q_vec, fmt="o", color=PLOT_COLOR, ecolor="#9ebcda",
                elinewidth=0.8, capsize=2, alpha=0.5,
                label=f"{m_name}: $\\pm${q_mean:.2f} HV")
    ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label=f"$R^2 = {r2:.3f}$")
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_xlabel("Actual HV", fontsize=fontsize)
    ax.set_ylabel("Predicted HV", fontsize=fontsize)
    ax.tick_params(axis="both", labelsize=fontsize)
    ax.legend(loc="upper left", fontsize=fontsize, frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)

fig.tight_layout()
fig.savefig(os.path.join(figure_dir, "conformal_comparison_all.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"\n[INFO] Figures        -> {figure_dir}")
print(f"[INFO] Summary        -> {conformal_csv}")
print(f"[INFO] Per-fold       -> {conformal_fold_csv}")
print(f"[INFO] Intervals      -> {intervals_csv}")
