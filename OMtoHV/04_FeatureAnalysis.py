"""
04_FeatureAnalysis.py - SHAP-based feature importance for hardness prediction.

Uses the same Leave-One-Specimen-Out Group CV as the model benchmark: for each of the 4 folds,
one specimen is held out and a Gradient Boosting Regressor is trained on the remaining three,
with Pearson collinearity screening (|r| >= 0.95) and StandardScaler fitted on the training
fold only. SHAP values are computed per fold with TreeExplainer on that fold's held-out
samples.

Because collinearity screening is refitted per fold, the retained feature set differs slightly
between folds. Only features retained in every fold are carried into the averaged importance,
so all folds contribute equally to each reported value. Publication-quality summary and bar
plots for the top 5 features are exported.
"""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 18
plt.rcParams["axes.unicode_minus"] = False

# =============================================================================
# 1. Paths & Configuration
# =============================================================================
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "data")
output_dir = os.path.join(data_dir, "figure_shap")
os.makedirs(output_dir, exist_ok=True)

input_csv = os.path.join(data_dir, "b_hv_with_features.csv")
importance_csv = os.path.join(data_dir, "e_shap_feature_importance.csv")
per_fold_csv = os.path.join(data_dir, "e_shap_per_fold.csv")

TARGET = "HV"
GROUP_COL = "SPECIMEN"
CORR_THRESHOLD = 0.95
TOP_N = 5

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

print(f"[INFO] Leave-One-Specimen-Out Group CV | {n_splits} folds | {len(df)} samples")

# =============================================================================
# 3. Per-fold training and SHAP computation
# =============================================================================
fold_importance = {}          # fold index -> {feature: mean |SHAP|}
fold_feature_sets = []
shap_by_feature = {}          # feature -> list of per-sample SHAP values across folds
value_by_feature = {}         # feature -> list of per-sample scaled feature values

for fold_idx, (train_idx, val_idx) in enumerate(outer_splits):
    df_train = df.iloc[train_idx].copy()
    df_val = df.iloc[val_idx].copy()

    # --- Collinearity screening, fitted on the training fold only ------------
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
    fold_feature_sets.append(set(selected_features))

    # --- Standardisation: fit on train, transform only on the held-out fold ---
    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_train[selected_features].values)
    y_train = df_train[TARGET].values
    X_val = scaler.transform(df_val[selected_features].values)

    print(f"\n>>> [Fold {fold_idx + 1}/{n_splits}] Held-out: {fold_labels[fold_idx]} | "
          f"Train: {len(df_train)} | Val: {len(df_val)} | {len(selected_features)} features")

    gb_model = GradientBoostingRegressor(n_estimators=200, random_state=42)
    gb_model.fit(X_train, y_train)

    explainer = shap.TreeExplainer(gb_model)
    shap_values = explainer(X_val)

    mean_abs = np.abs(shap_values.values).mean(axis=0)
    fold_importance[fold_idx] = dict(zip(selected_features, mean_abs))

    for k, feat in enumerate(selected_features):
        shap_by_feature.setdefault(feat, []).append(shap_values.values[:, k])
        value_by_feature.setdefault(feat, []).append(X_val[:, k])

# =============================================================================
# 4. Average importance over features retained in every fold
# =============================================================================
common_features = sorted(set.intersection(*fold_feature_sets))
print(f"\n[INFO] {len(common_features)} of {len(raw_feature_cols)} features retained in all "
      f"{n_splits} folds; averaged importance is computed on these.")

rows = []
for feat in common_features:
    vals = [fold_importance[f][feat] for f in range(n_splits)]
    rows.append({
        "Feature": feat,
        "Mean_Abs_SHAP": float(np.mean(vals)),
        "SD_Abs_SHAP": float(np.std(vals, ddof=1)),
        **{f"Fold{f + 1}_{fold_labels[f]}": float(fold_importance[f][feat]) for f in range(n_splits)},
    })

df_importance = pd.DataFrame(rows).sort_values("Mean_Abs_SHAP", ascending=False).reset_index(drop=True)
df_importance.insert(0, "Rank", np.arange(1, len(df_importance) + 1))
df_importance.to_csv(importance_csv, index=False, encoding="utf-8-sig")

per_fold_rows = []
for f in range(n_splits):
    for feat, val in sorted(fold_importance[f].items(), key=lambda kv: -kv[1]):
        per_fold_rows.append({
            "Fold": f + 1,
            "Held_Out": fold_labels[f],
            "Feature": feat,
            "Mean_Abs_SHAP": float(val),
            "Retained_In_All_Folds": feat in set(common_features),
        })
pd.DataFrame(per_fold_rows).to_csv(per_fold_csv, index=False, encoding="utf-8-sig")

print(f"\nTop {TOP_N} features:")
print(df_importance.head(TOP_N)[["Rank", "Feature", "Mean_Abs_SHAP", "SD_Abs_SHAP"]].to_string(
    index=False, float_format=lambda x: f"{x:.4f}"))

# =============================================================================
# 5. SHAP Summary Plot (top 5, pooled across folds)
# =============================================================================
top_features = list(df_importance.head(TOP_N)["Feature"])
shap_matrix = np.column_stack([np.concatenate(shap_by_feature[f]) for f in top_features])
value_matrix = np.column_stack([np.concatenate(value_by_feature[f]) for f in top_features])

print("\n[INFO] Generating SHAP visualizations...")

fig1 = plt.figure(figsize=(10, 4.5))

shap.summary_plot(shap_matrix, features=value_matrix, feature_names=top_features,
                  max_display=TOP_N, plot_type="dot", show=False)

ax1 = plt.gca()
ax1.tick_params(axis="both", labelsize=15)
ax1.set_xlabel("SHAP Value (Impact on Predicted HV)", fontsize=15, fontweight="bold", labelpad=15)

if len(fig1.axes) > 1:
    fig1.axes[-1].tick_params(labelsize=15)
    fig1.axes[-1].set_ylabel("Feature Value", fontsize=15, fontweight="bold", labelpad=15)

plt.subplots_adjust(left=0.30, right=0.93, top=0.75, bottom=0.32)

summary_save_path = os.path.join(output_dir, "gb_shap_summary_plot_top5.png")
plt.savefig(summary_save_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"[INFO] Saved figure: {summary_save_path}")

# =============================================================================
# 6. SHAP Bar Plot (top 5)
# =============================================================================
plt.figure(figsize=(15, 5))
ax2 = plt.gca()

top_5_features = top_features[::-1]
top_5_values = list(df_importance.head(TOP_N)["Mean_Abs_SHAP"])[::-1]

cmap = plt.get_cmap("viridis")
colors = cmap(np.array(top_5_values) / max(top_5_values))

bars = ax2.barh(top_5_features, top_5_values, color=colors, edgecolor="#222222", height=0.55)

ax2.tick_params(labelsize=20, left=False)
ax2.set_xlabel("Mean |SHAP Value| (Average Impact)", fontsize=22, fontweight="bold", labelpad=12)

ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.spines["left"].set_color("#cccccc")
ax2.spines["bottom"].set_color("#cccccc")

ax2.grid(axis="x", linestyle="--", alpha=0.4, color="#888888")
ax2.set_axisbelow(True)

max_val = max(top_5_values)
for bar in bars:
    width = bar.get_width()
    ax2.text(width + (max_val * 0.012), bar.get_y() + bar.get_height() / 2, f"{width:.3f}",
             va="center", ha="left", fontsize=20, color="#222222", fontweight="bold")

ax2.set_xlim(0, max_val * 1.12)
plt.subplots_adjust(left=0.30, right=0.93, top=0.90, bottom=0.18)

bar_save_path = os.path.join(output_dir, "gb_shap_bar_plot_top5.png")
plt.savefig(bar_save_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"[INFO] Saved figure: {bar_save_path}")

print(f"\n[INFO] Importance table -> {importance_csv}")
print(f"[INFO] Per-fold detail  -> {per_fold_csv}")
