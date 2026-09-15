# Hybrid Machine Learning Framework for Microstructure-Based Composition Reconstruction and Hardness Prediction of Al–10Si-2Cu Die-Casting Alloys

Two independent analysis pipelines based on Optical Microscopy (OM) images:

- **OMtoHV**: Microstructure feature extraction → ML-based Vickers hardness (HV) prediction
- **OMtoEDS**: GAN-based deep learning → EDS elemental spatial map prediction

---

## Project Structure

```
project/
├── OMtoEDS/
│   ├── 01_HyperparamGridSearch.py
│   ├── 02_OMtoEDS_pix2pix_deep_ensemble.py
│   ├── 03_SpatialMapEvaluation_Separate.py
│   ├── 04_MultiElemOverlay.py
│   ├── 05_SpatialMapVisualization_YGB.py
│   ├── 06_UncertaintyMap.py
│   ├── 07_MicrostructureDescriptorAnalysis.py
│   ├── 07_NewSampleInference.py
│   ├── data/
│        └── EDS
│        ├── MAP
│        ├── MASK
│        ├── OM
│        ├── result
│        └── pred_data/            # Data for new sample inference
│             ├── OM_hv/
│             ├── MAP_hv/
│             └── result/
├── result/
│   └── tversky/
│       ├── models_tversky/  # Trained model weights (.pth)
│       ├── splits.json      # Train/val/test split info
│       └── test_tversky/    # Evaluation CSVs and visualizations
│
├── OMtoHV/
      ├── 01_FeatureExtraction.py
      ├── 02_ModelComparison.py
      ├── 03_ConformalPrediction.py
      ├── 04_FeatureAnalysis.py
      └── data/
           ├── OM_hv/                          # OM images (optionally in SPC{n}/ subfolders)
           ├── a_hv.csv                        # SPECIMEN, FILE_NAME, HV
           ├── b_hv_with_features.csv
           ├── c_model_comparison_results.csv
           ├── c_per_fold_metrics.csv
           ├── c_predictions_all_models.csv
           ├── c_preprocessing_audit_log.csv
           ├── c_feature_removal_pairs.csv
           ├── d_conformal_results.csv
           ├── d_conformal_per_fold.csv
           ├── d_prediction_intervals.csv
           ├── e_shap_feature_importance.csv
           ├── e_shap_per_fold.csv
           ├── figure/
           ├── figure_conformal/
           └── figure_shap/
```

---

## Pipeline 1: OMtoHV

Extracts microstructure features from OM images and predicts Vickers hardness (HV) using machine learning.

### Execution Order

```
      01_FeatureExtraction.py
    → 02_ModelComparison.py
    → 03_ConformalPrediction.py
    → 04_FeatureAnalysis.py
```

### Validation Protocol (shared by 02–04)

- **Leave-one-specimen-out (LOSO) group CV:** 4 folds, grouped by `SPECIMEN` (SPC1–SPC4 = 169 / 120 / 110 / 130 samples; 529 in total). In each fold, one specimen is held out and the models are trained on the remaining three.
- **Per-fold preprocessing (training fold only):**
  - Pearson collinearity screening ($|r| \ge 0.95$): for each highly correlated pair, the feature with the weaker correlation to HV is dropped.
  - `StandardScaler` fitted on the training fold, then applied to the held-out fold.
- **Hyperparameters:** fixed (no search).
- All 529 samples are used (no outlier removal).

### Script Descriptions

**`01_FeatureExtraction.py`**  
Extracts microstructure features from OM images in parallel (`ProcessPoolExecutor`).
- **Input:** `data/OM_hv/` (images in `SPC{n}/` subfolders are also supported), `data/a_hv.csv`
- **Output:** `data/b_hv_with_features.csv` (529 samples, 48 features)
- **Features:** secondary phase morphology, dendrite orientation, DAS, eutectic structure (fraction, lamella thickness, skeleton), GLCM texture, LBP, intensity statistics, autocorrelation length
- **Visualization:** Feature maps for the first sample only are saved to `data/figure/`

**`02_ModelComparison.py`**  
Benchmarks 6 regression models under LOSO group CV and performs Wilcoxon signed-rank tests on pooled absolute errors against the best model.
- **Input:** `data/b_hv_with_features.csv`
- **Models:** Ridge (α=1.0), Lasso (α=0.1), SVR (RBF, C=10), RandomForest (n_estimators=200), GradientBoosting (n_estimators=200), XGBoost (n_estimators=200, learning_rate=0.05)
- **Metrics:** $R^2$, RMSE, MAE (pooled out-of-fold, plus per-fold mean ± SD)
- **Output:**
  - `data/c_model_comparison_results.csv`: summary and Wilcoxon p-values
  - `data/c_per_fold_metrics.csv`: per-fold metrics
  - `data/c_predictions_all_models.csv`: out-of-fold predictions
  - `data/c_preprocessing_audit_log.csv`: per-fold retained/removed features, scaler fit scope
  - `data/c_feature_removal_pairs.csv`: removed collinear pairs
  - `data/figure/`: Actual vs. Predicted plots

|  Model  |   R²  | RMSE  |  MAE  |
|---------|-------|-------|-------|
|    GB   | 0.848 | 1.141 | 0.840 |
| XGBoost | 0.838 | 1.180 | 0.858 |
|    RF   | 0.816 | 1.257 | 0.884 |

**`03_ConformalPrediction.py`**  
Computes 95% prediction intervals with cross-conformal prediction based on out-of-fold residuals for GradientBoosting, XGBoost, and RandomForest. Point predictions follow the same LOSO protocol as `02`.
- **Input:** `data/b_hv_with_features.csv`
- **Calibration:** For each held-out specimen, the absolute out-of-fold residuals of the other three folds are pooled as the calibration set. The held-out specimen does not contribute to its own interval width.
- **Interval:** $q_{\text{level}} = \lceil (n_{\text{cal}}+1)(1-\alpha) \rceil / n_{\text{cal}}$, with $\alpha = 0.05$. $\hat{q}$ is the $q_{\text{level}}$ empirical quantile of the calibration residuals, and each interval is the prediction $\pm \hat{q}$.
- **Coverage:** Empirical coverage is evaluated per held-out fold and across all folds.
- **Output:**
  - `data/d_conformal_results.csv`: summary
  - `data/d_conformal_per_fold.csv`: per-fold $\hat{q}$ and coverage
  - `data/d_prediction_intervals.csv`: per-sample intervals
  - `data/figure_conformal/`

|  Model  | CP Margin (HV, fold mean) | Empirical Coverage |
|---------|---------------------------|--------------------|
|    GB   | ±2.08                     | 95.27%             |
| XGBoost | ±2.32                     | 95.09%             |
|    RF   | ±2.60                     | 95.09%             |

**`04_FeatureAnalysis.py`**  
Interprets feature importance of Gradient Boosting using SHAP TreeExplainer under the same LOSO protocol.
- **Input:** `data/b_hv_with_features.csv`
- **Procedure:** In each fold, GB is trained on three specimens and SHAP values are computed on the held-out specimen. Because collinearity screening is refitted per fold, only features retained in all 4 folds (38 features) are used for the averaged importance.
- **Output:**
  - `data/e_shap_feature_importance.csv`: mean ± SD |SHAP| across folds
  - `data/e_shap_per_fold.csv`
  - `data/figure_shap/gb_shap_summary_plot_top5.png`
  - `data/figure_shap/gb_shap_bar_plot_top5.png`

---

## Pipeline 2: OMtoEDS

Predicts element-specific binary EDS spatial maps from OM images using a Pix2Pix GAN with a ResNet-34 encoder, CBAM attention, and Tversky loss.

Target elements: **Mg, Al, Si, Cu, Fe, Sr**

### Execution Order

```
      01_HyperparamGridSearch.py              ← Hyperparameter optimization
    → 02_OMtoEDS_pix2pix_deep_ensemble.py     ← Deep Ensemble model training
    → 03_SpatialMapEvaluation_Separate.py     ← Per-element quantitative evaluation
    → 04_MultiElemOverlay.py                  ← Multi-element composite overlay
    → 05_SpatialMapVisualization_YGB.py       ← Match/Miss/False qualitative visualization
    → 06_UncertaintyMap.py                    ← Ensemble pixel uncertainty mapping
    → 07_MicrostructureDescriptorAnalysis.py  ← Metallurgical descriptor (PSD & NND) validation
    → 08_NewSampleInference.py                ← Inference on new unseen sample
```

### Script Descriptions

**`01_HyperparamGridSearch.py`**  
Optimizes pixel-loss weighting ($\lambda_{\text{pix}}$), focal loss hyperparameters ($\alpha, \gamma$), and focal/Tversky loss ratios for representative elements.
- **Input:** `data/OM/`, `data/EDS/`, `data/MASK/`, `data/MAP/`, `result/tversky/splits.json`
- **Output:** `result/tversky/grid_search/grid_results_{elem}.csv`, `grid_search_best_params.csv`
- **Ranking Criteria:** Minimum validation Mean Absolute Error (MAE) and Intersection over Union (IoU)

**`02_OMtoEDS_pix2pix_deep_ensemble.py`**
Trains a Deep Ensemble ($N=3$ independently trained members per element) of Pix2Pix GANs.
- **Input:** `data/OM/`, `data/EDS/`, `data/MASK/`, `data/MAP/`
- **Output:** `result/tversky/models_tversky/best_model_{elem}_{idx}_{epoch}.pth`, `last_model_{elem}_{idx}.pth`, `splits.json`
- **Architecture:** U-Net Generator (ResNet-34 encoder + CBAM attention blocks in decoder) + PatchGAN Discriminator
- **Loss:** Element-specific Tversky Loss + Focal Loss + GAN Adversarial Loss
- **Hyperparameters:** $512 \times 512$ random crop, batch size = 32, epochs = 1001, $\text{Adam } (\text{lr}=2\times 10^{-4})$

**`03_SpatialMapEvaluation_Separate.py`**
Evaluates the test set across individual members, majority voting, and deep ensemble mean predictions for both `best` and `last` model checkpoints.
- **Input:** `data/OM/`, `data/EDS/`, `data/MASK/`, `data/MAP/`, `result/tversky/splits.json`
- **Output:** `result/test_tversky/results_per_sample.csv`, `results_area_summary.csv`, `vis_{elem}/`
- **Metrics:** Sample-level IoU, Dice coefficient, Area fraction standard deviation, and scalar area-based RMSE(%p), MAE(%p), MAPE(%).

**`04_MultiElemOverlay.py`**
Generates publication-quality composite multi-element spatial maps overlaid on faded OM grayscale backgrounds.
- **Output:** `result/test_tversky/all_elems_{tag}/` (6 elements: Al, Si, Mg, Fe, Cu, Sr), `prec_elems_{tag}/` (4 precipitate elements: Mg, Fe, Cu, Sr)
- **Features:** Distinct academic color palette with unified upper-left 2-column legends.

**`05_SpatialMapVisualization_YGB.py`**
Performs pixel-level classification error analysis with ensemble-agreement-weighted opacity.
- **Classification Categories:** Match (True Positive, Green), Miss (False Negative, Yellow), False (False Positive, Red)
- **Output:** `result/test_tversky/pure_mask_{elem}_{tag}/`
- **Ensemble Opacity:** Alpha blending ($0.33 \rightarrow 1.0$) proportionally scaled to member vote counts.

**`06_UncertaintyMap.py`** 
Quantifies pixel-level epistemic uncertainty (standard deviation across ensemble members) and exports filtered uncertainty metrics.
- **Output:** `result/test_tversky/uncertainty_maps/{tag}/{elem}/` (standalone heatmaps & OM overlays), `uncertainty_summary_filtered_{tag}.csv`
- **Filtering:** Excludes background/zero-uncertainty regions ($\sigma \le 10^{-6}$) to compute mean uncertainty for precipitates vs. matrix.

**`07_MicrostructureDescriptorAnalysis.py`**
Statistically validates metallurgical fidelity between Ground Truth and predicted microstructures.
- **Evaluated Descriptors:**
  - **Particle Size Distribution (PSD):** Connected-component area distributions for precipitate phases.
  - **Cross-Element Nearest-Neighbor Distance (Cross-NND):** Spatial distances between all 6 pairwise combinations of precipitate elements (Mg, Fe, Cu, Sr).
- **Metrics:** Kolmogorov-Smirnov (KS) test ($p$-value, statistic) and Wasserstein Distance.
- **Output:** `result/test_tversky/metallurgical_descriptors_{tag}/` (pooled CSVs, summary tables, and GT vs. Pred histogram plots).

**`07_NewSampleInference.py`**
Executes end-to-end inference on a single new sample without Ground Truth masks.
- **Input:** `data/pred_data/OM_hv/{NEW_BASE_NAME}.png`, `data/pred_data/MAP_hv/{NEW_BASE_NAME}.png`
- **Output:** `data/pred_data/result/new_sample_inference/`
  - Raw Area Ratio & 100% Normalized Area Ratio CSV (`mean ± std`)
  - Text summary report (`_summary_report.txt`)
  - 6-element (`_6elems_last_Pred.png`) and 4-precipitate (`_prec4_last_Pred.png`) overlay images
- **Configuration:** Set target sample name via `NEW_BASE_NAME` at the top of the script.
---

## Notes

- `OMtoHV/01_FeatureExtraction.py` uses relative paths (`base_dir = './'`); run it from inside the `OMtoHV/` folder.
- EDS filename suffixes: Mg=`01`, Al=`02`, Si=`03`, Ti=`04`, Mn=`05`, Fe=`06`, Cu=`07`, Zn=`08`, Sr=`09`
- Area fractions: Al uses the full image area as denominator; all other elements use the MAP validity region.
- `splits.json` is generated automatically during training and is shared across all evaluation and visualization scripts.
- Image paths containing non-ASCII characters are handled by the `imread_korean()` utility function.
