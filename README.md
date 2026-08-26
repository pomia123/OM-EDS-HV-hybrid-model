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
├── OMtoEDS/
      ├── 01_FeatureExtraction.py
      ├── 02_DataFiltering.ipynb
      ├── 03_FeatureSelectionCorr.ipynb
      ├── 04_ModelComparison.ipynb
      ├── 05_ConformalPrediction.ipynb
      └── 06_FeatureAnalysis.ipynb
      ├── data/
           └── figure/
           ├── OM_hv/
           ├── a_hv.csv
           ├── b_hv_with_features.csv
           ├── c_cleaned_hv_with_features.csv
           ├── d_hv_with_corr_features.csv
           ├── e_model_comparison_results.csv
           └── f_ensemble_conformal_results.csv
```

---

## Pipeline 1: OMtoHV

Extracts microstructure features from OM images and predicts Vickers hardness (HV) using machine learning.

### Execution Order

```
      01_FeatureExtraction.py
    → 02_DataFiltering.ipynb
    → 03_FeatureSelectionCorr.ipynb
    → 04_ModelComparison.ipynb
    → 05_ConformalPrediction.ipynb
    → 06_FeatureAnalysis.ipynb
```

### Script Descriptions

**`01_FeatureExtraction.py`**  
Extracts microstructure features from OM images in parallel.

- Input: `data/OM_hv/`, `data/a_hv.csv`
- Output: `data/b_hv_with_features.csv`
- Features: secondary phase, eutectic structure, dendrite orientation, DAS, GLCM texture, LBP

**`02_DataFiltering.ipynb`**  
Removes statistical outliers via Z-score analysis (threshold: 3.0).

- Input: `data/b_hv_with_features.csv`
- Output: `data/c_cleaned_hv_with_features.csv`
- 529 samples → 449 samples (80 removed)

**`03_FeatureSelectionCorr.ipynb`**  
Reduces multicollinearity by detecting highly correlated feature pairs ($|r| \ge 0.95$) and dropping the feature with lower correlation to HV.
- **Input:** `data/c_cleaned_hv_with_features.csv`
- **Output:** `data/d_hv_with_corr_features.csv`
- **Feature filtering:** 48 features → 42 features (6 collinear features removed)

**`04_ModelComparison.ipynb`**  
Benchmarks 6 regression models with 5-fold cross-validation and performs Wilcoxon signed-rank tests for statistical validation.
- **Models:** Ridge, Lasso, SVR, RandomForest, GradientBoosting, XGBoost
- **Metrics:** $R^2$, RMSE, MAE
- **Output:** `data/e_model_comparison_results.csv`, `data/figure/` (Actual vs. Predicted plots)

|  Model  |   R²  | RMSE  |  MAE  |
|---------|-------|-------|-------|
|    GB   | 0.873 | 1.004 | 0.762 |
| XGBoost | 0.851 | 1.089 | 0.803 |
|    RF   | 0.837 | 1.138 | 0.842 |

**`05_ConformalPrediction.ipynb`**  
Applies Cross-Validation Conformal Prediction to quantify prediction uncertainty (95% coverage interval) and saves the finalized deployment model.
- **Input:** `data/d_hv_with_corr_features.csv`
- **Output:** `data/f_ensemble_conformal_results.csv`, `data/figure_conformal/`, `data/figure_conformal/GradientBoosting_HV_prediction_model.pkl`

**`06_FeatureAnalysis.ipynb`**  
Interprets feature importance for the top Gradient Boosting model using SHAP TreeExplainer.
- **Input:** `data/d_hv_with_corr_features.csv`
- **Output:** `data/figure/gb_shap_summary_plot_top5.png`, `data/figure/gb_shap_bar_plot_top5.png`

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

- EDS filename suffixes: Mg=`01`, Al=`02`, Si=`03`, Ti=`04`, Mn=`05`, Fe=`06`, Cu=`07`, Zn=`08`, Sr=`09`
- Area fractions: Al uses the full image area as denominator; all other elements use the MAP validity region.
- `splits.json` is generated automatically during training and is shared across all evaluation and visualization scripts.
- Image paths containing non-ASCII characters are handled by the `imread_korean()` utility function.
