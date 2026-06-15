# Hybrid AI Framework for Microstructure-Based Composition Reconstruction and Hardness Prediction of Al–Si Die-Casting Alloys

Two independent analysis pipelines based on Optical Microscopy (OM) images:

- **OMtoHV**: Microstructure feature extraction → ML-based Vickers hardness (HV) prediction
- **OMtoEDS**: GAN-based deep learning → EDS elemental spatial map prediction

---

## Project Structure

```
project/
├── OMtoEDS/
│   ├── a_OMtoEDS_pix2pix.py
│   ├── b_SpatialMapEvaluation_Separate.py
│   ├── c_SpatialMapEvaluation_Combined.py
│   ├── d_AreaMetricsEvaluation.py
│   ├── e_SpatialMapVisualization_YGB.py
│   ├── f_NewSampleInference.py
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
      ├── a_FeatureExtraction.py
      ├── b_DataFiltering.ipynb
      ├── c_ModelComparison.ipynb
      └── d_FeatureAnalysis.ipynb
      ├── data/
           └── figure/
           ├── OM_hv/
           ├── a_hv.csv
           ├── b_hv_with_features.csv
           ├── c_cleaned_hv_with_features.csv
           └── d_model_comparison_results.csv
```

---

## Pipeline 1: OMtoHV

Extracts microstructure features from OM images and predicts Vickers hardness (HV) using machine learning.

### Execution Order

```
a_FeatureExtraction.py
    → b_DataFiltering.ipynb
    → c_ModelComparison.ipynb
    → d_FeatureAnalysis.ipynb
```

### Script Descriptions

**`a_FeatureExtraction.py`**  
Extracts microstructure features from OM images in parallel.

- Input: `data/OM_hv/`, `data/a_hv.csv`
- Output: `data/b_hv_with_features.csv`
- Features: secondary phase, eutectic structure, dendrite orientation, DAS, GLCM texture, LBP

**`b_DataFiltering.ipynb`**  
Removes statistical outliers via Z-score analysis (threshold: 3.0).

- Input: `data/b_hv_with_features.csv`
- Output: `data/c_cleaned_hv_with_features.csv`
- 529 samples → 448 samples (81 removed)

**`c_ModelComparison.ipynb`**  
Benchmarks 6 regression models with 5-fold cross-validation.

- Models: Ridge, Lasso, SVR, RandomForest, GradientBoosting, XGBoost
- Metrics: R², RMSE, MAE
- Output: `data/d_model_comparison_results.csv`, `data/figure/`

|  Model  |   R²  | RMSE  |  MAE  |
|---------|-------|-------|-------|
|    GB   | 0.890 | 0.927 | 0.736 |
| XGBoost | 0.868 | 1.012 | 0.777 |
|    RF   | 0.859 | 1.049 | 0.810 |

**`d_FeatureAnalysis.ipynb`**  
Analyzes the top 5 most important features using GradientBoosting + SHAP.

- Output: `data/figure/gb_shap_summary_plot_top5.png`, `gb_shap_bar_plot_top5.png`

---

## Pipeline 2: OMtoEDS

Predicts element-specific binary EDS spatial maps from OM images using a Pix2Pix GAN with a ResNet-34 encoder, CBAM attention, and Tversky loss.

Target elements: **Mg, Al, Si, Cu, Fe, Sr**

### Execution Order

```
a_OMtoEDS_pix2pix.py                    ← Train models
    → b_SpatialMapEvaluation_Separate.py ← Per-element evaluation
    → c_SpatialMapEvaluation_Combined.py ← Combined evaluation
    → d_AreaMetricsEvaluation.py         ← Area fraction metrics
    → e_SpatialMapVisualization_YGB.py   ← Overlay visualization
    → f_NewSampleInference.py            ← New sample inference
```

### Script Descriptions

**`a_OMtoEDS_pix2pix.py`**  
Trains one GAN model per element.

- Input: `data/OM/`, `data/EDS/`, `data/MASK/`, `data/MAP/`
- Output: `result/tversky/models_tversky/best_model_{elem}.pth`, `last_model_{elem}.pth`, `splits.json`
- Architecture: U-Net (ResNet-34 encoder) + CBAM + Tversky/Focal Loss
- Hyperparameters: 512×512 crop, batch=32, epochs=1000

**`b_SpatialMapEvaluation_Separate.py`**  
Evaluates best/last checkpoints per element on the test split.

- Metrics: IoU, Dice, RMSE(%p), MAE(%p), MAPE(%), R²
- Output: `result/test_tversky/results_per_sample.csv`

**`c_SpatialMapEvaluation_Combined.py`**  
Aggregates and compares best/last model results across all elements.

**`d_AreaMetricsEvaluation.py`**  
Computes accuracy of predicted elemental area fractions.

- Input: `result/test_tversky/results_per_sample.csv`
- Output: `results_metrics_best.csv`, `results_metrics_last.csv`
- Metrics: RMSE(%p), MAE(%p), R²

**`e_SpatialMapVisualization_YGB.py`**  
Overlays predictions on OM backgrounds for qualitative evaluation.

- Match (TP) → Green, Miss (FN) → Yellow, False (FP) → Red
- Output: `result/test_tversky/pure_mask_{elem}_{tag}/`

**`f_NewSampleInference.py`**  
Runs inference on a single new sample without ground truth.

- Input: `pred_data/OM_hv/`, `pred_data/MAP_hv/`
- Output: area fraction CSV + 6-element and 4-element overlay images
- Set the target filename via `NEW_BASE_NAME` at the top of the script

---

## Data

Google Drive: [Download](https://drive.google.com/drive/folders/1xXJMeItTgxSiJEPIyLS20qiyQgB2hNEb?usp=sharing)

Place the downloaded folders inside `data/`.

---

## Notes

- EDS filename suffixes: Mg=`01`, Al=`02`, Si=`03`, Ti=`04`, Mn=`05`, Fe=`06`, Cu=`07`, Zn=`08`, Sr=`09`
- Area fractions: Al uses the full image area as denominator; all other elements use the MAP validity region.
- `splits.json` is generated automatically during training and is shared across all evaluation and visualization scripts.
- Image paths containing non-ASCII characters are handled by the `imread_korean()` utility function.
