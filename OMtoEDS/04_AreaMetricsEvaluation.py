"""
Area Metrics Evaluation Script for EDS Spatial Mapping
- Evaluates the quantitative performance of the predicted EDS area fractions.
- Computes evaluation metrics: RMSE (%p), MAE (%p), and R-squared (R²) for 
  both 'best' and 'last' model checkpoints based on 'results_per_sample.csv'.
- Outputs the summarized metrics to the console and exports them as individual CSV files.
"""

import os
import numpy as np
import pandas as pd

def compute_area_metrics_for_csv(pred_series, gt_series):
    """
    Computes statistical error metrics for the predicted percentage data.
    
    Metrics:
    - RMSE (Root Mean Square Error): Calculated in percentage points (%p).
    - MAE (Mean Absolute Error): Calculated in percentage points (%p).
    - R² (Coefficient of Determination): Statistical measure of fit.

    Args:
        pred_series (pd.Series): Predicted target values.
        gt_series (pd.Series): Ground truth values.

    Returns:
        Tuple[float, float, float]: Returns RMSE, MAE, and R² values. 
                                    Returns NaNs if no valid data exists.
    """
    p = np.array(pred_series, dtype=float)
    g = np.array(gt_series, dtype=float)

    # Mask out missing values (NaNs) to ensure accurate metric calculation
    valid_mask = ~np.isnan(p) & ~np.isnan(g)
    if not valid_mask.any():
        return float('nan'), float('nan'), float('nan')
        
    p, g = p[valid_mask], g[valid_mask]

    # Calculate RMSE (Root Mean Square Error)
    rmse = float(np.sqrt(np.mean((p - g) ** 2)))
    
    # Calculate MAE (Mean Absolute Error)
    mae  = float(np.mean(np.abs(p - g)))

    # Calculate R² (Coefficient of Determination)
    # Added 1e-8 to the denominator to prevent ZeroDivisionError
    ss_res = np.sum((g - p) ** 2)
    ss_tot = np.sum((g - g.mean()) ** 2)
    r2 = float(1 - ss_res / (ss_tot + 1e-8))

    return rmse, mae, r2

def main():
    # 1. Configuration (Paths to the existing CSV results)
    RESULT_DIR = r'.\result'
    TEST_DIR   = os.path.join(RESULT_DIR, 'test_tversky')
    CSV_PATH   = os.path.join(TEST_DIR, "results_per_sample.csv")

    TARGET_ELEMS = ["Mg", "Al", "Si", "Fe", "Cu", "Sr"]

    # 2. Load the original experimental results CSV
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"[ERROR] CSV file not found. Please verify the path: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH, encoding='utf-8-sig')
    print(f"[INFO] Successfully loaded dataset: {len(df)} rows collected.")

    # 3. Iterative evaluation loop for different model checkpoints ('best', 'last')
    model_tags = ["best", "last"]
    
    for tag in model_tags:
        # Filter dataframe by model tag
        df_model = df[df['MODEL'] == tag].copy()
        print(f"\n[INFO] Filtering completed for MODEL == '{tag}': Commencing evaluation for {len(df_model)} samples.")
        
        metrics_results = []
        for elem in TARGET_ELEMS:
            pred_col = f"{elem}_pred(%)"
            gt_col   = f"{elem}_gt(%)"

            # Verify if the target columns exist in the filtered dataframe
            if pred_col in df_model.columns and gt_col in df_model.columns:
                pred_data = pd.to_numeric(df_model[pred_col], errors='coerce')
                gt_data   = pd.to_numeric(df_model[gt_col], errors='coerce')

                rmse, mae, r2 = compute_area_metrics_for_csv(pred_data, gt_data)
                
                metrics_results.append({
                    "Element": elem,
                    "RMSE(%p)": round(rmse, 4),
                    "MAE(%p)": round(mae, 4),
                    "R2": round(r2, 4)
                })
            else:
                # Append NaNs if element columns are missing
                metrics_results.append({
                    "Element": elem,
                    "RMSE(%p)": np.nan,
                    "MAE(%p)": np.nan,
                    "R2": np.nan
                })

        # Generate and display the evaluation metrics dataframe
        df_metrics = pd.DataFrame(metrics_results)
        print(f"\n[EVAL] Elemental Area Error Metrics Summary for [{tag.upper()} MODEL]")
        print("-" * 55)
        print(df_metrics.to_string(index=False))
        print("-" * 55)

        # 4. Export the quantitative results to individual CSV files
        output_csv_path = os.path.join(TEST_DIR, f"results_metrics_{tag}.csv")
        df_metrics.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        print(f"[SAVE] Evaluation metrics exported successfully: {output_csv_path}")

    print("\n[SUCCESS] Statistical evaluation and CSV exportation for all models ('best' & 'last') are complete.")

if __name__ == '__main__':
    main()