# -*- coding: utf-8 -*-
"""
EDS Area Metrics Evaluation Script (Best vs Last)
- results_per_sample.csv 파일을 로드하여 MODEL 컬럼의 'best'와 'last' 데이터 각각에 대해 성능 지표를 계산합니다.
- 평가지표: RMSE (%p), MAE (%p), R² (결정계수)
- 결과를 콘솔에 표 형태로 통합 출력하고, 각각을 별도의 CSV 데이터프레임으로 내보냅니다.
"""

import os
import numpy as np
import pandas as pd

def compute_area_metrics_for_csv(pred_series, gt_series):
    """
    % 단위의 데이터를 입력받아 오차 지표를 계산합니다.
    - RMSE, MAE : 단위는 %p (Percentage Points)
    - R²        : 결정계수
    """
    p = np.array(pred_series, dtype=float)
    g = np.array(gt_series, dtype=float)

    # 결측치(NaN) 제거 마스킹
    valid_mask = ~np.isnan(p) & ~np.isnan(g)
    if not valid_mask.any():
        return float('nan'), float('nan'), float('nan')
        
    p, g = p[valid_mask], g[valid_mask]

    # RMSE 계산
    rmse = float(np.sqrt(np.mean((p - g) ** 2)))
    
    # MAE 계산
    mae  = float(np.mean(np.abs(p - g)))

    # R² (결정계수) 계산
    ss_res = np.sum((g - p) ** 2)
    ss_tot = np.sum((g - g.mean()) ** 2)
    r2 = float(1 - ss_res / (ss_tot + 1e-8))

    return rmse, mae, r2

def main():
    # 1. 설정 (기존 CSV 파일이 저장된 경로)
    RESULT_DIR = r'.\result'
    TEST_DIR   = os.path.join(RESULT_DIR, 'test_tversky')
    CSV_PATH   = os.path.join(TEST_DIR, "results_per_sample.csv")

    TARGET_ELEMS = ["Mg", "Al", "Si", "Cu", "Fe", "Sr"]

    # 2. 원본 결과 CSV 데이터 로드
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV 파일이 존재하지 않습니다. 경로를 확인하세요: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH, encoding='utf-8-sig')
    print(f"📊 오리지널 데이터 로드 완료: 총 {len(df)}개 행 수집됨.")

    # 3. 모델 태그(best, last)별 통합 평가 루프
    model_tags = ["best", "last"]
    
    for tag in model_tags:
        df_model = df[df['MODEL'] == tag].copy()
        print(f"\n🎯 [MODEL == '{tag}'] 필터링 완료: 총 {len(df_model)}개 샘플 연산 시작.")
        
        metrics_results = []
        for elem in TARGET_ELEMS:
            pred_col = f"{elem}_pred(%)"
            gt_col   = f"{elem}_gt(%)"

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
                metrics_results.append({
                    "Element": elem,
                    "RMSE(%p)": np.nan,
                    "MAE(%p)": np.nan,
                    "R2": np.nan
                })

        # 결과 데이터프레임 생성 및 출력
        df_metrics = pd.DataFrame(metrics_results)
        print(f"📈 [{tag.upper()} MODEL] 원소별 면적 오차 성능 지표 요약")
        print("-" * 55)
        print(df_metrics.to_string(index=False))
        print("-" * 55)

        # 4. 성능 지표 결과를 CSV 파일로 개별 저장
        output_csv_path = os.path.join(TEST_DIR, f"results_metrics_{tag}.csv")
        df_metrics.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        print(f"💾 지표 데이터프레임 저장 완료: {output_csv_path}")

    print("\n✨ 모든 모델(best & last)의 요약 지표 연산 및 CSV 저장이 성공적으로 끝났습니다.")

if __name__ == '__main__':
    main()