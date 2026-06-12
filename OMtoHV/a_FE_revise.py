import cv2
import numpy as np
import pandas as pd
import os
import warnings
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from pathlib import Path

from skimage import measure, morphology, feature
from skimage.measure import regionprops
from skimage.filters import threshold_sauvola
from skimage.morphology import disk, closing, remove_small_objects, skeletonize
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from scipy import ndimage as ndi
from scipy.stats import circmean, circstd

warnings.filterwarnings('ignore')


def analyze_single_image(row_data, image_dir, vis_dir, pixel_size, save_vis=False):
    file_name = row_data['FILE_NAME']
    img_path = os.path.join(image_dir, file_name)

    try:
        extractor = MicrostructureFeatureExtractor(img_path, pixel_size=pixel_size)
        features = extractor.extract_all_features()

        if save_vis:
            base = os.path.splitext(file_name)[0]
            extractor.visualize_phase_features(os.path.join(vis_dir, f"{base}_01_phase.png"))
            extractor.visualize_eutectic_features(os.path.join(vis_dir, f"{base}_02_eutectic.png"))
            extractor.visualize_orientation_das_features(os.path.join(vis_dir, f"{base}_03_orientation_das.png"))
            extractor.visualize_texture_features(os.path.join(vis_dir, f"{base}_04_texture.png"))
            print(f"\n[INFO] 샘플 시각화 저장 완료: {vis_dir}")

        return features
    except Exception as e:
        return {"error": f"{file_name}: {str(e)}"}


class MicrostructureFeatureExtractor:
    """OM 이미지에서 미세조직 특징을 추출하는 클래스"""

    def __init__(self, image_path, pixel_size=1.0):
        self.image_path = image_path
        self.pixel_size = pixel_size  # µm/px

        arr = np.fromfile(str(image_path), dtype=np.uint8)
        self.image = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if self.image is None:
            raise ValueError(f"이미지를 불러올 수 없습니다: {image_path}")

        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.clahe_rgb = self._apply_clahe(self.image)
        self.clahe_gray = cv2.cvtColor(self.clahe_rgb, cv2.COLOR_RGB2GRAY)

        self.features = {}
        self.orientation_mask = None
        self.orientation_map = None

        # 시각화용 중간 결과 캐싱
        self._binary_phase = None
        self._labeled_phase = None
        self._props_phase = None
        self._eut_mask = None
        self._skel_mask = None
        self._dist_px = None
        self._lbp = None
        self._glcm = None

    def _apply_clahe(self, bgr_img, clahe_clip=1.2):
        hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
        hsv[..., 2] = clahe.apply(hsv[..., 2])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    def build_orientation_map(self, input_gray=None):
        target_img = input_gray if input_gray is not None else self.gray
        t = threshold_sauvola(target_img, window_size=101, k=0.1)
        sauvola = target_img < t
        binary = closing(sauvola, disk(6))
        binary = remove_small_objects(binary, min_size=200)
        final_mask = ~binary
        map_img = final_mask.astype(np.float64)
        return final_mask, map_img

    def extract_all_features(self):
        self.features.update(self.extract_phase_features())
        self.features.update(self.extract_orientation_features())
        self.features.update(self.estimate_eutectic_spacing())
        self.features.update(self.extract_texture_features())
        self.features['auto_correlation_length_h'] = self._calc_autocorr_profile(axis='h')
        self.features['auto_correlation_length_v'] = self._calc_autocorr_profile(axis='v')
        self.features['auto_correlation_length'] = np.mean([
            self.features['auto_correlation_length_h'],
            self.features['auto_correlation_length_v']
        ])
        return self.features

    # -------------------------------------------------------------------------
    # 1. 2차상 / 기공 특징
    # -------------------------------------------------------------------------
    def extract_phase_features(self):
        features = {}
        _, binary = cv2.threshold(self.gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary_phase = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

        self._binary_phase = binary_phase  # 캐싱

        labeled_phase = measure.label(binary_phase, connectivity=2)
        props_phase = regionprops(labeled_phase)
        self._labeled_phase = labeled_phase
        self._props_phase = props_phase

        features['phase_area_fraction'] = np.sum(binary_phase > 0) / binary_phase.size

        H, W = self.gray.shape
        # FIX: 단위 수정 — pixel_size(µm/px)로 전체 면적(µm²) 계산 후 mm²로 변환
        total_area_mm2 = (H * W * self.pixel_size ** 2) / 1e6

        if len(props_phase) > 0:
            areas_phase = [prop.area * (self.pixel_size ** 2) for prop in props_phase]
            # FIX: 개수 / mm² (올바른 밀도 단위)
            features['phase_count_density'] = len(props_phase) / total_area_mm2
            features['phase_mean_area'] = np.mean(areas_phase)
            features['phase_std_area'] = np.std(areas_phase)
            features['phase_max_area'] = np.max(areas_phase)
            features['phase_mean_eccentricity'] = np.mean([prop.eccentricity for prop in props_phase])
            features['phase_euler_number'] = measure.euler_number(binary_phase, connectivity=2)
            features['phase_area_cv'] = (np.std(areas_phase) / np.mean(areas_phase)
                                         if np.mean(areas_phase) > 0 else 0)
        else:
            for key in ['phase_count_density', 'phase_mean_area', 'phase_std_area',
                        'phase_max_area', 'phase_mean_eccentricity', 'phase_euler_number', 'phase_area_cv']:
                features[key] = 0.0
        return features

    # -------------------------------------------------------------------------
    # 2. 공정조직 특징
    # -------------------------------------------------------------------------
    def estimate_eutectic_spacing(self, target_dark_rgb=[27, 34, 52], otsu_weight=1.1, dark_margin=40):
        features = {}
        final_rgb = self.clahe_rgb
        otsu_thr, _ = cv2.threshold(self.clahe_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        adjusted_thr = otsu_thr * otsu_weight
        _, binary = cv2.threshold(self.clahe_gray, adjusted_thr, 255, cv2.THRESH_BINARY_INV)

        lower_dark = np.array([max(0, c - dark_margin) for c in target_dark_rgb])
        upper_dark = np.array([min(255, c + dark_margin) for c in target_dark_rgb])
        dark_region_mask = cv2.inRange(final_rgb, lower_dark, upper_dark)

        eut_mask_uint8 = cv2.bitwise_and(binary, cv2.bitwise_not(dark_region_mask))
        self._eut_mask = eut_mask_uint8 == 255

        dist_px = ndi.distance_transform_edt(self._eut_mask)
        self._dist_px = dist_px
        width_px_map = dist_px * 2.0
        self._skel_mask = skeletonize(self._eut_mask)
        samples = width_px_map[self._skel_mask > 0]
        samples = samples[samples > 0]

        if samples.size > 0:
            fraction = (np.sum(self._eut_mask) / self._eut_mask.size) * 100
            mean_thickness_px = np.mean(samples)

            features['eutectic_fraction'] = round(fraction, 3)
            # FIX: spacing → lamella_thickness (실제 측정량은 두께임)
            features['eutectic_lamella_thickness_px'] = float(mean_thickness_px)
            features['eutectic_lamella_thickness_px'] = float(mean_thickness_px * self.pixel_size)

            total_skel_length = np.sum(self._skel_mask)
            features['skel_length_total'] = int(total_skel_length)
            features['skel_complexity'] = float(total_skel_length / np.sum(self._eut_mask))
        else:
            features['eutectic_fraction'] = 0.0
            features['eutectic_lamella_thickness_px'] = 0.0
            features['eutectic_lamella_thickness_px'] = 0.0
            features['skel_length_total'] = 0.0
            features['skel_complexity'] = 0.0

        return features

    # -------------------------------------------------------------------------
    # 3. 방향성 및 DAS
    # -------------------------------------------------------------------------
    def extract_orientation_features(self):
        features = {}
        final_mask, map_img = self.build_orientation_map(input_gray=self.clahe_gray)
        self.orientation_mask = final_mask
        self.orientation_map = map_img

        vis_map_img = np.array(self.orientation_map, dtype=np.float64)
        sobelx = cv2.Sobel(vis_map_img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(vis_map_img, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
        orientation = np.arctan2(sobely, sobelx)

        valid_mask = magnitude > np.percentile(magnitude, 50)

        if np.sum(valid_mask) > 0:
            angles = (np.rad2deg(orientation[valid_mask]) + 90) % 180
            angles_rad = np.deg2rad(angles * 2)
            raw_mean = float(np.rad2deg(circmean(angles_rad)) / 2)
            # FIX: 음수 각도 보정
            features["orientation_mean_deg"] = float(raw_mean % 180)
            features["orientation_std_deg"] = float(np.rad2deg(circstd(angles_rad)) / 2)
            hist, _ = np.histogram(angles, bins=36, range=(0, 180))
            hist_norm = hist / (np.sum(hist) + 1e-7)
            features["orientation_entropy"] = -np.sum(
                hist_norm[hist_norm > 0] * np.log2(hist_norm[hist_norm > 0]))
            features["orientation_dominant_strength"] = float(np.max(hist_norm))

            # 시각화용 캐싱
            self._orientation_angles = angles
            self._orientation_magnitude = magnitude
            self._sobelx = sobelx
            self._sobely = sobely
        else:
            features["orientation_mean_deg"] = 0.0
            features["orientation_std_deg"] = 0.0
            features["orientation_entropy"] = 0.0
            features["orientation_dominant_strength"] = 0.0

        # --- DAS ---
        das_binary = final_mask

        def _sample_one_angle(binary_img, angle_deg, n_lines=40):
            H, W = binary_img.shape
            angle_rad = np.deg2rad(angle_deg)
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            diag = np.sqrt(H ** 2 + W ** 2)
            offsets = np.linspace(-diag / 2, diag / 2, n_lines + 2)[1:-1]
            cx, cy = W / 2, H / 2
            das_vals = []
            for offset in offsets:
                px_c, py_c = cx - sin_a * offset, cy + cos_a * offset
                pts = []
                for t in np.linspace(-diag, diag, 4000):
                    x, y = int(round(px_c + cos_a * t)), int(round(py_c + sin_a * t))
                    if 0 <= x < W and 0 <= y < H:
                        pts.append((x, y))
                if len(pts) < 10:
                    continue
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                profile = binary_img[ys, xs].astype(int)
                transitions = np.abs(np.diff(profile))

                # FIX: 경계 라인 문제 — 시작/끝이 1인 경우 경계에서 잘린 것으로 보고 제외
                if profile[0] == 1 or profile[-1] == 1:
                    continue

                n_trans = int(transitions.sum())
                if n_trans >= 2:
                    # FIX: 전체 길이 대신 유효 구간 길이 사용 (첫 전환 ~ 마지막 전환)
                    trans_indices = np.where(transitions > 0)[0]
                    valid_length = trans_indices[-1] - trans_indices[0] + 1
                    das_vals.append(valid_length / (n_trans / 2))
            return das_vals

        angles_das = [0, 45, 90, 135]
        all_das, per_angle = [], {}
        for ang in angles_das:
            vals = _sample_one_angle(das_binary, ang)
            if len(vals) > 0:
                per_angle[ang] = float(np.mean(vals))
                all_das.extend(vals)

        if len(all_das) >= 4:
            arr = np.array(all_das)
            q25, q75 = np.percentile(arr, 25), np.percentile(arr, 75)
            iqr = q75 - q25
            mask = (arr >= q25 - 1.5 * iqr) & (arr <= q75 + 1.5 * iqr)
            filt = arr[mask]
            filt_px = filt * self.pixel_size

            features["DAS_mean_px"] = float(np.mean(filt_px))
            features["DAS_std_px"] = float(np.std(filt_px))
            features["DAS_median_px"] = float(np.median(filt_px))
            features["DAS_n_lines"] = int(mask.sum())
            features["DAS_0deg_px"] = float(per_angle.get(0, np.nan) * self.pixel_size)
            features["DAS_45deg_px"] = float(per_angle.get(45, np.nan) * self.pixel_size)
            features["DAS_90deg_px"] = float(per_angle.get(90, np.nan) * self.pixel_size)
            features["DAS_135deg_px"] = float(per_angle.get(135, np.nan) * self.pixel_size)

            das_dirs = np.array([per_angle.get(a, np.nan) for a in [0, 45, 90, 135]])
            das_dirs = das_dirs[~np.isnan(das_dirs)]
            features["DAS_anisotropy"] = (float(np.std(das_dirs) / np.mean(das_dirs))
                                          if len(das_dirs) > 1 else 0.0)
        else:
            for k in ["DAS_mean_px", "DAS_std_px", "DAS_median_px", "DAS_n_lines",
                      "DAS_0deg_px", "DAS_45deg_px", "DAS_90deg_px", "DAS_135deg_px", "DAS_anisotropy"]:
                features[k] = 0.0

        return features

    # -------------------------------------------------------------------------
    # 4. 텍스처
    # -------------------------------------------------------------------------
    def extract_texture_features(self):
        features = {}
        features.update(self.calculate_glcm_features())
        features.update(self.calculate_lbp_features())
        features['intensity_mean'] = float(np.mean(self.gray))
        features['intensity_std'] = float(np.std(self.gray))
        features['intensity_skewness'] = self._skewness(self.gray)
        features['intensity_kurtosis'] = self._kurtosis(self.gray)
        return features

    def calculate_glcm_features(self):
        features = {}
        gray_reduced = (self.gray // 4).astype(np.uint8)
        distances = [1, 3, 5]
        angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

        glcm = graycomatrix(gray_reduced, distances=distances, angles=angles,
                            levels=64, symmetric=True, normed=True)
        self._glcm = glcm

        for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']:
            try:
                values = graycoprops(glcm, prop)
                features[f'glcm_{prop}_mean'] = float(np.mean(values))
                features[f'glcm_{prop}_std'] = float(np.std(values))
            except:
                features[f'glcm_{prop}_mean'] = 0.0
                features[f'glcm_{prop}_std'] = 0.0
        return features

    def calculate_lbp_features(self):
        features = {}
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(self.gray, n_points, radius, method='uniform')
        self._lbp = lbp

        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)

        features['lbp_mean'] = float(np.mean(lbp))
        features['lbp_std'] = float(np.std(lbp))
        features['lbp_entropy'] = float(-np.sum(hist[hist > 0] * np.log2(hist[hist > 0])))

        # FIX: uniform pattern = 값이 n_points+1 미만인 픽셀 비율
        # skimage 'uniform' 메서드에서 n_points+2개 bin 중 마지막(n_points+1)이 non-uniform
        features['lbp_uniformity'] = float(np.sum(lbp < n_points + 1) / lbp.size)

        return features

    # -------------------------------------------------------------------------
    # 5. 자기상관 길이 (수평/수직 분리)
    # -------------------------------------------------------------------------
    def _calc_autocorr_profile(self, img=None, axis='h'):
        """axis='h' 수평, 'v' 수직"""
        if img is None:
            img = self.gray
        std_val = np.std(img)
        if std_val == 0:
            return 0.0

        img_norm = (img - np.mean(img)) / std_val
        f_transform = np.fft.fft2(img_norm)
        power_spectrum = np.abs(f_transform) ** 2
        autocorr = np.fft.ifft2(power_spectrum).real
        autocorr = np.fft.fftshift(autocorr)

        center_r, center_c = autocorr.shape[0] // 2, autocorr.shape[1] // 2

        # FIX: 수평/수직 방향 각각 계산
        if axis == 'h':
            profile = autocorr[center_r, center_c:]
        else:
            profile = autocorr[center_r:, center_c]

        profile = profile / (profile[0] + 1e-12)

        try:
            crossing_idx = np.where(profile < 1 / np.e)[0]
            corr_len = crossing_idx[0] * self.pixel_size if len(crossing_idx) > 0 else len(profile) * self.pixel_size
        except:
            corr_len = 0.0

        return float(corr_len)

    # legacy wrapper
    def calculate_autocorrelation_length(self, img=None):
        return np.mean([self._calc_autocorr_profile(img, 'h'), self._calc_autocorr_profile(img, 'v')])

    def _skewness(self, data):
        std = np.std(data)
        return float(np.mean(((data - np.mean(data)) / std) ** 3)) if std > 0 else 0.0

    def _kurtosis(self, data):
        std = np.std(data)
        return float(np.mean(((data - np.mean(data)) / std) ** 4) - 3) if std > 0 else 0.0

    # =========================================================================
    # 시각화 (논문용 300 dpi, white background)
    # =========================================================================

    def _save_img(self, img_data, output_path, cmap=None):
        """이미지 데이터를 제목/축 없이 개별 파일로 저장"""
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['font.size'] = 15
        plt.rcParams['axes.labelsize'] = 15
        plt.rcParams['axes.titlesize'] = 15

        h, w = img_data.shape[:2]
        fig = plt.figure(figsize=(w / 300, h / 300), dpi=300)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(img_data, cmap=cmap)
        ax.axis('off')
        fig.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0,
                    facecolor='white')
        plt.close(fig)
        print(f"[Saved] {output_path}")

    def _save_plot(self, plot_fn, output_path):
        """플롯(그래프) 계열을 개별 파일로 저장. plot_fn(ax)을 받아 그림"""
        fig, ax = plt.subplots(figsize=(4, 3))
        plot_fn(ax)
        fig.savefig(output_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close(fig)
        print(f"[Saved] {output_path}")

    def visualize_phase_features(self, output_path='phase_features.png'):
        """phase_01_original, _02_mask, _03_area_hist 개별 저장"""
        if self._binary_phase is None:
            self.extract_phase_features()

        base = output_path.replace('.png', '')

        # 원본
        self._save_img(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB), f"{base}_01_original.png")

        # Phase binary mask
        self._save_img(self._binary_phase, f"{base}_02_mask.png", cmap='gray')

        # Area histogram
        if self._props_phase:
            areas = [p.area * self.pixel_size ** 2 for p in self._props_phase]
            mean_area = np.mean(areas)
            def plot_area(ax):
                ax.hist(areas, bins=30, color='steelblue', edgecolor='white', linewidth=0.3)
                ax.axvline(mean_area, color='tomato', lw=1.5, label=f'Mean = {mean_area:.1f} px²')
                ax.legend()
                ax.set_xlabel('Area (px²)')
                ax.set_ylabel('Count')
            self._save_plot(plot_area, f"{base}_03_area_hist.png")

    def visualize_eutectic_features(self, output_path='eutectic_features.png'):
        """eutectic_01_original, _02_mask_overlay, _03_thickness_map 개별 저장"""
        if self._eut_mask is None:
            self.estimate_eutectic_spacing()

        base = output_path.replace('.png', '')

        # 원본
        self._save_img(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB), f"{base}_01_original.png")

        # Eutectic mask overlay
        overlay = np.stack([self.gray] * 3, axis=-1)
        if self._eut_mask is not None:
            color_layer = np.zeros_like(overlay)
            color_layer[self._eut_mask] = [0, 180, 60]
            overlay = (overlay * 0.6 + color_layer * 0.4).astype(np.uint8)
        self._save_img(overlay, f"{base}_02_mask_overlay.png")

        # Lamella thickness map
        if self._dist_px is not None:
            thickness_map = self._dist_px * 2 * self.pixel_size
            # colormap 적용 후 uint8로 변환하여 저장
            norm = plt.Normalize(vmin=thickness_map.min(), vmax=thickness_map.max())
            rgba = plt.cm.inferno(norm(thickness_map))
            rgb = (rgba[..., :3] * 255).astype(np.uint8)
            self._save_img(rgb, f"{base}_03_thickness_map.png")

    def visualize_orientation_das_features(self, output_path='orientation_das_features.png'):
        """orientation_01_original, _02_orientation_map, _03_polar_hist, _04_das_bar 개별 저장"""
        if self.orientation_mask is None:
            self.extract_orientation_features()

        base = output_path.replace('.png', '')

        self._save_img(self.orientation_mask.astype(np.uint8) * 255, f"{base}_00_final_mask.png", cmap='gray')

        # 원본
        self._save_img(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB), f"{base}_01_original.png")

        # Orientation color map
        if hasattr(self, '_sobelx'):
            mag = np.sqrt(self._sobelx ** 2 + self._sobely ** 2)
            ori = np.arctan2(self._sobely, self._sobelx)
            ori_norm = ((np.rad2deg(ori) + 90) % 180) / 180.0
            mag_norm = mag / (mag.max() + 1e-8)
            rgb_ori = (plt.cm.hsv(ori_norm)[..., :3] * mag_norm[..., np.newaxis] * 255).astype(np.uint8)
            self._save_img(rgb_ori, f"{base}_02_orientation_map.png")

        # Polar histogram — 0~180° 반원만 표시 (0°=180° 동일하므로 미러링 불필요)
        if hasattr(self, '_orientation_angles'):
            hist, bin_edges = np.histogram(self._orientation_angles, bins=18, range=(0, 180))
            hist_norm = hist / (hist.sum() + 1e-7)
            bin_centers = np.deg2rad((bin_edges[:-1] + bin_edges[1:]) / 2)
            mean_rad = np.deg2rad(self.features.get('orientation_mean_deg', 0))
            fig = plt.figure(figsize=(4, 4))
            ax = fig.add_subplot(111, projection='polar')
            ax.set_thetamin(0)
            ax.set_thetamax(180)
            ax.bar(bin_centers, hist_norm, width=np.pi / 18,
                   color='steelblue', alpha=0.8, edgecolor='none')
            ax.plot([mean_rad, mean_rad], [0, hist_norm.max()], color='tomato', lw=1.5)
            fig.savefig(f"{base}_03_polar_hist.png", dpi=300, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            plt.close(fig)
            print(f"[Saved] {base}_03_polar_hist.png")

        # DAS bar chart
        das_dir_keys = ['DAS_0deg_px', 'DAS_45deg_px', 'DAS_90deg_px', 'DAS_135deg_px']
        das_labels = ['0°', '45°', '90°', '135°']
        das_vals = [self.features.get(k, 0) for k in das_dir_keys]
        mean_das = self.features.get('DAS_mean_px', 0)
        def plot_das(ax):
            ax.bar(das_labels, das_vals, color='steelblue', edgecolor='white', linewidth=0.5, width=0.5)
            ax.axhline(mean_das, color='tomato', lw=1.5, ls='--',
                       label=f'Mean = {mean_das:.1f} px')

            ax.set_yticks([0, 100, 200, 300])
            ax.set_ylim(0, 300)

            ax.legend()
            ax.set_ylabel('DAS (px)')
        self._save_plot(plot_das, f"{base}_04_das_bar.png")

    def visualize_texture_features(self, output_path='texture_features.png'):
        """texture_01_lbp_map, _02_acf_profile, _03_intensity_hist 개별 저장"""
        if self._lbp is None:
            self.extract_texture_features()

        base = output_path.replace('.png', '')

        # LBP map
        if self._lbp is not None:
            norm = plt.Normalize(vmin=self._lbp.min(), vmax=self._lbp.max())
            rgb_lbp = (plt.cm.gray(norm(self._lbp))[..., :3] * 255).astype(np.uint8)
            self._save_img(rgb_lbp, f"{base}_01_lbp_map.png")

        # Autocorrelation profile
        img_norm = (self.gray.astype(float) - np.mean(self.gray)) / (np.std(self.gray) + 1e-8)
        f_t = np.fft.fft2(img_norm)
        autocorr = np.fft.fftshift(np.fft.ifft2(np.abs(f_t) ** 2).real)
        cr, cc = autocorr.shape[0] // 2, autocorr.shape[1] // 2
        prof_h = autocorr[cr, cc:]
        prof_v = autocorr[cr:, cc]
        prof_h = prof_h / (prof_h[0] + 1e-12)
        prof_v = prof_v / (prof_v[0] + 1e-12)
        max_lag = min(200, len(prof_h), len(prof_v))
        lags = np.arange(max_lag) * self.pixel_size
        def plot_acf(ax):
            ax.plot(lags, prof_h[:max_lag], color='steelblue', lw=1.5, label='Horizontal')
            ax.plot(lags, prof_v[:max_lag], color='darkorange', lw=1.5, label='Vertical')
            ax.axhline(1 / np.e, color='gray', lw=1, ls='--', label='1/e')
            ax.legend()
            ax.set_xlabel('Lag (px)')
            ax.set_ylabel('Normalized ACF')
        self._save_plot(plot_acf, f"{base}_02_acf_profile.png")

        # Intensity histogram
        hist_int, bin_edges = np.histogram(self.gray.ravel(), bins=64, range=(0, 256))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        mean_intensity = self.features.get('intensity_mean', 0)
        
        def plot_intensity(ax):
            ax.fill_between(bin_centers, hist_int, color='steelblue', alpha=0.5, step='mid')
            ax.step(bin_centers, hist_int, color='steelblue', lw=1.5)
            ax.axvline(mean_intensity, color='tomato', lw=1.5,
                       label=f'Mean = {mean_intensity:.1f}')
            
            # Y축 틱 포맷을 k(Kilo) 단위로 변경하는 포매터 함수 적용
            ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{int(x*1e-3)}k' if x != 0 else '0'))
            
            ax.legend()
            ax.set_xlabel('Intensity')
            ax.set_ylabel('Count')
            
        self._save_plot(plot_intensity, f"{base}_03_intensity_hist.png")

        # GLCM matrix heatmap (d=1, θ=0°)
        if self._glcm is not None:
            glcm_d1_a0 = self._glcm[:, :, 0, 0]
            norm = plt.Normalize(vmin=0, vmax=glcm_d1_a0.max())
            rgb_glcm = (plt.cm.viridis(norm(glcm_d1_a0))[..., :3] * 255).astype(np.uint8)
            self._save_img(rgb_glcm, f"{base}_04_glcm_matrix.png")

            # GLCM contrast heatmap (distance × angle)
            from skimage.feature import graycoprops
            angle_labels = ['0°', '45°', '90°', '135°']
            dist_labels = ['d=1', 'd=3', 'd=5']
            def plot_glcm_contrast(ax):
                contrast_mat = graycoprops(self._glcm, 'contrast')  # shape (3, 4)
                im = ax.imshow(contrast_mat, cmap='YlOrRd', aspect='auto')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                ax.set_xticks(range(4))
                ax.set_xticklabels(angle_labels)
                ax.set_yticks(range(3))
                ax.set_yticklabels(dist_labels)
                ax.set_xlabel('Angle')
                ax.set_ylabel('Distance')
                for i in range(3):
                    for j in range(4):
                        ax.text(j, i, f'{contrast_mat[i, j]:.2f}',
                                ha='center', va='center', fontsize=8, color='black')
            self._save_plot(plot_glcm_contrast, f"{base}_05_glcm_contrast.png")

    # 기존 단일 시각화 (하위 호환)
    def visualize_segmentation(self, output_path='comprehensive_analysis.png'):
        self.visualize_phase_features(output_path.replace('.png', '_phase.png'))
        self.visualize_eutectic_features(output_path.replace('.png', '_eutectic.png'))
        self.visualize_orientation_das_features(output_path.replace('.png', '_orientation.png'))
        self.visualize_texture_features(output_path.replace('.png', '_texture.png'))


# =============================================================================
# 배치 처리
# =============================================================================

def process_microstructure_parallel():
    base_dir = r'C:\Taehyun\2026\999_temp\OMtoHV'
    image_dir = os.path.join(base_dir, 'data', 'OM_hv')
    dataset_path = os.path.join(base_dir, 'data', 'a_hv.csv')
    output_path = os.path.join(base_dir, 'data', 'b_hv_with_features.csv')
    vis_dir = os.path.join(base_dir, 'data', 'figure')
    os.makedirs(vis_dir, exist_ok=True)

    df = pd.read_csv(dataset_path)

    rows_to_process = df.to_dict('records')
    all_results = [None] * len(df)

    with ProcessPoolExecutor() as executor:
        futures = {}
        first_valid_submitted = False

        for i, row in enumerate(rows_to_process):
            should_save = False
            if not first_valid_submitted:
                should_save = True
                first_valid_submitted = True
            futures[executor.submit(
                analyze_single_image, row, image_dir, vis_dir, 1.0, should_save
            )] = i

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            idx = futures[future]
            try:
                all_results[idx] = future.result()
            except Exception as e:
                all_results[idx] = {"error": str(e)}

    features_df = pd.DataFrame(all_results)
    final_df = pd.concat([df, features_df], axis=1)
    final_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n완료! → {output_path}")


if __name__ == "__main__":
    process_microstructure_parallel()
