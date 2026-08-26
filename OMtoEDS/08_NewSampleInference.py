"""
=============================================================================
Description:
  - Performs inference using a Deep Ensemble of 'last_model' weights
    (N_ENSEMBLE independently trained members per element) on a single
    target sample without requiring Ground Truth (GT) mask files.
  - Evaluation Metrics:
    * Raw Area Ratio (%): Denominator is fixed to 512x512 total image area.
    * Normalized Area Ratio (%): Scales all 6 predicted element area ratios
      so that their total sum equals 100% per ensemble member.
    * Formatted 'mean +/- std' strings are saved directly into CSV and TXT report.
  - Visualization: Generates publication-ready multi-element overlays with
    unified upper-left 2-column legends.
=============================================================================
"""

import os
import csv
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# =============================================================================
# [Configuration] Target Sample & Directory Settings
# =============================================================================
NEW_BASE_NAME = "0La_x500_0241"
# NEW_BASE_NAME = "0La_x500_0420"

OM_DIR   = r'.\data\pred_data\OM_hv'
MAP_DIR  = r'.\data\pred_data\MAP_hv'
RESULT_DIR = r'.\data\pred_data\result'

MODEL_DIR  = os.path.join(r'.\result', 'tversky', 'models_tversky')
OUTPUT_DIR = os.path.join(RESULT_DIR, 'new_sample_inference')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Element sets configuration
ALL_ELEMS   = ["Al", "Si", "Mg", "Fe", "Cu", "Sr"]
PREC_ELEMS  = ["Mg", "Fe", "Cu", "Sr"]   # Precipitation elements
THRESHOLD   = 0.5
N_ENSEMBLE  = 3
CROP_W, CROP_H = 512, 512

# Matplotlib global font settings for publication quality
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial'] + plt.rcParams['font.sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# High-contrast complementary RGB colors for multi-element visualization
ELEM_COLORS_RGB = {
    "Al": (215, 205, 190),   # soft beige-gray
    "Si": (235, 190,  70),   # muted gold
    "Mg": (  0, 200, 255),   # cyan
    "Fe": (150,  70, 255),   # vivid purple
    "Cu": (220,  40,  40),   # crimson red
    "Sr": ( 80, 230,  80),   # lime green
}

# =============================================================================
# Data Loading & Center Crop Utilities
# =============================================================================
def imread_korean(path, mode=cv2.IMREAD_COLOR):
    try:
        n = np.fromfile(path, np.uint8)
        return cv2.imdecode(n, mode)
    except:
        return None

def load_and_center_crop_data(base):
    om_path  = os.path.join(OM_DIR, f"{base}.png")
    map_path = os.path.join(MAP_DIR, f"{base}.png")

    om_raw  = imread_korean(om_path, 1)
    g_map   = imread_korean(map_path, 0)

    if om_raw is None or g_map is None:
        raise FileNotFoundError(f"[Error] Failed to load base files.\nOM: {om_path}\nMAP: {map_path}")

    h, w = g_map.shape
    sy = (h - CROP_H) // 2
    sx = (w - CROP_W) // 2

    om_crop  = cv2.cvtColor(om_raw, cv2.COLOR_BGR2RGB)[sy:sy+CROP_H, sx:sx+CROP_W]
    map_crop = g_map[sy:sy+CROP_H, sx:sx+CROP_W]
    om_raw_crop = om_raw[sy:sy+CROP_H, sx:sx+CROP_W]

    valid_mask_np = (map_crop > 0).astype(np.uint8)

    om_t = torch.from_numpy(om_crop).permute(2, 0, 1).float().unsqueeze(0)
    om_t = (om_t / 127.5) - 1.0
    valid_mask_t = torch.from_numpy(valid_mask_np).float().unsqueeze(0).unsqueeze(0)

    return om_t, valid_mask_t, valid_mask_np, om_raw_crop

# =============================================================================
# Network Architecture (Generator with CBAM)
# =============================================================================
class ChannelAttention(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        mid = max(ch // r, 4)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.mx  = nn.AdaptiveMaxPool2d(1)
        self.fc  = nn.Sequential(
            nn.Conv2d(ch, mid, 1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(mid, ch, 1, bias=False),
        )
        self.sig = nn.Sigmoid()
    def forward(self, x):
        return x * self.sig(self.fc(self.avg(x)) + self.fc(self.mx(x)))

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sig  = nn.Sigmoid()
    def forward(self, x):
        avg = x.mean(1, keepdim=True)
        mx, _ = x.max(1, keepdim=True)
        return x * self.sig(self.conv(torch.cat([avg, mx], 1)))

class CBAM(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.ca = ChannelAttention(ch)
        self.sa = SpatialAttention()
    def forward(self, x):
        return self.sa(self.ca(x))

class ConvBnRelu(nn.Sequential):
    def __init__(self, in_ch, out_ch):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = nn.Sequential(
            ConvBnRelu(in_ch // 2 + skip_ch, out_ch),
            ConvBnRelu(out_ch, out_ch),
        )
        self.cbam = CBAM(out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        return self.cbam(self.conv(torch.cat([x, skip], 1)))

class Generator(nn.Module):
    def __init__(self, out_ch=1):
        super().__init__()
        bb = resnet34(weights=None)
        self.enc0 = nn.Sequential(bb.conv1, bb.bn1, bb.relu)
        self.pool = bb.maxpool
        self.enc1 = bb.layer1
        self.enc2 = bb.layer2
        self.enc3 = bb.layer3
        self.enc4 = bb.layer4
        self.dec4 = DecoderBlock(512, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128,  64,  64)
        self.dec1 = DecoderBlock( 64,  64,  32)
        self.dec0 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 2, stride=2),
            ConvBnRelu(32, 32),
            CBAM(32),
        )
        self.head = nn.Conv2d(32, out_ch, 1)

    def forward(self, x):
        H, W = x.shape[2:]
        e0 = self.enc0(x)
        e1 = self.enc1(self.pool(e0))
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        d = self.dec4(e4, e3)
        d = self.dec3(d,  e2)
        d = self.dec2(d,  e1)
        d = self.dec1(d,  e0)
        d = self.dec0(d)
        out = self.head(d)
        if out.shape[2:] != (H, W):
            out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        return out

# =============================================================================
# Visualization Helpers
# =============================================================================
def create_multi_elem_overlay(om_bgr, bins, valid_mask_np, elem_names, elem_alpha=0.6):
    om_gray  = cv2.cvtColor(om_bgr, cv2.COLOR_BGR2GRAY)
    om_faded = cv2.addWeighted(om_gray, 0.4, np.full_like(om_gray, 255), 0.6, 0)
    canvas   = cv2.cvtColor(om_faded, cv2.COLOR_GRAY2RGB).astype(np.float32)

    for en in elem_names:
        pred = (bins[en] > 0) & (valid_mask_np > 0)
        r, g, b = ELEM_COLORS_RGB[en]
        color = np.array([r, g, b], dtype=np.float32)
        canvas[pred] = (1.0 - elem_alpha) * canvas[pred] + elem_alpha * color

    return canvas.clip(0, 255).astype(np.uint8)

def save_overlay_plot(overlay_img, elem_names, legend_ncol, filename):
    fig, ax = plt.subplots(figsize=(15, 15), dpi=300)
    ax.imshow(overlay_img)
    ax.axis('off')

    custom_handles = [
        Line2D([0], [0], marker='o', color='none',
               markerfacecolor=tuple(c/255 for c in ELEM_COLORS_RGB[en]),
               markersize=25, label=en, alpha=0.9)
        for en in elem_names
    ]
    
    ax.legend(
        handles=custom_handles,
        loc='upper left',
        ncol=legend_ncol,
        fontsize=40,
        frameon=True,
        facecolor='white',
        edgecolor='none',
        handletextpad=0.2,
        columnspacing=0.6,
        labelspacing=0.4,
        borderpad=0.4
    )
    plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

# =============================================================================
# Model Loader
# =============================================================================
def load_model_ensemble(elem_name, device):
    key = elem_name.lower()
    models = []
    for idx in range(N_ENSEMBLE):
        path = os.path.join(MODEL_DIR, f"last_model_{key}_{idx}.pth")
        if not os.path.exists(path):
            continue
        G = Generator(out_ch=1).to(device)
        sd = torch.load(path, map_location=device)
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
        G.load_state_dict(sd)
        G.eval()
        models.append(G)
        print(f"[INFO] Loaded ensemble member {idx} for [{elem_name}]: {os.path.basename(path)}")
    return models

# =============================================================================
# Main Inference Loop
# =============================================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Inference Device: {device}")
    print(f"[INFO] Target Sample ID: {NEW_BASE_NAME}\n")

    # 1. Load Data & Center Crop
    om_t, valid_mask_t, valid_mask_np, om_raw_crop = load_and_center_crop_data(NEW_BASE_NAME)
    total_image_pixels = float(CROP_W * CROP_H) 
    
    if valid_mask_np.sum() == 0:
        raise ValueError("[Error] Valid (white) region not found in the MAP image.")

    # 2. Load Deep Ensemble Weights ('last_model' tag)
    models = {}
    for en in ALL_ELEMS:
        G_list = load_model_ensemble(en, device)
        if G_list:
            models[en] = G_list
        else:
            print(f"[WARN] No 'last_model' ensemble weights found for {en}. Skipping element.")

    if not models:
        print("[Error] No 'last_model' weights found in the directory. Please check file names.")
        return

    # 3. Deep Ensemble Inference
    raw_member_areas = {en: [0.0] * N_ENSEMBLE for en in ALL_ELEMS}
    pred_bins = {}

    for en in ALL_ELEMS:
        if en in models:
            member_probs = []
            for m_idx, G in enumerate(models[en]):
                G.eval()
                with torch.no_grad():
                    logit = G(om_t.to(device))
                    pred_sig = torch.sigmoid(logit).cpu().squeeze().numpy()
                member_probs.append(pred_sig)

                if en == "Al":
                    m_bin = (pred_sig > THRESHOLD).astype(np.uint8)
                else:
                    m_bin = ((pred_sig > THRESHOLD) * valid_mask_np).astype(np.uint8)
                
                raw_member_areas[en][m_idx] = (m_bin.sum() / total_image_pixels) * 100.0

            # Ensemble-mean probability for overlay visualization
            mean_prob = np.mean(np.stack(member_probs, axis=0), axis=0)
            if en == "Al":
                pred_bin = (mean_prob > THRESHOLD).astype(np.uint8)
            else:
                pred_bin = ((mean_prob > THRESHOLD) * valid_mask_np).astype(np.uint8)
        else:
            pred_bin = np.zeros((CROP_H, CROP_W), dtype=np.uint8)

        pred_bins[en] = pred_bin

    # 4. Normalize to 100% per ensemble member
    member_totals = [
        sum(raw_member_areas[en][m_idx] for en in ALL_ELEMS)
        for m_idx in range(N_ENSEMBLE)
    ]

    norm_member_areas = {en: [0.0] * N_ENSEMBLE for en in ALL_ELEMS}
    for en in ALL_ELEMS:
        for m_idx in range(N_ENSEMBLE):
            if member_totals[m_idx] > 0:
                norm_member_areas[en][m_idx] = (raw_member_areas[en][m_idx] / member_totals[m_idx]) * 100.0
            else:
                norm_member_areas[en][m_idx] = 0.0

    # 5. Format & Print Console Output
    header_line = f"{'='*78}"
    title_line  = f"  Element  |  Raw Area (%): mean +/- std   |  Normalized (100%): mean +/- std"
    
    print(f"\n{header_line}")
    print(title_line)
    print(header_line)

    txt_lines = [header_line, title_line, header_line]
    area_results = []

    for en in ALL_ELEMS:
        raw_m = float(np.mean(raw_member_areas[en]))
        raw_s = float(np.std(raw_member_areas[en]))
        norm_m = float(np.mean(norm_member_areas[en]))
        norm_s = float(np.std(norm_member_areas[en]))

        raw_str  = f"{raw_m:7.4f} +/- {raw_s:.4f} %"
        norm_str = f"{norm_m:7.4f} +/- {norm_s:.4f} %"

        row_print = f"   {en:<7} |     {raw_str}     |       {norm_str}"
        print(row_print)
        txt_lines.append(row_print)

        # Dictionary for CSV saving (Formatted strings included)
        row = {
            "Element": en,
            "Raw_Area_mean_std(%)": raw_str,
            "Norm_Area_mean_std(%)": norm_str,
            "Raw_Area_mean(%)": round(raw_m, 4),
            "Raw_Area_std(%p)": round(raw_s, 4),
            "Norm_Area_mean(%)": round(norm_m, 4),
            "Norm_Area_std(%p)": round(norm_s, 4),
        }
        for i in range(N_ENSEMBLE):
            row[f"Raw_Area_{i}(%)"] = round(raw_member_areas[en][i], 4)
            row[f"Norm_Area_{i}(%)"] = round(norm_member_areas[en][i], 4)
        area_results.append(row)

    print(header_line)
    txt_lines.append(header_line)

    # 6. Export Quantitative Metrics to CSV & TXT
    csv_out_path = os.path.join(OUTPUT_DIR, f"{NEW_BASE_NAME}_last_area_metrics_normalized_100.csv")
    csv_fieldnames = [
        "Element",
        "Raw_Area_mean_std(%)",
        "Norm_Area_mean_std(%)",
        "Raw_Area_mean(%)", "Raw_Area_std(%p)", 
        "Norm_Area_mean(%)", "Norm_Area_std(%p)"
    ] + [f"Raw_Area_{i}(%)" for i in range(N_ENSEMBLE)] + [f"Norm_Area_{i}(%)" for i in range(N_ENSEMBLE)]

    with open(csv_out_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
        writer.writeheader()
        writer.writerows(area_results)
    print(f"[INFO] Metrics CSV saved to: {csv_out_path}")

    # Save formatted print string as TXT report
    txt_out_path = os.path.join(OUTPUT_DIR, f"{NEW_BASE_NAME}_summary_report.txt")
    with open(txt_out_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(txt_lines) + "\n")
    print(f"[INFO] Formatted Report TXT saved to: {txt_out_path}")

    # 7. Generate Multi-Element Overlay Visualizations (ncol=2 applied)
    active_all = [en for en in ALL_ELEMS if en in models]
    if active_all:
        pred_overlay_all = create_multi_elem_overlay(om_raw_crop, pred_bins, valid_mask_np, active_all)
        save_overlay_plot(pred_overlay_all, active_all, 2, os.path.join(OUTPUT_DIR, f"{NEW_BASE_NAME}_6elems_last_Pred.png"))

    active_prec = [en for en in PREC_ELEMS if en in models]
    if active_prec:
        pred_overlay_prec = create_multi_elem_overlay(om_raw_crop, pred_bins, valid_mask_np, active_prec)
        save_overlay_plot(pred_overlay_prec, active_prec, 2, os.path.join(OUTPUT_DIR, f"{NEW_BASE_NAME}_prec4_last_Pred.png"))

    print(f"[INFO] Publication-ready overlay images saved gracefully. Directory: {OUTPUT_DIR}")
    print("\n[SUCCESS] Pipeline execution completed with 100% normalized composition metrics!")


if __name__ == '__main__':
    main()