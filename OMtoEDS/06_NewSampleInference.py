"""
=============================================================================
Description:
  - Performs inference using the 'last_model' weights on a single target sample 
    without requiring Ground Truth (GT) mask files.
  - Evaluation Metrics: 
    * The denominator for the area ratio calculation is fixed to the total 
      image area (512x512) for all elements.
    * Al (Numerator): Predicted pixels across the entire image area.
    * Other Elements (Numerator): Predicted pixels strictly within the valid 
      Region of Interest (ROI / MAP validity region).
  - Visualization: Generates high-resolution multi-element overlay maps, 
    restricting the predictions strictly within the valid MAP area for 
    enhanced visibility and academic presentation.
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

OM_DIR   = r'.\pred_data\OM_hv'
MAP_DIR  = r'.\pred_data\MAP_hv'
RESULT_DIR = r'.\pred_data\result'

MODEL_DIR  = os.path.join(r'.\result', 'tversky','models_tversky')
OUTPUT_DIR = os.path.join(RESULT_DIR, 'new_sample_inference')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Element sets configuration
ALL_ELEMS   = ["Al", "Si", "Mg", "Fe", "Cu", "Sr"]
PREC_ELEMS  = ["Mg", "Fe", "Cu", "Sr"]   # Precipitation elements
THRESHOLD   = 0.5
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
    """
    Reads image files with paths that may contain non-ASCII (e.g., Korean) characters.
    """
    try:
        n = np.fromfile(path, np.uint8)
        return cv2.imdecode(n, mode)
    except:
        return None

def load_and_center_crop_data(base):
    """
    Loads OM and MAP images, performs a center crop to (CROP_W, CROP_H),
    and returns tensors and numpy arrays prepared for inference.
    """
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

    # Binary validity mask from the MAP file
    valid_mask_np = (map_crop > 0).astype(np.uint8)

    # Convert OM image to PyTorch tensor and normalize to [-1.0, 1.0]
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
    """
    Creates a multi-element color overlay on top of a faded OM image.
    Predictions are strictly restricted within the MAP validity mask for visualization.
    """
    om_gray  = cv2.cvtColor(om_bgr, cv2.COLOR_BGR2GRAY)
    om_faded = cv2.addWeighted(om_gray, 0.4, np.full_like(om_gray, 255), 0.6, 0)
    canvas   = cv2.cvtColor(om_faded, cv2.COLOR_GRAY2RGB).astype(np.float32)

    for en in elem_names:
        # Visibility Correction: Render only within the valid MAP region regardless of metric formulation
        pred = (bins[en] > 0) & (valid_mask_np > 0)
        r, g, b = ELEM_COLORS_RGB[en]
        color = np.array([r, g, b], dtype=np.float32)
        canvas[pred] = (1.0 - elem_alpha) * canvas[pred] + elem_alpha * color

    return canvas.clip(0, 255).astype(np.uint8)

def save_overlay_plot(overlay_img, elem_names, legend_ncol, filename):
    """
    Saves the overlay image with a formatted publication-ready legend.
    """
    fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
    ax.imshow(overlay_img)
    ax.axis('off')

    custom_handles = [
        Line2D([0], [0], marker='o', color='none',
               markerfacecolor=tuple(c/255 for c in ELEM_COLORS_RGB[en]),
               markersize=7, label=en, alpha=0.9)
        for en in elem_names
    ]
    
    ax.legend(
        handles=custom_handles, loc='lower right', ncol=legend_ncol, fontsize=9,
        frameon=True, facecolor='white', edgecolor='none',
        handletextpad=0.2, columnspacing=0.6, labelspacing=0.4, borderpad=0.4
    )
    plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

# =============================================================================
# Model Loader
# =============================================================================
def load_last_model(elem_name, device):
    """
    Loads the 'last_model' weights for a specific element.
    """
    key = elem_name.lower()
    path = os.path.join(MODEL_DIR, f"last_model_{key}.pth")
    
    if not os.path.exists(path):
        return None

    print(f"[INFO] Loaded last_model for [{elem_name}]: {os.path.basename(path)}")
    G = Generator(out_ch=1).to(device)
    sd = torch.load(path, map_location=device)
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    G.load_state_dict(sd)
    return G

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

    # 2. Load Models Exclusive to 'last_model' Weights
    models = {}
    for en in ALL_ELEMS:
        G = load_last_model(en, device)
        if G is not None:
            models[en] = G
        else:
            print(f"[WARN] 'last_model' weights for {en} not found. Skipping element.")

    if not models:
        print("[Error] No 'last_model' weights found in the directory. Please check file names.")
        return

    # 3. Model Inference & Area Percentage Calculation
    pred_bins = {}
    area_results = []

    print(f"\n{'='*45}")
    print(f"  Element    |   Predicted Area Ratio (%)  ")
    print(f"{'='*45}")

    for en in ALL_ELEMS:
        if en in models:
            models[en].eval()
            with torch.no_grad():
                logit = models[en](om_t.to(device))
                pred_sig = torch.sigmoid(logit).cpu().squeeze().numpy()
                
                if en == "Al":
                    # For 'Al', extract predicted pixels across the entire cropped image
                    pred_bin = (pred_sig > THRESHOLD).astype(np.uint8)
                else:
                    # For other elements, strictly restrict predicted pixels to the valid MAP region
                    pred_bin = ((pred_sig > THRESHOLD) * valid_mask_np).astype(np.uint8)
        else:
            pred_bin = np.zeros((CROP_H, CROP_W), dtype=np.uint8)

        pred_bins[en] = pred_bin
        pred_area_pct = (pred_bin.sum() / total_image_pixels) * 100

        print(f"      {en:<8} |       {pred_area_pct:8.4f} %")
        area_results.append({"Element": en, "Pred_Area(%)": round(pred_area_pct, 4)})

    print(f"{'='*45}")

    # 4. Export Quantitative Metrics to CSV
    csv_out_path = os.path.join(OUTPUT_DIR, f"{NEW_BASE_NAME}_last_area_metrics_fixed_total_denom.csv")
    with open(csv_out_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=["Element", "Pred_Area(%)"])
        writer.writeheader()
        writer.writerows(area_results)
    print(f"[INFO] Metrics successfully saved to CSV: {csv_out_path}")

    # 5. Generate Multi-Element Overlay Visualizations
    active_all = [en for en in ALL_ELEMS if en in models]
    if active_all:
        pred_overlay_all = create_multi_elem_overlay(om_raw_crop, pred_bins, valid_mask_np, active_all)
        save_overlay_plot(pred_overlay_all, active_all, 3, os.path.join(OUTPUT_DIR, f"{NEW_BASE_NAME}_6elems_last_Pred.png"))

    active_prec = [en for en in PREC_ELEMS if en in models]
    if active_prec:
        pred_overlay_prec = create_multi_elem_overlay(om_raw_crop, pred_bins, valid_mask_np, active_prec)
        save_overlay_plot(pred_overlay_prec, active_prec, 2, os.path.join(OUTPUT_DIR, f"{NEW_BASE_NAME}_prec4_last_Pred.png"))

    print(f"[INFO] Publication-ready overlay images saved gracefully. Directory: {OUTPUT_DIR}")
    print("\n[SUCCESS] Pipeline execution completed with applied metric formulations!")


if __name__ == '__main__':
    main()