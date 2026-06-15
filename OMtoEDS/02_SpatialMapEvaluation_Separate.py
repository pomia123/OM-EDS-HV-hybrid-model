"""
Evaluation Script for EDS Spatial Map Prediction Model
- Simultaneously evaluates the 'best' and 'last' checkpoint models for targeted elements.
- Evaluation Metrics: IoU, Dice Coefficient, RMSE (%p), MAE (%p), MAPE (%), R² (based on area %).
  * Denominator (Common): Number of valid pixels within the primary spatial MASK (white regions).
  * Numerator (Al): Number of predicted/GT pixels constrained within the primary MASK region.
  * Numerator (Others): Number of predicted/GT pixels constrained within the intersection of MASK and MAP regions.
- Utilizes the predefined test data split from 'splits.json' generated during the training phase.
"""

import os
import json
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet34
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv


# =============================================================================
# Configuration Parameters
# =============================================================================
OM_DIR   = r'.\data\OM'
EDS_DIR  = r'.\data\EDS'
MASK_DIR = r'.\data\MASK'
MAP_DIR  = r'.\data\MAP'
RESULT_DIR = r'.\result'
MODEL_DIR  = os.path.join(RESULT_DIR, 'tversky', 'models_tversky')
TEST_DIR   = os.path.join(RESULT_DIR, 'test_tversky')
os.makedirs(TEST_DIR, exist_ok=True)

TARGET_ELEMS = ["Mg", "Al", "Si", "Fe", "Cu", "Sr"]
THRESHOLD   = 0.5 
CROP_W, CROP_H = 512, 512

EDS_SUFFIXES = {"Mg":"01", "Al":"02", "Si":"03", "Fe":"06", "Cu":"07", "Sr":"09"}
ELEM_NAMES   = ["Mg", "Al", "Si", "Fe", "Cu", "Sr"]

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial'] + plt.rcParams['font.sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# =============================================================================
# Utility Functions and Dataset Definition
# =============================================================================
def imread_korean(path, mode=cv2.IMREAD_COLOR):
    """
    Helper function to safely read image paths containing non-ASCII (e.g., Korean) characters.
    """
    try:
        n = np.fromfile(path, np.uint8)
        return cv2.imdecode(n, mode)
    except:
        return None

def preprocess_om_color(om_bgr):
    """Converts the Optical Microscopy (OM) image from BGR to RGB color space."""
    return cv2.cvtColor(om_bgr, cv2.COLOR_BGR2RGB)

class TestDataset(Dataset):
    """
    Dataset class for the inference/testing phase.
    Loads and preprocesses OM images, corresponding Ground Truth (GT) EDS maps, and spatial masks.
    """
    def __init__(self, file_list, elem_name):
        self.file_list = file_list
        self.elem_name = elem_name
        self.suf = EDS_SUFFIXES[elem_name]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        base = self.file_list[idx]

        # Load images
        om_raw = imread_korean(os.path.join(OM_DIR,   f"{base}.png"), 1)
        g_mask = imread_korean(os.path.join(MASK_DIR, f"{base}.png"), 0)
        g_map  = imread_korean(os.path.join(MAP_DIR,  f"{base}.png"), 0)
        raw_eds = imread_korean(os.path.join(EDS_DIR, f"{base}_{self.suf}.png"), 0)
        
        # Center cropping to defined dimensions
        h, w = g_mask.shape
        sy = (h - CROP_H) // 2
        sx = (w - CROP_W) // 2

        om_c    = preprocess_om_color(om_raw)[sy:sy+CROP_H, sx:sx+CROP_W]
        mask_c  = g_mask[sy:sy+CROP_H, sx:sx+CROP_W]
        map_c   = g_map [sy:sy+CROP_H, sx:sx+CROP_W]
        eds_c   = raw_eds[sy:sy+CROP_H, sx:sx+CROP_W]

        # Element-specific thresholding based on empirical percentile distributions
        if self.elem_name == "Al":
            th = np.percentile(eds_c, 20)
        elif self.elem_name == "Si":
            th = np.percentile(eds_c, 90)
        else:
            th = np.percentile(eds_c, 99)
        eds_bin = (eds_c > th).astype(np.uint8)

        # Normalization and tensor conversion
        om_t    = (torch.from_numpy(om_c).permute(2, 0, 1).float() / 127.5) - 1.0
        eds_t   = torch.from_numpy(eds_bin).float().unsqueeze(0)
        align_t = (torch.from_numpy(mask_c) > 0).float()
        almap_t = (torch.from_numpy(map_c)  > 0).float()
        
        # Valid region definitions
        valid_mask  = (align_t * almap_t).unsqueeze(0)
        align_only  = align_t.unsqueeze(0)

        return om_t, eds_t, valid_mask, align_only, base

# =============================================================================
# Model Architecture (Generator with CBAM & Decoder)
# =============================================================================
class ChannelAttention(nn.Module):
    """Channel Attention Module for the CBAM architecture."""
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
    """Spatial Attention Module for the CBAM architecture."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sig  = nn.Sigmoid()
    def forward(self, x):
        avg = x.mean(1, keepdim=True)
        mx, _ = x.max(1, keepdim=True)
        return x * self.sig(self.conv(torch.cat([avg, mx], 1)))

class CBAM(nn.Module):
    """Convolutional Block Attention Module (CBAM)."""
    def __init__(self, ch):
        super().__init__()
        self.ca = ChannelAttention(ch)
        self.sa = SpatialAttention()
    def forward(self, x):
        return self.sa(self.ca(x))

class ConvBnRelu(nn.Sequential):
    """Standard Convolution-BatchNorm-ReLU block."""
    def __init__(self, in_ch, out_ch):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

class DecoderBlock(nn.Module):
    """
    Decoder block featuring transposed convolutions, skip connections, 
    and CBAM for feature refinement.
    """
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
        # Handle shape mismatch if any
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        return self.cbam(self.conv(torch.cat([x, skip], 1)))

class Generator(nn.Module):
    """
    U-Net style Generator based on a ResNet-34 encoder.
    Maps OM images to predicted EDS spatial maps.
    """
    def __init__(self, out_ch=1, pretrained=False):
        super().__init__()
        bb = resnet34(weights=None)

        # Encoder sections (ResNet-34 blocks)
        self.enc0 = nn.Sequential(bb.conv1, bb.bn1, bb.relu)
        self.pool = bb.maxpool
        self.enc1 = bb.layer1
        self.enc2 = bb.layer2
        self.enc3 = bb.layer3
        self.enc4 = bb.layer4

        # Decoder sections
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
# Evaluation Metrics Formulation
# =============================================================================
def compute_segmentation_metrics(pred_bin, gt_bin):
    """
    Calculates spatial overlap metrics: Intersection over Union (IoU) and Dice Coefficient.
    """
    inter = (pred_bin & gt_bin).sum()
    union = (pred_bin | gt_bin).sum()
    sum_  = pred_bin.sum() + gt_bin.sum()

    iou  = (inter + 1e-8) / (union + 1e-8)
    dice = (2 * inter + 1e-8) / (sum_  + 1e-8)
    return float(iou), float(dice)

def compute_area_metrics(pred_area_pct, gt_area_pct):
    """
    Calculates regression metrics based on the predicted vs. GT area percentages.
    Includes RMSE, MAE, MAPE, and R-squared (R²).
    """
    p = np.array(pred_area_pct)
    g = np.array(gt_area_pct)

    rmse = float(np.sqrt(np.mean((p - g) ** 2)))
    mae  = float(np.mean(np.abs(p - g)))

    mask = g > 1e-6
    mape = float(np.mean(np.abs((p[mask] - g[mask]) / g[mask])) * 100) if mask.any() else float('nan')

    ss_res = np.sum((g - p) ** 2)
    ss_tot = np.sum((g - g.mean()) ** 2)
    r2 = float(1 - ss_res / (ss_tot + 1e-8))

    return rmse, mae, mape, r2

# =============================================================================
# Main Evaluation Loop with Spatial Filtering
# =============================================================================
def evaluate_model(model, loader, device, elem_name, is_al, save_vis_dir=None, model_tag=""):
    """
    Evaluates the prediction model utilizing spatially constrained denominators and numerators.
    """
    model.eval()

    all_iou, all_dice = [], []
    all_pred_area, all_gt_area = [], []
    per_sample = []

    with torch.no_grad():
        for om, eds, v_mask, align, base_names in loader:
            om    = om.to(device)
            eds   = eds.to(device)
            
            # Spatial masks for area constraining
            v_mask_np = v_mask.cpu().numpy()     # (B,1,H,W) - MAP 유효 마스크
            align_np  = align.cpu().numpy()      # (B,1,H,W) - MASK 유효 마스크

            logit = model(om)
            pred_sig = torch.sigmoid(logit).cpu().numpy()
            gt_raw   = eds.cpu().numpy()

            for b in range(om.shape[0]):
                p_img = pred_sig[b, 0]
                g_img = gt_raw[b, 0]
                
                mask_b = (align_np[b, 0] > 0).astype(np.uint8)  # MASK 영역
                map_b  = (v_mask_np[b, 0] > 0).astype(np.uint8) # MAP 영역

                if is_al:
                    pred_bin = ((p_img > THRESHOLD) * mask_b).astype(np.uint8)
                    gt_bin   = ((g_img > 0.5) * mask_b).astype(np.uint8)
                else:
                    pred_bin = ((p_img > THRESHOLD) * mask_b * map_b).astype(np.uint8)
                    gt_bin   = ((g_img > 0.5) * mask_b * map_b).astype(np.uint8)

                iou, dice = compute_segmentation_metrics(pred_bin, gt_bin)
                all_iou.append(iou)
                all_dice.append(dice)

                total_valid_px = float(mask_b.sum())
                if total_valid_px == 0:
                    total_valid_px = 1e-8 

                pred_area = (pred_bin.sum() / total_valid_px) * 100
                gt_area   = (gt_bin.sum()   / total_valid_px) * 100
                all_pred_area.append(pred_area)
                all_gt_area.append(gt_area)

                per_sample.append({
                    "file":      base_names[b],
                    "iou":       iou,
                    "dice":      dice,
                    "pred_area": pred_area,
                    "gt_area":   gt_area,
                })

                # Export qualitative visualization figures
                if save_vis_dir:
                    base_fname = f"{base_names[b]}_{elem_name}_{model_tag}"

                    om_vis = ((om[b].cpu().permute(1,2,0).numpy() + 1) / 2).clip(0, 1)
                    fig, ax = plt.subplots(figsize=(4, 4))
                    ax.imshow(om_vis)
                    ax.axis('off')
                    plt.savefig(os.path.join(save_vis_dir, f"{base_fname}_OM.png"), dpi=300, bbox_inches='tight', pad_inches=0)
                    plt.close()

                    gt_vis = (gt_bin * 255).astype(np.uint8)
                    fig, ax = plt.subplots(figsize=(4, 4))
                    ax.imshow(gt_vis, cmap='gray', vmin=0, vmax=255)
                    ax.axis('off')
                    plt.savefig(os.path.join(save_vis_dir, f"{base_fname}_GT.png"), dpi=300, bbox_inches='tight', pad_inches=0)
                    plt.close()

                    pred_vis = (pred_bin * 255).astype(np.uint8)
                    fig, ax = plt.subplots(figsize=(4, 4))
                    ax.imshow(pred_vis, cmap='gray', vmin=0, vmax=255)
                    ax.axis('off')
                    plt.savefig(os.path.join(save_vis_dir, f"{base_fname}_Pred.png"), dpi=300, bbox_inches='tight', pad_inches=0)
                    plt.close()

    rmse, mae, mape, r2 = compute_area_metrics(all_pred_area, all_gt_area)

    summary = {
        "iou":  float(np.mean(all_iou)),
        "dice": float(np.mean(all_dice)),
        "rmse": rmse,
        "mae":  mae,
        "mape": mape,
        "r2":   r2,
    }
    return summary, per_sample

# =============================================================================
# Main Execution Pipeline
# =============================================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using Device: {device}\n")

    splits_path = os.path.join(RESULT_DIR, 'tversky', 'splits.json')
    if not os.path.exists(splits_path):
        raise FileNotFoundError(f"[ERROR] 'splits.json' missing at {splits_path}\nPlease run the training script first.")

    with open(splits_path) as f:
        splits = json.load(f)
    test_files = splits['test']
    print(f"[INFO] Test set defined: {len(test_files)} samples\n")

    per_sample_all = {"best": {f: {} for f in test_files},
                      "last": {f: {} for f in test_files}}
    summary_all    = {"best": {}, "last": {}} 

    for elem_name in TARGET_ELEMS:
        key   = elem_name.lower()
        is_al = (elem_name == "Al")
        print(f"{'='*60}")
        print(f"[EVALUATION] Element: {elem_name} (Threshold: {THRESHOLD})")
        print(f"{'='*60}")

        best_path = os.path.join(MODEL_DIR, f"best_model_{key}.pth")
        last_path = os.path.join(MODEL_DIR, f"last_model_{key}.pth")

        models_to_test = []
        for tag, path in [("best", best_path), ("last", last_path)]:
            if os.path.exists(path):
                models_to_test.append((tag, path))
            else:
                print(f"  [WARNING] {tag.capitalize()} model not found: {path}")

        if not models_to_test:
            print(f"  [ERROR] No valid models located for {elem_name}. Skipping evaluation.\n")
            continue

        dataset = TestDataset(test_files, elem_name)
        loader  = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

        vis_dir = os.path.join(TEST_DIR, f"vis_{elem_name}")
        os.makedirs(vis_dir, exist_ok=True)

        elem_summary = {}
        for model_tag, model_path in models_to_test:
            print(f"\n  [INFO] Loading {model_tag.upper()} model: {model_path}")
            G = Generator(out_ch=1, pretrained=False).to(device)
            G.load_state_dict(torch.load(model_path, map_location=device))

            summary, per_sample = evaluate_model(
                G, loader, device, elem_name, is_al,
                save_vis_dir=vis_dir,
                model_tag=model_tag,
            )

            print(f"\n  ── {elem_name} [{model_tag.capitalize()}] Final Metrics ──")
            print(f"  IoU     : {summary['iou']:.4f}")
            print(f"  Dice    : {summary['dice']:.4f}")
            print(f"  RMSE    : {summary['rmse']:.4f} %p")
            print(f"  MAE     : {summary['mae']:.4f} %p")
            print(f"  MAPE    : {summary['mape']:.2f} %")
            print(f"  R²      : {summary['r2']:.4f}")

            summary_all[model_tag][elem_name] = summary
            elem_summary[model_tag] = summary

            for s in per_sample:
                per_sample_all[model_tag][s["file"]][elem_name] = {
                    "iou":       s["iou"],
                    "dice":      s["dice"],
                    "pred_area": s["pred_area"],
                    "gt_area":   s["gt_area"],
                }
        
    # Export comprehensive results to CSV
    iou_dice_cols = [f"{e}_{m}" for e in TARGET_ELEMS for m in ("iou", "dice")]
    pred_cols     = [f"{e}_pred(%)" for e in TARGET_ELEMS]
    gt_cols       = [f"{e}_gt(%)"   for e in TARGET_ELEMS]
    fieldnames    = ["FILE_NAME", "MODEL"] + iou_dice_cols + pred_cols + gt_cols

    csv_path = os.path.join(TEST_DIR, "results_per_sample.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for model_tag in ("best", "last"):
            for file_name in test_files:
                row = {"FILE_NAME": file_name, "MODEL": model_tag}
                elem_data = per_sample_all[model_tag][file_name]

                for e in TARGET_ELEMS:
                    if e in elem_data:
                        row[f"{e}_iou"]     = f"{elem_data[e]['iou']:.4f}"
                        row[f"{e}_dice"]    = f"{elem_data[e]['dice']:.4f}"
                        row[f"{e}_pred(%)"] = f"{elem_data[e]['pred_area']:.4f}"
                        row[f"{e}_gt(%)"]   = f"{elem_data[e]['gt_area']:.4f}"
                    else:
                        row[f"{e}_iou"]     = ""
                        row[f"{e}_dice"]    = ""
                        row[f"{e}_pred(%)"] = ""
                        row[f"{e}_gt(%)"]   = ""

                writer.writerow(row)

    print(f"\n[SUCCESS] Comprehensive results aggregated and saved to: {csv_path}")
    print(f"[SUCCESS] Test evaluation pipeline executed successfully. Output directory: {TEST_DIR}")


if __name__ == '__main__':
    main()