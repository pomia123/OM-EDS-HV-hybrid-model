"""
Evaluation Script for EDS Spatial Map Prediction Model
- Evaluates 'best' and 'last' checkpoint ensembles for targeted elements.
- Computes sample-level IoU, Dice, and Area metrics for deep ensemble, majority vote, and individual members.
- Computes and exports overall scalar area-based RMSE, MAE, and MAPE to a summary CSV.
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
from torchvision.models import resnet34, ResNet34_Weights
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
# TARGET_ELEMS = ["Mg", "Cu"]
THRESHOLD   = 0.5
N_ENSEMBLE  = 3
CROP_W, CROP_H = 512, 512

EDS_SUFFIXES = {"Mg":"01", "Al":"02", "Si":"03", "Fe":"06", "Cu":"07", "Sr":"09"}
# EDS_SUFFIXES = {"Mg":"01", "Cu":"07"}


plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial'] + plt.rcParams['font.sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# =============================================================================
# Utility Functions and Dataset Definition
# =============================================================================
def imread_korean(path, mode=cv2.IMREAD_COLOR):
    """Safely reads image files from paths containing non-ASCII characters."""
    try:
        n = np.fromfile(path, np.uint8)
        return cv2.imdecode(n, mode)
    except:
        return None

def preprocess_om_color(om_bgr):
    """Converts OM image from BGR to RGB format."""
    return cv2.cvtColor(om_bgr, cv2.COLOR_BGR2RGB)

class TestDataset(Dataset):
    """Dataset for loading OM images, GT EDS maps, and spatial masks."""
    def __init__(self, file_list, elem_name):
        self.file_list = file_list
        self.elem_name = elem_name
        self.suf = EDS_SUFFIXES[elem_name]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        base = self.file_list[idx]

        # Load raw images
        om_raw = imread_korean(os.path.join(OM_DIR,   f"{base}.png"), 1)
        g_mask = imread_korean(os.path.join(MASK_DIR, f"{base}.png"), 0)
        g_map  = imread_korean(os.path.join(MAP_DIR,  f"{base}.png"), 0)
        raw_eds = imread_korean(os.path.join(EDS_DIR, f"{base}_{self.suf}.png"), 0)
        
        # Center cropping
        h, w = g_mask.shape
        sy = (h - CROP_H) // 2
        sx = (w - CROP_W) // 2

        om_c    = preprocess_om_color(om_raw)[sy:sy+CROP_H, sx:sx+CROP_W]
        mask_c  = g_mask[sy:sy+CROP_H, sx:sx+CROP_W]
        map_c   = g_map [sy:sy+CROP_H, sx:sx+CROP_W]
        eds_c   = raw_eds[sy:sy+CROP_H, sx:sx+CROP_W]

        # Element-specific thresholding based on raw EDS distribution
        if self.elem_name == "Al":
            th = np.percentile(raw_eds, 20)
        elif self.elem_name == "Si":
            th = np.percentile(raw_eds, 90)
        else:
            th = np.percentile(raw_eds, 99)
        eds_bin = (eds_c > th).astype(np.uint8)

        # Tensor conversion and normalization
        om_t    = (torch.from_numpy(om_c).permute(2, 0, 1).float() / 127.5) - 1.0
        eds_t   = torch.from_numpy(eds_bin).float().unsqueeze(0)
        align_t = (torch.from_numpy(mask_c) > 0).float()
        almap_t = (torch.from_numpy(map_c)  > 0).float()
        
        valid_mask  = (align_t * almap_t).unsqueeze(0)
        align_only  = align_t.unsqueeze(0)

        return om_t, eds_t, valid_mask, align_only, base

# =============================================================================
# Model Architecture (Generator with CBAM & Decoder)
# =============================================================================
class ChannelAttention(nn.Module):
    """Channel Attention block for CBAM."""
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
    """Spatial Attention block for CBAM."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sig  = nn.Sigmoid()
    def forward(self, x):
        avg = x.mean(1, keepdim=True)
        mx, _ = x.max(1, keepdim=True)
        return x * self.sig(self.conv(torch.cat([avg, mx], 1)))

class CBAM(nn.Module):
    """Convolutional Block Attention Module."""
    def __init__(self, ch):
        super().__init__()
        self.ca = ChannelAttention(ch)
        self.sa = SpatialAttention()
    def forward(self, x):
        return self.sa(self.ca(x))

class ConvBnRelu(nn.Sequential):
    """Convolution - BatchNorm - ReLU block."""
    def __init__(self, in_ch, out_ch):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

class DecoderBlock(nn.Module):
    """Decoder block with transposed convolution, skip connection, and CBAM refinement."""
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
    """ResNet34-based U-Net Generator with CBAM attention."""
    def __init__(self, out_ch=1, pretrained=False):
        super().__init__()
        bb = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)

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
# Evaluation Metrics Formulation
# =============================================================================
def compute_area_metrics(pred_area_pct, gt_area_pct):
    """Calculates scalar Area percentage regression metrics: RMSE, MAE, MAPE."""
    p = np.array(pred_area_pct)
    g = np.array(gt_area_pct)

    rmse = float(np.sqrt(np.mean((p - g) ** 2)))
    mae  = float(np.mean(np.abs(p - g)))

    mask = g > 1e-6
    mape = float(np.mean(np.abs((p[mask] - g[mask]) / g[mask])) * 100) if mask.any() else float('nan')

    return rmse, mae, mape

# =============================================================================
# Main Evaluation Loop
# =============================================================================
def evaluate_ensemble(models, loader, device, elem_name, is_al, save_vis_dir=None, model_tag="best"):
    """Evaluates ensemble and per-member predictions, calculating overlap and area metrics."""
    for m in models:
        m.eval()

    all_pred_area, all_gt_area = [], []
    per_sample = []
    majority_thresh = len(models) // 2 + 1

    with torch.no_grad():
        for om_t, eds_t, valid_mask, align_only, base in loader:
            om_t = om_t.to(device)
            eds_t = eds_t.to(device)
            c_mask = align_only.to(device) if is_al else valid_mask.to(device)

            # Get predicted probabilities across all ensemble members
            member_probs = [torch.sigmoid(m(om_t)) for m in models]
            stacked = torch.stack(member_probs, dim=0)
            mean_prob = stacked.mean(dim=0)

            B = om_t.size(0)
            for b in range(B):
                fname = base[b]
                c_mask_b = c_mask[b, 0]
                total_valid_px = float(align_only[b, 0].sum().item()) or 1e-8

                # Ground Truth calculation
                gt_bin = (eds_t[b, 0] > 0.5).float() * c_mask_b
                gt_area = (gt_bin.sum().item() / total_valid_px) * 100.0
                all_gt_area.append(gt_area)

                # Deep Ensemble Overlap Metrics (Pixel-wise average thresholded)
                pred_bin = (mean_prob[b, 0] > THRESHOLD).float() * c_mask_b
                inter = (pred_bin * gt_bin).sum().item()
                union = (pred_bin + gt_bin).clamp(0, 1).sum().item()
                iou_ens = (inter + 1e-8) / (union + 1e-8)
                dice_ens = (2.0 * inter + 1e-8) / (pred_bin.sum().item() + gt_bin.sum().item() + 1e-8)

                # Per-Member Metrics
                per_member_dict = {}
                m_areas_this_sample = []
                m_bins_this_sample = []
                for i, m_p in enumerate(member_probs):
                    m_bin = (m_p[b, 0] > THRESHOLD).float() * c_mask_b
                    m_inter = (m_bin * gt_bin).sum().item()
                    m_union = (m_bin + gt_bin).clamp(0, 1).sum().item()
                    m_iou = (m_inter + 1e-8) / (m_union + 1e-8)
                    m_dice = (2.0 * m_inter + 1e-8) / (m_bin.sum().item() + gt_bin.sum().item() + 1e-8)
                    m_area = (m_bin.sum().item() / total_valid_px) * 100.0

                    m_areas_this_sample.append(m_area)
                    m_bins_this_sample.append(m_bin)

                    per_member_dict[f"dice_{i}"] = m_dice
                    per_member_dict[f"iou_{i}"] = m_iou
                    per_member_dict[f"pred_{i}"] = m_area

                # Majority Vote Metrics
                vote_count = torch.stack(m_bins_this_sample, dim=0).sum(dim=0)
                mv_bin = (vote_count >= majority_thresh).float() * c_mask_b
                mv_inter = (mv_bin * gt_bin).sum().item()
                mv_union = (mv_bin + gt_bin).clamp(0, 1).sum().item()
                mv_iou = (mv_inter + 1e-8) / (mv_union + 1e-8)
                mv_dice = (2.0 * mv_inter + 1e-8) / (mv_bin.sum().item() + gt_bin.sum().item() + 1e-8)

                pred_area_std = float(np.std(m_areas_this_sample))
                all_pred_area.append(float(np.mean(m_areas_this_sample)))

                sample_dict = {
                    "file": fname,
                    "dice_deep_ensemble": dice_ens,
                    "iou_deep_ensemble": iou_ens,
                    "dice_majority_vote": mv_dice,
                    "iou_majority_vote": mv_iou,
                    "gt_area": gt_area,
                    "pred_area_std": pred_area_std,
                    **per_member_dict
                }
                per_sample.append(sample_dict)

                # Export visualization artifacts for last model
                if save_vis_dir is not None and model_tag == "last":
                    pred_img = (mean_prob[b, 0] * c_mask_b).cpu().numpy()
                    gt_img = gt_bin.cpu().numpy()

                    fig, ax = plt.subplots(figsize=(10, 10))
                    ax.imshow(pred_img, cmap='gray', vmin=0, vmax=1.0)
                    ax.axis('off')
                    plt.savefig(os.path.join(save_vis_dir, f"{fname}_pred_{model_tag}.png"), dpi=150, bbox_inches='tight', pad_inches=0)
                    plt.close()

                    fig, ax = plt.subplots(figsize=(10, 10))
                    ax.imshow(gt_img, cmap='gray', vmin=0, vmax=1.0)
                    ax.axis('off')
                    plt.savefig(os.path.join(save_vis_dir, f"{fname}_gt.png"), dpi=150, bbox_inches='tight', pad_inches=0)
                    plt.close()

    rmse, mae, mape = compute_area_metrics(all_pred_area, all_gt_area)
    summary = {"rmse": rmse, "mae": mae, "mape": mape}

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
        print(f"[EVALUATION] Element: {elem_name}")
        print(f"{'='*60}")

        tag_paths = {"best": [], "last": []}
        for tag in ("best", "last"):
            for idx in range(N_ENSEMBLE):
                p = os.path.join(MODEL_DIR, f"{tag}_model_{key}_{idx}.pth")
                if os.path.exists(p):
                    tag_paths[tag].append(p)
                else:
                    print(f"  [WARNING] {tag.capitalize()} member {idx} not found: {p}")

        models_to_test = [(tag, paths) for tag, paths in tag_paths.items() if paths]
        if not models_to_test:
            print(f"  [ERROR] No models located for {elem_name}. Skipping evaluation.\n")
            continue

        dataset = TestDataset(test_files, elem_name)
        loader  = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

        vis_dir = os.path.join(TEST_DIR, f"vis_{elem_name}")
        os.makedirs(vis_dir, exist_ok=True)

        for model_tag, model_paths in models_to_test:
            print(f"  [PROGRESS] Evaluating {model_tag.upper()} ensemble ({len(model_paths)}/{N_ENSEMBLE} members) for {elem_name}...")
            models = []
            for p in model_paths:
                G = Generator(out_ch=1, pretrained=False).to(device)
                G.load_state_dict(torch.load(p, map_location=device))
                models.append(G)

            summary, per_sample = evaluate_ensemble(
                models, loader, device, elem_name, is_al,
                save_vis_dir=vis_dir,
                model_tag=model_tag,
            )

            summary_all[model_tag][elem_name] = summary

            for s in per_sample:
                per_sample_all[model_tag][s["file"]][elem_name] = s

            for m in models:
                m.cpu()
            del models
            torch.cuda.empty_cache()
            print(f"  [PROGRESS] Finished {model_tag.upper()} evaluation for {elem_name}.")
        print()

    # 1. Export sample-level results to results_per_sample.csv
    print("[INFO] Exporting sample-level evaluation results to CSV...")
    fieldnames = ["file_name", "model"]
    for e in TARGET_ELEMS:
        fieldnames += [
            f"{e}_dice_deep_ensemble", f"{e}_iou_deep_ensemble",
            f"{e}_dice_majority_vote", f"{e}_iou_majority_vote",
        ]
        for i in range(N_ENSEMBLE):
            fieldnames += [f"{e}_dice_{i}", f"{e}_iou_{i}"]
        for i in range(N_ENSEMBLE):
            fieldnames += [f"{e}_pred_{i}(%)"]
        fieldnames += [f"{e}_gt(%)", f"{e}_area_std(%p)"]

    csv_per_sample_path = os.path.join(TEST_DIR, "results_per_sample.csv")
    with open(csv_per_sample_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for model_tag in ("best", "last"):
            for file_name in test_files:
                row = {"file_name": file_name, "model": model_tag}
                elem_data = per_sample_all[model_tag][file_name]

                for e in TARGET_ELEMS:
                    if e in elem_data:
                        d = elem_data[e]
                        row[f"{e}_dice_deep_ensemble"] = f"{d['dice_deep_ensemble']:.4f}"
                        row[f"{e}_iou_deep_ensemble"]  = f"{d['iou_deep_ensemble']:.4f}"
                        row[f"{e}_dice_majority_vote"] = f"{d['dice_majority_vote']:.4f}"
                        row[f"{e}_iou_majority_vote"]  = f"{d['iou_majority_vote']:.4f}"

                        for i in range(N_ENSEMBLE):
                            row[f"{e}_dice_{i}"] = f"{d.get(f'dice_{i}', 0.0):.4f}"
                            row[f"{e}_iou_{i}"]  = f"{d.get(f'iou_{i}', 0.0):.4f}"
                        for i in range(N_ENSEMBLE):
                            row[f"{e}_pred_{i}(%)"] = f"{d.get(f'pred_{i}', 0.0):.4f}"

                        row[f"{e}_gt(%)"]        = f"{d['gt_area']:.4f}"
                        row[f"{e}_area_std(%p)"] = f"{d['pred_area_std']:.4f}"

                writer.writerow(row)

    # 2. Export overall area regression summary to results_area_summary.csv
    print("[INFO] Exporting overall area metric summary to CSV...")
    summary_fieldnames = ["model", "element", "rmse(%p)", "mae(%p)", "mape(%)"]
    csv_summary_path = os.path.join(TEST_DIR, "results_area_summary.csv")
    with open(csv_summary_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
        writer.writeheader()

        for model_tag in ("best", "last"):
            for e in TARGET_ELEMS:
                if e in summary_all[model_tag]:
                    s = summary_all[model_tag][e]
                    writer.writerow({
                        "model": model_tag,
                        "element": e,
                        "rmse(%p)": f"{s['rmse']:.4f}",
                        "mae(%p)": f"{s['mae']:.4f}",
                        "mape(%)": f"{s['mape']:.2f}" if not np.isnan(s['mape']) else "NaN"
                    })

    print(f"\n[SUCCESS] Sample-level results saved to: {csv_per_sample_path}")
    print(f"[SUCCESS] Area summary results saved to: {csv_summary_path}")
    print(f"[SUCCESS] Test evaluation pipeline completed successfully.")


if __name__ == '__main__':
    main()