"""
Uncertainty Map Generation & Quantitative Summary Script for EDS Prediction Model
- Computes pixel-wise standard deviation (Deep Ensemble uncertainty).
- Excludes zero & low uncertainty (sigma <= EPSILON) pixels from quantitative statistics & visualizations.
- Uses subtle faded grayscale OM background with directly blended uncertainty overlay (03-style pipeline).
- Exports publication-ready standalone heatmaps & overlays with full-contrast colorbars (Titles removed).
- Quantifies and saves uncertainty statistics to CSV.
"""

import os
import csv
import json
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
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# =============================================================================
# Configuration Parameters
# =============================================================================
OM_DIR   = r'.\data\OM'
EDS_DIR  = r'.\data\EDS'
MASK_DIR = r'.\data\MASK'
MAP_DIR  = r'.\data\MAP'
RESULT_DIR = r'.\result'

MODEL_DIR = os.path.join(RESULT_DIR, 'tversky', 'models_tversky')
TEST_DIR  = os.path.join(RESULT_DIR, 'test_tversky')

ALL_ELEMS   = ["Al", "Si", "Mg", "Fe", "Cu", "Sr"]
N_ENSEMBLE  = 3
CROP_W, CROP_H = 512, 512
EDS_SUFFIXES = {"Mg":"01", "Al":"02", "Si":"03", "Fe":"06", "Cu":"07", "Sr":"09"}

UNCERTAINTY_DIR = os.path.join(TEST_DIR, 'uncertainty_maps')
CMAP          = 'viridis'
DPI           = 300
EPSILON       = 1e-6
OVERLAY_ALPHA = 0.6        # 03-style pixel blending ratio (0.6 color + 0.4 OM)
VMIN, VMAX    = 0.0, 0.58  # Fixed range for std colorbar

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial'] + plt.rcParams['font.sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# =============================================================================
# Utility Functions and Dataset Definition
# =============================================================================
def imread_korean(path, mode=cv2.IMREAD_COLOR):
    try:
        n = np.fromfile(path, np.uint8)
        return cv2.imdecode(n, mode)
    except:
        return None

def preprocess_om_color(om_bgr):
    return cv2.cvtColor(om_bgr, cv2.COLOR_BGR2RGB)

class MultiElemTestDataset(Dataset):
    def __init__(self, file_list, elem_names):
        self.file_list  = file_list
        self.elem_names = elem_names

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        base = self.file_list[idx]

        om_raw = imread_korean(os.path.join(OM_DIR,   f"{base}.png"), 1)
        g_mask = imread_korean(os.path.join(MASK_DIR, f"{base}.png"), 0)
        g_map  = imread_korean(os.path.join(MAP_DIR,  f"{base}.png"), 0)

        h, w = g_mask.shape
        sy = (h - CROP_H) // 2
        sx = (w - CROP_W) // 2

        om_raw_crop = om_raw[sy:sy+CROP_H, sx:sx+CROP_W]
        om_c   = preprocess_om_color(om_raw)[sy:sy+CROP_H, sx:sx+CROP_W]
        mask_c = g_mask[sy:sy+CROP_H, sx:sx+CROP_W]
        map_c  = g_map [sy:sy+CROP_H, sx:sx+CROP_W]

        eds_bins = {}
        for en in self.elem_names:
            suf = EDS_SUFFIXES[en]
            raw = imread_korean(os.path.join(EDS_DIR, f"{base}_{suf}.png"), 0)
            if raw is not None:
                raw = raw[sy:sy+CROP_H, sx:sx+CROP_W]
                if en == "Al":
                    th = np.percentile(raw, 20)
                elif en == "Si":
                    th = np.percentile(raw, 90)
                else:
                    th = np.percentile(raw, 99)
                eds_bins[en] = torch.from_numpy((raw > th).astype(np.uint8)).float().unsqueeze(0)
            else:
                eds_bins[en] = torch.zeros(1, CROP_H, CROP_W)

        om_t       = (torch.from_numpy(om_c).permute(2,0,1).float() / 127.5) - 1.0
        align_t    = (torch.from_numpy(mask_c) > 0).float()
        almap_t    = (torch.from_numpy(map_c)  > 0).float()

        valid_mask_al    = align_t.unsqueeze(0)
        valid_mask_other = (align_t * almap_t).unsqueeze(0)

        return om_t, eds_bins, valid_mask_al, valid_mask_other, base, om_raw_crop

def collate_fn(batch):
    om_list, eds_list, mask_al_list, mask_other_list, names, crops = zip(*batch)
    om_t         = torch.stack(om_list)
    mask_al_t    = torch.stack(mask_al_list)
    mask_other_t = torch.stack(mask_other_list)
    crops  = list(crops)
    elem_names = list(eds_list[0].keys())
    eds_stacked = {en: torch.stack([b[en] for b in eds_list]) for en in elem_names}
    return om_t, eds_stacked, mask_al_t, mask_other_t, list(names), crops

# =============================================================================
# Model Architecture
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

def load_model_ensemble(elem_name, model_tag, device):
    key = elem_name.lower()
    models = []
    for idx in range(N_ENSEMBLE):
        path = os.path.join(MODEL_DIR, f"{model_tag}_model_{key}_{idx}.pth")
        if not os.path.exists(path):
            continue
        G = Generator(out_ch=1).to(device)
        sd = torch.load(path, map_location=device)
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
        G.load_state_dict(sd)
        G.eval()
        models.append(G)
    return models

def compute_std_map(models, om_t):
    with torch.no_grad():
        member_probs = [torch.sigmoid(G(om_t)) for G in models]
        prob_stack = torch.stack(member_probs, dim=0)
        std_map = prob_stack.std(dim=0)[:, 0]
    return std_map.cpu().numpy()

# =============================================================================
# Figure Saving & Direct Blending Helpers (03-style)
# =============================================================================
def _masked_cmap():
    cmap_obj = plt.get_cmap(CMAP).copy()
    cmap_obj.set_bad(alpha=0)
    return cmap_obj

def create_uncertainty_overlay(om_bgr, std_map, valid_mask, elem_alpha=OVERLAY_ALPHA):
    """
    Directly blends pixel-wise uncertainty colors into the faded OM canvas (03-style).
    """
    om_gray  = cv2.cvtColor(om_bgr, cv2.COLOR_BGR2GRAY)
    om_faded = cv2.addWeighted(om_gray, 0.4, np.full_like(om_gray, 255), 0.6, 0)
    canvas   = cv2.cvtColor(om_faded, cv2.COLOR_GRAY2RGB).astype(np.float32)

    norm = mcolors.Normalize(vmin=VMIN, vmax=VMAX, clip=True)
    cmap = plt.get_cmap(CMAP)
    std_rgb = (cmap(norm(std_map))[:, :, :3] * 255.0).astype(np.float32)

    target_pixels = valid_mask & (std_map > EPSILON)
    canvas[target_pixels] = (1.0 - elem_alpha) * canvas[target_pixels] + elem_alpha * std_rgb[target_pixels]
    return canvas.clip(0, 255).astype(np.uint8)

def save_standalone(std_masked, base, elem, model_tag, save_dir):
    fig, ax = plt.subplots(figsize=(15, 15), dpi=DPI)
    im = ax.imshow(std_masked, cmap=_masked_cmap(), vmin=VMIN, vmax=VMAX)
    ax.axis('off')

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Ensemble std (sigmoid prob.)', fontsize=40, labelpad=20)
    cbar.ax.tick_params(labelsize=40, length=10, width=2, pad=8)

    fname = os.path.join(save_dir, f"{base}_{model_tag}_{elem}_uncertainty_standalone.png")
    plt.savefig(fname, dpi=DPI, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)

def save_overlay(om_bgr, std_map, valid_mask, base, elem, model_tag, save_dir):
    overlay_img = create_uncertainty_overlay(om_bgr, std_map, valid_mask, elem_alpha=OVERLAY_ALPHA)

    fig, ax = plt.subplots(figsize=(15, 15), dpi=DPI)
    ax.imshow(overlay_img)
    ax.axis('off')

    norm = mcolors.Normalize(vmin=VMIN, vmax=VMAX)
    sm = cm.ScalarMappable(cmap=plt.get_cmap(CMAP), norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Ensemble std (sigmoid prob.)', fontsize=40, labelpad=20)
    cbar.ax.tick_params(labelsize=40, length=10, width=2, pad=8)

    fname = os.path.join(save_dir, f"{base}_{model_tag}_{elem}_uncertainty_overlay.png")
    plt.savefig(fname, dpi=DPI, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)

# =============================================================================
# Per-Element Pipeline
# =============================================================================
def run_element(elem, model_tag, test_files, device):
    models = load_model_ensemble(elem, model_tag, device)
    if len(models) < 2:
        print(f"  [WARNING] {elem} ({model_tag}): fewer than 2 ensemble members found, skipping.")
        return None
    for G in models:
        G.eval()

    dataset = MultiElemTestDataset(test_files, [elem])
    loader = DataLoader(
        dataset, batch_size=4, shuffle=False, num_workers=0, collate_fn=collate_fn
    )

    save_dir = os.path.join(UNCERTAINTY_DIR, model_tag, elem)
    os.makedirs(save_dir, exist_ok=True)

    overall_stds = []
    fg_stds      = []
    bg_stds      = []

    with torch.no_grad():
        for om_t, eds_bins, v_mask_al, v_mask_other, base_names, om_raw_crops in loader:
            om_t = om_t.to(device)
            std_batch = compute_std_map(models, om_t)           # [B,H,W]
            v_mask = v_mask_al if elem == "Al" else v_mask_other
            mask_np = v_mask.cpu().numpy()[:, 0] > 0.5           # [B,H,W]

            B = om_t.shape[0]
            for b in range(B):
                m_b = mask_np[b]
                std_b = std_batch[b]
                gt_b = eds_bins[elem][b, 0].numpy() > 0.5

                significant_mask = std_b > EPSILON

                # 1. Overall Significant
                valid_pix = std_b[m_b & significant_mask]
                if len(valid_pix) > 0:
                    overall_stds.append(np.mean(valid_pix))

                # 2. Precipitate Significant (GT=1 & sigma > EPSILON)
                fg_pix = std_b[m_b & gt_b & significant_mask]
                if len(fg_pix) > 0:
                    fg_stds.append(np.mean(fg_pix))

                # 3. Background/Matrix Significant (GT=0 & sigma > EPSILON)
                bg_pix = std_b[m_b & (~gt_b) & significant_mask]
                if len(bg_pix) > 0:
                    bg_stds.append(np.mean(bg_pix))

                om_bgr = om_raw_crops[b] if isinstance(om_raw_crops[b], np.ndarray) else om_raw_crops[b].numpy()
                
                # Standalone masked array
                std_masked = np.ma.masked_where((~m_b) | (~significant_mask), std_b)

                save_standalone(std_masked, base_names[b], elem, model_tag, save_dir)
                save_overlay(om_bgr, std_b, m_b, base_names[b], elem, model_tag, save_dir)

    for G in models:
        G.cpu()
    del models
    torch.cuda.empty_cache()
    print(f"  [SUCCESS] {elem} ({model_tag}): uncertainty maps saved to {save_dir}")

    return {
        "Element": elem,
        "Overall_Mean_sigma_filtered": f"{np.mean(overall_stds):.4f}" if overall_stds else "0.0000",
        "Overall_Std_sigma_filtered": f"{np.std(overall_stds):.4f}" if overall_stds else "0.0000",
        "Precipitate_Mean_sigma_filtered": f"{np.mean(fg_stds):.4f}" if fg_stds else "0.0000",
        "Precipitate_Std_sigma_filtered": f"{np.std(fg_stds):.4f}" if fg_stds else "0.0000",
        "Matrix_Mean_sigma_filtered": f"{np.mean(bg_stds):.4f}" if bg_stds else "0.0000",
        "Matrix_Std_sigma_filtered": f"{np.std(bg_stds):.4f}" if bg_stds else "0.0000",
    }

# =============================================================================
# Main Execution Pipeline
# =============================================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}\n")

    splits_path = os.path.join(RESULT_DIR, 'tversky', 'splits.json')
    if not os.path.exists(splits_path):
        raise FileNotFoundError(f"[ERROR] 'splits.json' missing at {splits_path}")

    with open(splits_path) as f:
        splits = json.load(f)
    test_files = splits['test']
    print(f"[INFO] Test set: {len(test_files)} samples\n")

    for model_tag in ["last"]:
        print(f"\n[INFO] Generating filtered uncertainty maps & metrics ({model_tag.upper()})...")
        metrics_summary = []
        for elem in ALL_ELEMS:
            res = run_element(elem, model_tag, test_files, device)
            if res is not None:
                metrics_summary.append(res)

        # Export metrics directly to CSV
        csv_path = os.path.join(UNCERTAINTY_DIR, f"uncertainty_summary_filtered_{model_tag}.csv")
        os.makedirs(UNCERTAINTY_DIR, exist_ok=True)
        if metrics_summary:
            with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=list(metrics_summary[0].keys()))
                writer.writeheader()
                writer.writerows(metrics_summary)
            print(f"  [SUCCESS] Filtered uncertainty summary CSV saved to: {csv_path}")

    print(f"\n[SUCCESS] All tasks completed! Outputs saved under: {UNCERTAINTY_DIR}")

if __name__ == '__main__':
    main()