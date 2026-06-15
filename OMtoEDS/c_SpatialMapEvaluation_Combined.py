"""
Evaluation Script for EDS Spatial Map Prediction Model - Multi-Element Combined Visualization
- Generates composite multi-element spatial maps overlaying on OM images.
- Evaluates the 'best' and 'last' checkpoint models to visualize elemental distributions simultaneously.
- Produces two versions of visual outputs:
  1. All 6 Elements (Al, Si, Mg, Cu, Fe, Sr)
  2. Precipitate Elements Only (Mg, Cu, Fe, Sr)
- Utilizes customized academic color palettes for distinct multi-class representation.
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
from matplotlib.lines import Line2D

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

ALL_ELEMS   = ["Al", "Si", "Mg", "Cu", "Fe", "Sr"]
PREC_ELEMS  = ["Mg", "Cu", "Fe", "Sr"]   # Version excluding dominant phases (Al, Si)
THRESHOLD   = 0.5
CROP_W, CROP_H = 512, 512
EDS_SUFFIXES = {"Mg":"01", "Al":"02", "Si":"03", "Cu":"06", "Fe":"07", "Sr":"09"}

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial'] + plt.rcParams['font.sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# Academic complementary color palette for elements (RGB specification)
ELEM_COLORS_RGB = {
    # Dominant phases
    "Al": (215, 205, 190),   # soft beige-gray
    "Si": (235, 190,  70),   # muted gold

    # Sparse precipitates
    "Mg": (  0, 200, 255),   # cyan
    "Cu": (150,  70, 255),   # vivid purple
    "Fe": (220,  40,  40),   # crimson red
    "Sr": ( 80, 230,  80),   # lime green
}

# =============================================================================
# Utility Functions and Multi-Element Dataset Definition
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

class MultiElemTestDataset(Dataset):
    """
    Dataset class for the inference/testing phase for multiple elements simultaneously.
    Loads and preprocesses OM images, GT EDS maps for multiple elements, and spatial masks.
    """
    def __init__(self, file_list, elem_names):
        self.file_list  = file_list
        self.elem_names = elem_names

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        base = self.file_list[idx]

        # Load images using the helper function to support Korean filenames
        om_raw = imread_korean(os.path.join(OM_DIR,   f"{base}.png"), 1)
        g_mask = imread_korean(os.path.join(MASK_DIR, f"{base}.png"), 0)
        g_map  = imread_korean(os.path.join(MAP_DIR,  f"{base}.png"), 0)

        # Center cropping to defined dimensions
        h, w = g_mask.shape
        sy = (h - CROP_H) // 2
        sx = (w - CROP_W) // 2
        
        om_raw_crop = om_raw[sy:sy+CROP_H, sx:sx+CROP_W]
        om_c   = preprocess_om_color(om_raw)[sy:sy+CROP_H, sx:sx+CROP_W]
        mask_c = g_mask[sy:sy+CROP_H, sx:sx+CROP_W]
        map_c  = g_map [sy:sy+CROP_H, sx:sx+CROP_W]

        # Construct binarized Ground Truth dictionary for all target elements
        eds_bins = {}
        for en in self.elem_names:
            suf = EDS_SUFFIXES[en]
            raw = imread_korean(os.path.join(EDS_DIR, f"{base}_{suf}.png"), 0)
            if raw is not None:
                raw = raw[sy:sy+CROP_H, sx:sx+CROP_W]

                # Element-specific thresholding based on empirical percentile distributions
                if en == "Al":
                    th = np.percentile(raw, 20)
                elif en == "Si":
                    th = np.percentile(raw, 90)
                else:
                    th = np.percentile(raw, 99)
                eds_bins[en] = torch.from_numpy((raw > th).astype(np.uint8)).float().unsqueeze(0)
            else:
                eds_bins[en] = torch.zeros(1, CROP_H, CROP_W)

        # Normalization and tensor conversion
        om_t       = (torch.from_numpy(om_c).permute(2,0,1).float() / 127.5) - 1.0
        align_t    = (torch.from_numpy(mask_c) > 0).float()
        almap_t    = (torch.from_numpy(map_c)  > 0).float()
        
        # Define common valid mask (align * almap) for all elements including Al
        valid_mask = (align_t * almap_t).unsqueeze(0)

        return om_t, eds_bins, valid_mask, base, om_raw_crop

def collate_fn(batch):
    """
    Custom collate function to handle variable element dictionaries and numpy array crops.
    """
    om_list, eds_list, mask_list, names, crops = zip(*batch)
    om_t   = torch.stack(om_list)
    mask_t = torch.stack(mask_list)
    crops  = list(crops)
    elem_names = list(eds_list[0].keys())
    eds_stacked = {en: torch.stack([b[en] for b in eds_list]) for en in elem_names}
    return om_t, eds_stacked, mask_t, list(names), crops

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
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        return self.cbam(self.conv(torch.cat([x, skip], 1)))

class Generator(nn.Module):
    """
    U-Net style Generator based on a ResNet-34 encoder.
    Maps OM images to predicted EDS spatial maps.
    """
    def __init__(self, out_ch=1):
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
# Image Processing & Overlay Formulation
# =============================================================================

def create_multi_elem_overlay(om_bgr, pred_bins, valid_mask_np, elem_names, elem_alpha=0.6):
    """
    Blends multiple elemental binary predictions into a single overlay on the OM background.
    """
    om_gray  = cv2.cvtColor(om_bgr, cv2.COLOR_BGR2GRAY)
    
    # Process light gray faded background (0.4 * gray + 153)
    om_faded = cv2.addWeighted(om_gray, 0.4, np.full_like(om_gray, 255), 0.6, 0)
    canvas   = cv2.cvtColor(om_faded, cv2.COLOR_GRAY2RGB).astype(np.float32)

    for en in elem_names:
        pred = (pred_bins[en] > 0) & valid_mask_np
        r, g, b = ELEM_COLORS_RGB[en]
        color = np.array([r, g, b], dtype=np.float32)
        
        # Apply pixel blending with specific element alpha
        canvas[pred] = (1.0 - elem_alpha) * canvas[pred] + elem_alpha * color

    return canvas.clip(0, 255).astype(np.uint8)

# =============================================================================
# Core Evaluation and Visualization Logic
# =============================================================================
def evaluate_and_save_multi_overlay(models, loader, device, elem_names, save_dir, model_tag, legend_ncol):
    """
    Runs model inference per batch and exports GT & Predicted multi-element overlays.
    """

    os.makedirs(save_dir, exist_ok=True)

    for G in models.values():
        G.eval()

    with torch.no_grad():
        for om_t, eds_bins, v_mask, base_names, om_raw_crops in loader:
            om_t    = om_t.to(device)
            mask_np = v_mask.cpu().numpy()[:, 0]

            # Batch-wise inference for each element
            pred_bins_batch = {}
            for en in elem_names:
                logit = models[en](om_t)
                pred_sig = torch.sigmoid(logit).cpu().numpy()
                pred_bins_batch[en] = (pred_sig[:, 0] > THRESHOLD).astype(np.uint8)

            B = om_t.shape[0]
            for b in range(B):
                m_b = mask_np[b] > 0.5

                # Extract Binary Maps
                pred_bins_b = {en: pred_bins_batch[en][b] for en in elem_names}
                gt_bins_b   = {en: eds_bins[en][b, 0].numpy().astype(np.uint8) for en in elem_names}

                om_bgr = om_raw_crops[b] if isinstance(om_raw_crops[b], np.ndarray) else om_raw_crops[b].numpy()

                # Generate Overlays
                pred_overlay = create_multi_elem_overlay(om_bgr, pred_bins_b, m_b, elem_names)
                gt_overlay   = create_multi_elem_overlay(om_bgr, gt_bins_b, m_b, elem_names)

                # Shared Legend Configuration
                custom_handles = [
                    Line2D(
                        [0], [0],
                        marker='o',
                        color='none',
                        markerfacecolor=tuple(c/255 for c in ELEM_COLORS_RGB[en]),
                        markersize=7,
                        label=en,
                        alpha=0.9
                    )
                    for en in elem_names
                ]

                # ---------- Save Ground Truth (GT) ----------
                fig, ax = plt.subplots(figsize=(5, 5), dpi=300)

                ax.imshow(gt_overlay)
                ax.axis('off')

                ax.legend(
                    handles=custom_handles,
                    loc='lower right',
                    ncol=legend_ncol,
                    fontsize=9,
                    frameon=True,
                    facecolor='white',
                    edgecolor='none',
                    handletextpad=0.2,
                    columnspacing=0.6,
                    labelspacing=0.4,
                    borderpad=0.4
                )

                gt_fname = os.path.join(
                    save_dir,
                    f"{base_names[b]}_{model_tag}_GT.png"
                )

                plt.savefig(
                    gt_fname,
                    dpi=300,
                    bbox_inches='tight',
                    pad_inches=0
                )

                plt.close()

                # ---------- Save Predictions ----------
                fig, ax = plt.subplots(figsize=(5, 5), dpi=300)

                ax.imshow(pred_overlay)
                ax.axis('off')

                ax.legend(
                    handles=custom_handles,
                    loc='lower right',
                    ncol=legend_ncol,
                    fontsize=9,
                    frameon=True,
                    facecolor='white',
                    edgecolor='none',
                    handletextpad=0.2,
                    columnspacing=0.6,
                    labelspacing=0.4,
                    borderpad=0.4
                )

                pred_fname = os.path.join(
                    save_dir,
                    f"{base_names[b]}_{model_tag}_Pred.png"
                )

                plt.savefig(
                    pred_fname,
                    dpi=300,
                    bbox_inches='tight',
                    pad_inches=0
                )

                plt.close()

# =============================================================================
# Helper function to load model checkpoints
# =============================================================================
def load_model(elem_name, model_tag, device):
    """Loads a specific element's generator model checkpoint."""
    key  = elem_name.lower()
    path = os.path.join(MODEL_DIR, f"{model_tag}_model_{key}.pth")
    if not os.path.exists(path):
        return None

    G = Generator(out_ch=1).to(device)
    sd = torch.load(path, map_location=device)
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    G.load_state_dict(sd)
    return G

# =============================================================================
# Main Execution Pipeline
# =============================================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using Device for Visualization: {device}\n")

    splits_path = os.path.join(RESULT_DIR, 'tversky','splits.json')
    if not os.path.exists(splits_path):
        raise FileNotFoundError(f"[ERROR] 'splits.json' missing at {splits_path}\nPlease ensure the training script ran correctly.")

    with open(splits_path) as f:
        splits = json.load(f)
    test_files = splits['test']
    print(f"[INFO] Test set defined: {len(test_files)} samples\n")

    for model_tag in ["best", "last"]:
        print(f"\n[INFO] Starting multi-element synthesis using {model_tag.upper()} models...")

        # Synthesis for all 6 Elements (Grid: 3 cols, 2 rows)
        avail_all = {en: load_model(en, model_tag, device) for en in ALL_ELEMS}
        avail_all = {en: m for en, m in avail_all.items() if m is not None}

        if avail_all:
            dataset = MultiElemTestDataset(test_files, list(avail_all.keys()))
            loader  = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0, collate_fn=collate_fn)
            save_dir = os.path.join(TEST_DIR, f"all_elems_{model_tag}")
            
            evaluate_and_save_multi_overlay(
                models=avail_all, loader=loader, device=device,
                elem_names=list(avail_all.keys()), save_dir=save_dir, model_tag=model_tag,
                legend_ncol=2
            )
            print(f"  [SUCCESS] All 6-elements group saved to: {save_dir}")
        else:
            print(f"  [WARNING] All 6-elements group skipped: Valid models not found.")

        # Synthesis for Precipitate Elements (Al, Si excluded / Grid: 2 cols, 2 rows)
        avail_prec = {en: load_model(en, model_tag, device) for en in PREC_ELEMS}
        avail_prec = {en: m for en, m in avail_prec.items() if m is not None}

        if avail_prec:
            dataset = MultiElemTestDataset(test_files, list(avail_prec.keys()))
            loader  = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0, collate_fn=collate_fn)
            save_dir = os.path.join(TEST_DIR, f"prec_elems_{model_tag}")
            
            evaluate_and_save_multi_overlay(
                models=avail_prec, loader=loader, device=device,
                elem_names=list(avail_prec.keys()), save_dir=save_dir, model_tag=model_tag,
                legend_ncol=2 
            )
            print(f"  [SUCCESS] Precipitate 4-elements group saved to: {save_dir}")
        else:
            print(f"  [WARNING] Precipitate 4-elements group skipped: Valid models not found.")

    print(f"\n[SUCCESS] Multi-element overlay visualizations generated successfully!")
    print(f"[SUCCESS] Output directory: {TEST_DIR}")


if __name__ == '__main__':
    main()