"""
Visualization Script for EDS Spatial Map Predictions
- Generates precise overlay maps combining grayscale Optical Microscopy (OM) backgrounds
  with predicted vs. Ground Truth (GT) overlap masks.
- Categorizes pixels into Match (True Positive), Miss (False Negative), and False (False Positive).
- Tailored for high-quality, academic-level qualitative evaluation figures.
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
import matplotlib.patches as mpatches

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

TARGET_ELEMS = ["Mg", "Al", "Si", "Fe", "Cu", "Sr"]
THRESHOLD  = 0.5
CROP_W, CROP_H = 512, 512

EDS_SUFFIXES = {"Mg":"01", "Al":"02", "Si":"03", "Fe":"06", "Cu":"07", "Sr":"09"}

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans'] + plt.rcParams['font.sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# =============================================================================
# Utility Functions and Dataset Definition
# =============================================================================
def imread_korean(path, mode=cv2.IMREAD_COLOR):
    """
    Helper function to safely read image paths containing non-ASCII characters.
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
    Dataset class for the visualization phase.
    Loads and preprocesses OM images, GT EDS maps, and spatial masks.
    Additionally retains the raw cropped BGR OM image for background overlay processing.
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

        # Retain raw OM crop for background overlay processing
        om_raw_crop = om_raw[sy:sy+CROP_H, sx:sx+CROP_W]
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

        return om_t, eds_t, valid_mask, align_only, base, om_raw_crop

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

        # Encoder sections
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
# Visualization Processing Functions
# =============================================================================
def create_om_overlay_comparison(om_bgr, gt_bin, pred_bin):
    """
    Creates an overlay comparison map by converting the OM image into a subtle 
    grayscale background, and applying vividly colored masks to highlight 
    Match, Miss, and False prediction regions.
    """
    # 1. Convert cropped OM image to grayscale background for academic visualization
    om_gray = cv2.cvtColor(om_bgr, cv2.COLOR_BGR2GRAY)
    
    # 2. Adjust brightness to blend OM details lightly without overpowering the mask colors
    om_faded = cv2.addWeighted(om_gray, 0.4, np.full_like(om_gray, 255), 0.6, 0)
    canvas = cv2.cvtColor(om_faded, cv2.COLOR_GRAY2RGB)

    # 3. Define logical masks for error analysis
    tp_mask = (gt_bin == 1) & (pred_bin == 1)
    fn_mask = (gt_bin == 1) & (pred_bin == 0)
    fp_mask = (gt_bin == 0) & (pred_bin == 1)

    # 4. Map distinct colors to the respective regions on the grayscale canvas
    canvas[tp_mask] = [40, 180, 70]   # Match -> Green
    canvas[fn_mask] = [230, 180, 30]  # Miss  -> Yellow
    canvas[fp_mask] = [220, 50, 50]   # False -> Red

    return canvas

def evaluate_and_visualize_overlap(model, loader, device, elem_name, is_al, save_dir, model_tag):
    """
    Iterates through the dataset, generates predictions, and saves detailed 
    overlap comparison visualizations for each sample.
    """
    model.eval()
    target_save_path = os.path.join(save_dir, f"pure_mask_{elem_name}_{model_tag}")
    os.makedirs(target_save_path, exist_ok=True)

    with torch.no_grad():
        for om, eds, v_mask, align, base_names, om_raw_crops in loader:
            om = om.to(device)
            c_mask = align.to(device) if is_al else v_mask.to(device)

            logit = model(om)
            pred_sig = torch.sigmoid(logit)

            pred_np = pred_sig.cpu().numpy()
            gt_np   = eds.cpu().numpy()
            mask_np = c_mask.cpu().numpy()

            for b in range(om.shape[0]):
                m_b = mask_np[b, 0] > 0.5
                gt_raw   = gt_np[b, 0] > 0.5
                pred_raw = pred_np[b, 0] > THRESHOLD

                gt_bin   = np.where(m_b, gt_raw, 0).astype(np.uint8)
                pred_bin = np.where(m_b, pred_raw, 0).astype(np.uint8)

                # Extract raw BGR numpy image for the single batch crop
                single_om_bgr = om_raw_crops[b].numpy()

                # Generate the overlay comparison map
                overlay_comparison_map = create_om_overlay_comparison(single_om_bgr, gt_bin, pred_bin)

                fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
                ax.imshow(overlay_comparison_map)
                ax.axis('off')

                # Configure legend patches
                patch_match = mpatches.Patch(color='#28B446', label='Match')
                patch_miss  = mpatches.Patch(color='#E6B41E', label='Miss')
                patch_false = mpatches.Patch(color='#DC3232', label='False')
                
                # Maintain vertical alignment in the lower-right corner
                ax.legend(handles=[patch_match, patch_miss, patch_false], 
                          loc='lower right', ncol=1, fontsize=9, 
                          frameon=True, facecolor='white', edgecolor='none',
                          handletextpad=0.5, labelspacing=0.5, borderpad=0.5)

                save_fname = os.path.join(target_save_path, f"{base_names[b]}_mask_analysis.png")
                plt.savefig(save_fname, dpi=300, bbox_inches='tight', pad_inches=0.02)
                plt.close()

# =============================================================================
# Main Execution Pipeline
# =============================================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Visualization Device: {device}\n")

    splits_path = os.path.join(RESULT_DIR, 'tversky', 'splits.json')
    if not os.path.exists(splits_path):
        raise FileNotFoundError(f"[ERROR] 'splits.json' missing at {splits_path}\nPlease run the training script first.")

    with open(splits_path) as f:
        splits = json.load(f)
    test_files = splits['test']
    print(f"[INFO] Test samples loaded: {len(test_files)}\n")

    for elem_name in TARGET_ELEMS:
        key = elem_name.lower()
        is_al = (elem_name == "Al")

        print(f"{'='*60}")
        print(f"[VISUALIZATION] Element: {elem_name}")
        print(f"{'='*60}")

        best_path = os.path.join(MODEL_DIR, f"best_model_{key}.pth")
        last_path = os.path.join(MODEL_DIR, f"last_model_{key}.pth")

        models_to_test = []
        for tag, path in [("best", best_path), ("last", last_path)]:
            if os.path.exists(path):
                models_to_test.append((tag, path))

        if not models_to_test:
            print(f"  [WARNING] No valid models located for {elem_name}. Skipping. (Path: {MODEL_DIR})")
            continue

        dataset = TestDataset(test_files, elem_name)
        loader  = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

        for model_tag, model_path in models_to_test:
            print(f"  [INFO] Generating overlap maps for [{model_tag.upper()}] model...")
            G = Generator(out_ch=1).to(device)
            
            # Load weights and strip 'module.' prefix if trained with DataParallel
            state_dict = torch.load(model_path, map_location=device)
            strip_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            G.load_state_dict(strip_dict)

            evaluate_and_visualize_overlap(
                G, loader, device, elem_name, is_al,
                save_dir=TEST_DIR, model_tag=model_tag
            )
            print(f"  [SUCCESS] Visualizations saved for [{model_tag.upper()}].")

    print(f"\n[SUCCESS] Qualitative overlap visualizations for all elements completed!")
    print(f"[SUCCESS] Output directory: {TEST_DIR}")

if __name__ == '__main__':
    main()