"""
Hyperparameter Grid Search for OM-to-EDS Binary Map Prediction.

This module evaluates non-Tversky loss hyperparameters for the two
representative elements selected by 00_DatasetClassImbalance.py. Each
candidate configuration is trained for a short run and ranked by validation
Mean Absolute Error (MAE) and Intersection over Union (IoU). The best 
configuration for each class-imbalance group is exported for use by the full 
Pix2Pix training pipeline.

Usage:
    python 00_HyperparamGridSearch_v2.py
    python 00_HyperparamGridSearch_v2.py --epochs 50 --batch-size 16
"""

import os
import json
import random
import argparse
import itertools
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torchvision.models import resnet34, ResNet34_Weights
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("albumentations.check_version").setLevel(logging.ERROR)
import albumentations as A

SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================================================
# 1. Dataset Paths and Search Configuration
# =============================================================================
OM_DIR   = r'.\data\OM'
EDS_DIR  = r'.\data\EDS'
MASK_DIR = r'.\data\MASK'
MAP_DIR  = r'.\data\MAP'
SAVE_DIR = r'.\result\tversky'
GRID_DIR = os.path.join(SAVE_DIR, 'grid_search')
os.makedirs(GRID_DIR, exist_ok=True)

CROP_W, CROP_H = 512, 512
EDS_SUFFIXES = {"Mg": "01", "Al": "02", "Si": "03", "Fe": "06", "Cu": "07", "Sr": "09"}
DEFAULT_GRID_ELEMENTS = ["Al", "Si", "Fe"]
LR_INIT = 0.0002
FULL_TRAIN_EPOCHS = 1000

# Fixed element-specific Tversky coefficients calibrated for class imbalance.
# These coefficients are intentionally excluded from the grid search.
TVERSKY_PARAMS = {
    'mg': (0.3, 0.7), 'al': (0.7, 0.3), 'si': (0.5, 0.5),
    'fe': (0.3, 0.7), 'cu': (0.3, 0.7), 'sr': (0.3, 0.7),
}

# Search space for pixel-loss weighting and focal/Tversky loss composition.
GRID = {
    "lambda_pix":       [10.0],
    "focal_alpha_pos":  [0.7, 0.8, 0.9],
    "focal_gamma":      [1.0, 2.0, 3.0],
    "loss_ratio":       [(0.3, 0.7), (0.5, 0.5), (0.7, 0.3)],  # (focal_w, tversky_w)
}

# Convergence diagnostic
VAL_INTERVAL = 5
CONVERGENCE_PATIENCE = 10
CONVERGENCE_MIN_DELTA = 0.05 
CONVERGENCE_PATIENCE_CHECKS = max(1, CONVERGENCE_PATIENCE // VAL_INTERVAL)

def imread_korean(path, mode=cv2.IMREAD_COLOR):
    try:
        n = np.fromfile(path, np.uint8)
        return cv2.imdecode(n, mode)
    except Exception:
        return None

def preprocess_om_color(om_bgr):
    return cv2.cvtColor(om_bgr, cv2.COLOR_BGR2RGB)

geom_aug = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Transpose(p=0.3),
    A.ShiftScaleRotate(
        shift_limit=0.05, scale_limit=0.1, rotate_limit=30, p=0.4,
        interpolation=cv2.INTER_LINEAR,
        mask_interpolation=cv2.INTER_NEAREST,
        border_mode=cv2.BORDER_REFLECT,
    ),
], additional_targets={
    'mask_ref': 'mask', 'map_ref': 'mask', 'eds0': 'mask',
})

color_aug = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    A.RandomGamma(p=0.2),
])

# =============================================================================
# 2. Dataset Definition and Augmentation Pipeline
# =============================================================================
class MetallurgyDataset(Dataset):
    def __init__(self, file_list, elem_name, is_train=True):
        self.file_names = file_list
        self.elem_name = elem_name
        self.suf = EDS_SUFFIXES[elem_name]
        self.is_train = is_train
        
        self.th_cache = {}
        for base in self.file_names:
            raw = imread_korean(os.path.join(EDS_DIR, f"{base}_{self.suf}.png"), 0)
            if self.elem_name == "Al":
                self.th_cache[base] = np.percentile(raw, 20)
            elif self.elem_name == "Si":
                self.th_cache[base] = np.percentile(raw, 90)
            else:
                self.th_cache[base] = np.percentile(raw, 99)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        base = self.file_names[idx]
        om_raw = imread_korean(os.path.join(OM_DIR, f"{base}.png"), 1)
        om_rgb = preprocess_om_color(om_raw)
        g_mask = imread_korean(os.path.join(MASK_DIR, f"{base}.png"), 0)
        g_map  = imread_korean(os.path.join(MAP_DIR, f"{base}.png"), 0)

        h, w = g_mask.shape
        if self.is_train:
            sy = random.randint(0, h - CROP_H); sx = random.randint(0, w - CROP_W)
        else:
            sy = (h - CROP_H) // 2; sx = (w - CROP_W) // 2

        om_c   = om_rgb[sy:sy+CROP_H, sx:sx+CROP_W]
        mask_c = g_mask[sy:sy+CROP_H, sx:sx+CROP_W]
        map_c  = g_map[sy:sy+CROP_H, sx:sx+CROP_W]

        raw = imread_korean(os.path.join(EDS_DIR, f"{base}_{self.suf}.png"), 0)
        
        th = self.th_cache[base]
        
        c = raw[sy:sy+CROP_H, sx:sx+CROP_W]
        eds_crop = (c > th).astype(np.uint8) * 255

        if self.is_train:
            aug = geom_aug(image=om_c, mask_ref=mask_c, map_ref=map_c, eds0=eds_crop)
            om_c, mask_c, map_c, eds_crop = aug["image"], aug["mask_ref"], aug["map_ref"], aug["eds0"]
            om_c = color_aug(image=om_c)["image"]

        om_t = (torch.from_numpy(om_c).permute(2, 0, 1).float() / 127.5) - 1.0
        eds_t = torch.from_numpy(eds_crop).float().unsqueeze(0) / 255.0
        align_t = (torch.from_numpy(mask_c) > 0).float()
        almap_t = (torch.from_numpy(map_c) > 0).float()
        valid_mask = (align_t * almap_t).unsqueeze(0)
        align_only = align_t.unsqueeze(0)
        return om_t, eds_t, valid_mask, align_only

# =============================================================================
# 3. Model Architecture Framework
# =============================================================================
class ChannelAttention(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        mid = max(ch // r, 4)
        self.avg = nn.AdaptiveAvgPool2d(1); self.mx = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(nn.Conv2d(ch, mid, 1, bias=False), nn.ReLU(inplace=True),
                                 nn.Conv2d(mid, ch, 1, bias=False))
        self.sig = nn.Sigmoid()
    def forward(self, x):
        return x * self.sig(self.fc(self.avg(x)) + self.fc(self.mx(x)))

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False); self.sig = nn.Sigmoid()
    def forward(self, x):
        avg = x.mean(1, keepdim=True); mx, _ = x.max(1, keepdim=True)
        return x * self.sig(self.conv(torch.cat([avg, mx], 1)))

class CBAM(nn.Module):
    def __init__(self, ch):
        super().__init__(); self.ca = ChannelAttention(ch); self.sa = SpatialAttention()
    def forward(self, x): return self.sa(self.ca(x))

class ConvBnRelu(nn.Sequential):
    def __init__(self, in_ch, out_ch):
        super().__init__(nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                          nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = nn.Sequential(ConvBnRelu(in_ch // 2 + skip_ch, out_ch), ConvBnRelu(out_ch, out_ch))
        self.cbam = CBAM(out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        return self.cbam(self.conv(torch.cat([x, skip], 1)))

class Generator(nn.Module):
    def __init__(self, out_ch=1, pretrained=True):
        super().__init__()
        bb = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        self.enc0 = nn.Sequential(bb.conv1, bb.bn1, bb.relu); self.pool = bb.maxpool
        self.enc1, self.enc2, self.enc3, self.enc4 = bb.layer1, bb.layer2, bb.layer3, bb.layer4
        self.dec4 = DecoderBlock(512, 256, 256); self.dec3 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128, 64, 64); self.dec1 = DecoderBlock(64, 64, 32)
        self.dec0 = nn.Sequential(nn.ConvTranspose2d(32, 32, 2, stride=2), ConvBnRelu(32, 32), CBAM(32))
        self.head = nn.Conv2d(32, out_ch, 1)

    def forward(self, x):
        H, W = x.shape[2:]
        e0 = self.enc0(x); e1 = self.enc1(self.pool(e0)); e2 = self.enc2(e1); e3 = self.enc3(e2); e4 = self.enc4(e3)
        d = self.dec4(e4, e3)
        d = self.dec3(d, e2)
        d = self.dec2(d, e1)
        d = self.dec1(d, e0)
        d = self.dec0(d)
        out = self.head(d)
        if out.shape[2:] != (H, W):
            out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        return out

class Discriminator(nn.Module):
    def __init__(self, in_ch=4):
        super().__init__()
        def dl(i, o, norm=True):
            layers = [nn.Conv2d(i, o, 4, 2, 1)]
            if norm: layers.append(nn.InstanceNorm2d(o))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)
        self.model = nn.Sequential(dl(in_ch, 64, norm=False), dl(64, 128), dl(128, 256), dl(256, 512),
                                    nn.Conv2d(512, 1, 3, 1, 1))
    def forward(self, om, eds): return self.model(torch.cat([om, eds], 1))

# =============================================================================
# 4. Loss Functions and Evaluation Metrics
# =============================================================================
def focal_loss(p_logit, t, valid_mask, gamma, alpha_pos):
    bce = F.binary_cross_entropy_with_logits(p_logit, t, reduction='none')
    p = torch.sigmoid(p_logit)
    pt = torch.where(t > 0.5, p, 1 - p)
    at = torch.where(t > 0.5, torch.full_like(t, alpha_pos), torch.full_like(t, 1 - alpha_pos))
    loss = at * (1 - pt) ** gamma * bce * valid_mask
    return loss.sum() / (valid_mask.sum() + 1e-8)

def tversky_loss(p_logit, t, valid_mask, alpha, beta, eps=1e-6):
    p = torch.sigmoid(p_logit) * valid_mask; t = t * valid_mask
    inter = (p * t).sum(dim=(2, 3)); fp = (p * (1 - t)).sum(dim=(2, 3)); fn = ((1 - p) * t).sum(dim=(2, 3))
    return (1 - (inter + eps) / (inter + alpha * fp + beta * fn + eps)).mean()

def calculate_iou(pred_logit, target, mask, threshold=0.5):
    pred_bin = (torch.sigmoid(pred_logit) > threshold).float() * mask
    target_bin = (target > 0.5).float() * mask
    inter = (pred_bin * target_bin).sum(dim=(2, 3))
    union = (pred_bin + target_bin).clamp(0, 1).sum(dim=(2, 3))
    return (inter + 1e-8) / (union + 1e-8)

def validate_generator(model, val_loader, is_al):
    """Returns validation-wide Mean IoU and Mean MAE for one configuration."""
    model.eval()
    iou_sum = 0.0
    sample_count = 0
    all_pred_area = []
    all_gt_area = []

    with torch.no_grad():
        for om_v, eds_v, v_mask_v, align_v in val_loader:
            om_v, eds_v = om_v.to(device), eds_v.to(device)
            
            # base_mask: For denominator calculation (strictly fixed to the MASK region)
            base_mask = align_v.to(device)
            # metric_mask (m_v): For numerator and IoU calculation (MASK for Al, MASK ∩ MAP for others)
            m_v = base_mask if is_al else v_mask_v.to(device)
            
            pred_logit = model(om_v)
            pred_sig = torch.sigmoid(pred_logit)

            # Calculate IoU
            iou_sum += calculate_iou(pred_logit, eds_v, m_v).sum().item()
            
            # Calculate Area per sample in the batch for MAE computation
            for b in range(om_v.size(0)):
                mask_b = base_mask[b, 0]  # MASK for denominator
                map_b  = m_v[b, 0]        # Region for numerator
                
                p_img = pred_sig[b, 0]
                g_img = eds_v[b, 0]
                
                pred_bin = (p_img > 0.5).float() * map_b
                gt_bin   = (g_img > 0.5).float() * map_b
                
                total_valid_px = mask_b.sum().item()
                if total_valid_px == 0: total_valid_px = 1e-8
                
                pred_area = (pred_bin.sum().item() / total_valid_px) * 100.0
                gt_area   = (gt_bin.sum().item() / total_valid_px) * 100.0
                
                all_pred_area.append(pred_area)
                all_gt_area.append(gt_area)
                
            sample_count += om_v.size(0)

    mean_iou = iou_sum / sample_count
    
    # Calculate MAE over the entire validation set
    p_arr = np.array(all_pred_area)
    g_arr = np.array(all_gt_area)
    mean_mae = float(np.mean(np.abs(p_arr - g_arr)))
    
    return mean_iou, mean_mae

# =============================================================================
# 5. Single-Configuration Training and Evaluation
# =============================================================================
def train_one_config(elem_name, config, train_files, val_files, epochs, batch_size, num_workers):
    key = elem_name.lower()
    is_al = (elem_name == "Al")
    ta, tb = TVERSKY_PARAMS[key]

#########################################################################################################################

    train_loader = DataLoader(MetallurgyDataset(train_files, elem_name, is_train=True),
                               batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(MetallurgyDataset(val_files, elem_name, is_train=False),
                             batch_size=4, shuffle=False, num_workers=max(num_workers // 2, 0))

#########################################################################################################################

    G = Generator(out_ch=1, pretrained=True).to(device)
    D = Discriminator(in_ch=4).to(device)

    opt_G = torch.optim.Adam(G.parameters(), LR_INIT, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), LR_INIT, betas=(0.5, 0.999))

    def lr_lambda(epoch):
        return 1.0 if epoch < 100 else max(
            0.0, 1.0 - (epoch - 100) / (FULL_TRAIN_EPOCHS - 100 + 1)
        )

    sched_G = torch.optim.lr_scheduler.LambdaLR(opt_G, lr_lambda)
    sched_D = torch.optim.lr_scheduler.LambdaLR(opt_D, lr_lambda)
    scaler_G, scaler_D = GradScaler(), GradScaler()

    focal_w, tversky_w = config["loss_ratio"]
    history = []
    
    best_val_mae = float('inf') # Lower MAE is better
    best_val_iou = -1.0         # For tracking purposes
    epoch_of_best = 0
    checks_since_improve = 0
    
    for epoch in range(epochs):
        G.train(); D.train()
        print(f"    Epoch {epoch + 1}/{epochs}", flush=True)

        for om, eds, v_mask, align_mask in train_loader:
            om, eds = om.to(device), eds.to(device)
            v_mask, align_mask = v_mask.to(device), align_mask.to(device)
            c_mask = align_mask if is_al else v_mask

            opt_G.zero_grad()
            with autocast():
                fake_logit = G(om); fake_sig = torch.sigmoid(fake_logit)
                d_fake = D(om, fake_sig * c_mask)
                g_gan = F.binary_cross_entropy_with_logits(d_fake, torch.ones_like(d_fake))
                loss_pix = (focal_w * focal_loss(fake_logit, eds, c_mask, config["focal_gamma"], config["focal_alpha_pos"]) +
                            tversky_w * tversky_loss(fake_logit, eds, c_mask, ta, tb))
                loss_G = 1.0 * g_gan + config["lambda_pix"] * loss_pix
            scaler_G.scale(loss_G).backward()
            scaler_G.unscale_(opt_G)
            nn.utils.clip_grad_norm_(G.parameters(), max_norm=0.5)
            scaler_G.step(opt_G); scaler_G.update()

            opt_D.zero_grad()
            with autocast():
                d_real = D(om, eds * c_mask)
                d_fake_d = D(om, fake_sig.detach() * c_mask)
                loss_D = 0.5 * (F.binary_cross_entropy_with_logits(d_real, torch.full_like(d_real, 0.9)) +
                                 F.binary_cross_entropy_with_logits(d_fake_d, torch.zeros_like(d_fake_d)))
            scaler_D.scale(loss_D).backward(); scaler_D.step(opt_D); scaler_D.update()

        sched_G.step()
        sched_D.step()

        if (epoch + 1) % VAL_INTERVAL == 0 or epoch == epochs - 1:
            val_iou, val_mae = validate_generator(G, val_loader, is_al)
            # history tuple: (epoch, val_iou, val_mae)
            history.append((epoch + 1, val_iou, val_mae)) 
            
            # Evaluate improvement based on val_mae (lower is better)
            if val_mae < best_val_mae - CONVERGENCE_MIN_DELTA:
                best_val_mae = val_mae
                best_val_iou = val_iou
                epoch_of_best = epoch + 1
                checks_since_improve = 0
            else:
                checks_since_improve += 1

    converged = checks_since_improve >= CONVERGENCE_PATIENCE_CHECKS
    return {
        "val_iou": history[-1][1],
        "val_mae": history[-1][2],
        "best_val_iou": best_val_iou,
        "best_val_mae": best_val_mae,
        "epoch_of_best": epoch_of_best,
        "converged": converged,
        "history": history,
    }

def run_grid_search(elem_name, train_files, val_files, epochs, batch_size, num_workers):
    keys = list(GRID.keys())
    combos = list(itertools.product(*[GRID[k] for k in keys]))
    print(f"\n[INFO] {elem_name}: starting grid search with {len(combos)} configurations "
          f"({epochs} epochs per configuration)")

    results = []
    histories = {}
    for i, combo in enumerate(combos):
        config = dict(zip(keys, combo))
        print(f"  [{i+1}/{len(combos)}] {config}")
        out = train_one_config(
            elem_name, config, train_files, val_files, epochs, batch_size, num_workers
        )
        row = {
            **config,
            "val_iou": out["val_iou"],
            "val_mae": out["val_mae"],
            "best_val_iou": out["best_val_iou"],
            "best_val_mae": out["best_val_mae"],
            "epoch_of_best": out["epoch_of_best"],
            "converged": out["converged"],
        }
        results.append(row)
        histories[f"combo_{i+1}"] = {**config, "loss_ratio": list(config["loss_ratio"]), "history": out["history"]}
        status = "converged" if out["converged"] else "NOT converged (still improving at last epoch)"
        print(f"      -> val_mae = {out['val_mae']:.4f} %p, val_iou = {out['val_iou']:.4f} "
              f"(best_mae={out['best_val_mae']:.4f} @ epoch {out['epoch_of_best']}, {status})")

    df = pd.DataFrame(results)

    df["loss_ratio"] = df["loss_ratio"].apply(lambda x: f"{x[0]}:{x[1]}")
    csv_path = os.path.join(GRID_DIR, f"grid_results_{elem_name.lower()}.csv")
    df.to_csv(csv_path, index=False)
    print(f"[SAVED] {elem_name} grid-search results -> {csv_path}")

    history_path = os.path.join(GRID_DIR, f"grid_history_{elem_name.lower()}.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(histories, f, indent=2)
    print(f"[SAVED] {elem_name} per-epoch validation history -> {history_path}")

    n_not_converged = (~df["converged"]).sum()
    if n_not_converged > 0:
        print(f"[WARN] {elem_name}: {n_not_converged}/{len(df)} configuration(s) had not converged "
              f"by epoch {epochs} -- consider raising --epochs or inspecting {history_path}")

    # Change the criterion for selecting the best parameters to the lowest val_mae
    best_row = df.loc[df["val_mae"].idxmin()]
    print(f"[BEST] {elem_name}: {best_row.to_dict()}")
    return best_row.to_dict()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=51,
                    help="Training epochs per grid point (much shorter than full training).")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=12)
    ap.add_argument("--elements", nargs="+", default=DEFAULT_GRID_ELEMENTS, choices=EDS_SUFFIXES.keys(),
                    help="Element maps to search, e.g. --elements Al Fe Cu.")
    args = ap.parse_args()

    splits_path = os.path.join(SAVE_DIR, "splits.json")
    if not os.path.exists(splits_path):
        raise FileNotFoundError(
            f"[ERROR] Dataset split file not found: {splits_path}\n"
            "Create the train/validation/test split before starting the grid search."
        )

    with open(splits_path, encoding="utf-8") as f:
        splits = json.load(f)
    train_files = splits["train"]
    val_files = splits["val"]
    if not train_files or not val_files:
        raise ValueError("[ERROR] The train and validation entries in splits.json must both be non-empty.")

    print(f"[INFO] Selected elements: {', '.join(args.elements)}")
    print(f"[INFO] Loaded dataset split: {splits_path}")
    print(f"[INFO] Dataset split | Train: {len(train_files)} | Validation: {len(val_files)}")

    best_configs = {}
    for elem_name in args.elements:
        best_configs[elem_name] = run_grid_search(
            elem_name, train_files, val_files, args.epochs, args.batch_size, args.num_workers
        )

    best_df = pd.DataFrame.from_dict(best_configs, orient="index").reset_index()
    best_df = best_df.rename(columns={"index": "element"})
    best_csv_path = os.path.join(GRID_DIR, "grid_search_best_params.csv")
    best_df.to_csv(best_csv_path, index=False)
    print(f"\n[SAVED] Best hyperparameters -> {best_csv_path}")

if __name__ == '__main__':
    main()