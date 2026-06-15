"""
Deep Learning Pipeline for OM-to-EDS Binary Map Prediction.

This module provides a comprehensive pipeline to train a Pix2Pix-based 
Generative Adversarial Network (GAN). It incorporates Convolutional Block 
Attention Modules (CBAM) and Tversky/Focal loss functions to accurately 
predict element-specific Energy Dispersive X-ray Spectroscopy (EDS) 
binary maps from Optical Microscopy (OM) images.
"""


import os
import random
import cv2
import json
import warnings
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torchvision.models import resnet34, ResNet34_Weights

# Suppress non-critical warnings for clean execution
warnings.filterwarnings("ignore")
logging.getLogger("albumentations.check_version").setLevel(logging.ERROR)
import albumentations as A

matplotlib.use('Agg')

# =============================================================================
# 1. Global Configuration and Utilities
# =============================================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def imread_korean(path, mode=cv2.IMREAD_COLOR):
    """Helper function to decode image paths containing non-ASCII/Korean characters."""
    try:
        n = np.fromfile(path, np.uint8)
        return cv2.imdecode(n, mode)
    except Exception:
        return None

# =============================================================================
# 2. Dataset Paths and Hyperparameters
# =============================================================================

OM_DIR   = r'.\data\OM'
EDS_DIR  = r'.\data\EDS'
MASK_DIR = r'.\data\MASK'
MAP_DIR  = r'.\data\MAP'
SAVE_DIR = r'.\result\tversky'
MODEL_DIR = os.path.join(SAVE_DIR, 'models_tversky')
os.makedirs(MODEL_DIR, exist_ok=True)

CROP_W, CROP_H = 512, 512
BATCH_SIZE = 32
EPOCHS     = 1000

EDS_SUFFIXES = ["01", "02", "03", "06", "07", "09"]
ELEM_NAMES   = ["Mg", "Al", "Si", "Cu", "Fe", "Sr"]
NUM_CLASSES  = 1
elem_keys    = [n.lower() for n in ELEM_NAMES]

LAMBDAS_PIX = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
LAMBDA_GAN = 1.0

best_ious = {key: 0.0 for key in elem_keys}

# =============================================================================
# 3. Data Preprocessing & Augmentation Pipelines
# =============================================================================

def preprocess_om_color(om_bgr):
    """Converts BGR Optical Microscopy images to RGB color space."""
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
# 4. Dataset Definition
# =============================================================================

class MetallurgyDataset(Dataset):
    """
    Custom Dataset class for loading and augmenting paired OM and EDS images.
    Extracts element-specific spatial distributions via dynamic thresholding.
    """
    def __init__(self, file_list, target_idx, is_train=True):
        self.file_names = file_list
        self.target_idx = target_idx  # Target element index domain (0~5)
        self.is_train = is_train

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        base_name = self.file_names[idx]
        i = self.target_idx

        om_raw  = imread_korean(os.path.join(OM_DIR,   f"{base_name}.png"), 1)
        om_rgb  = preprocess_om_color(om_raw)
        g_mask  = imread_korean(os.path.join(MASK_DIR, f"{base_name}.png"),  0)
        g_map   = imread_korean(os.path.join(MAP_DIR,  f"{base_name}.png"), 0)
        
        h, w = g_mask.shape
        if self.is_train:
            sy = random.randint(0, h - CROP_H)
            sx = random.randint(0, w - CROP_W)
        else:
            sy = (h - CROP_H) // 2
            sx = (w - CROP_W) // 2

        om_c   = om_rgb[sy:sy+CROP_H, sx:sx+CROP_W]
        mask_c = g_mask[sy:sy+CROP_H, sx:sx+CROP_W]
        map_c  = g_map [sy:sy+CROP_H, sx:sx+CROP_W]

        suf = EDS_SUFFIXES[i]
        raw = imread_korean(os.path.join(EDS_DIR, f"{base_name}_{suf}.png"), 0)
        
        # Element-Specific Background Thresholding Strategy
        if ELEM_NAMES[i] == "Al":
            th = np.percentile(raw, 20)
        elif ELEM_NAMES[i] == "Si":
            th = np.percentile(raw, 90)
        else:
            th = np.percentile(raw, 99)
        
        c   = raw[sy:sy+CROP_H, sx:sx+CROP_W]
        eds_crop = (c > th).astype(np.uint8) * 255

        if self.is_train:
            augmented = geom_aug(image=om_c, mask_ref=mask_c, map_ref=map_c, eds0=eds_crop)
            om_c     = augmented["image"]
            mask_c   = augmented["mask_ref"]
            map_c    = augmented["map_ref"]
            eds_crop = augmented["eds0"]
            om_c = color_aug(image=om_c)["image"]

        # Tensor Normalization
        om_t    = (torch.from_numpy(om_c).permute(2, 0, 1).float() / 127.5) - 1.0
        eds_t   = torch.from_numpy(eds_crop).float().unsqueeze(0) / 255.0
        align_t = (torch.from_numpy(mask_c) > 0).float()
        almap_t = (torch.from_numpy(map_c)  > 0).float()
        
        valid_mask = (align_t * almap_t).unsqueeze(0)  
        align_only = align_t.unsqueeze(0)              

        return om_t, eds_t, valid_mask, align_only, base_name


# =============================================================================
# 5. Model Architecture Framework
# =============================================================================

class ChannelAttention(nn.Module):
    """Computes channel attention weights using global average and max pooling."""
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
    """Computes spatial attention weights prioritizing informational regions."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sig  = nn.Sigmoid()
    
    def forward(self, x):
        avg = x.mean(1, keepdim=True)
        mx, _ = x.max(1, keepdim=True)
        return x * self.sig(self.conv(torch.cat([avg, mx], 1)))

class CBAM(nn.Module):
    """Convolutional Block Attention Module sequentially applying Channel and Spatial attention."""
    def __init__(self, ch):
        super().__init__()
        self.ca = ChannelAttention(ch)
        self.sa = SpatialAttention()
    def forward(self, x):
        return self.sa(self.ca(x))

class ConvBnRelu(nn.Sequential):
    """Standard Convolutional sequence mapping block."""
    def __init__(self, in_ch, out_ch):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

class DecoderBlock(nn.Module):
    """Upsampling block utilized within the Generator architecture."""
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
    """ResNet34-based U-Net Generator predicting pixel-wise material properties."""
    def __init__(self, out_ch=NUM_CLASSES, pretrained=True):
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

class Discriminator(nn.Module):
    """PatchGAN Discriminator classifying structural coherence of OM-EDS pairs."""
    def __init__(self, in_ch=3 + NUM_CLASSES):
        super().__init__()
        def dl(i, o, norm=True):
            layers = [nn.Conv2d(i, o, 4, 2, 1)]
            if norm:
                layers.append(nn.InstanceNorm2d(o))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)
        self.model = nn.Sequential(
            dl(in_ch, 64,  norm=False),
            dl(64,  128),
            dl(128, 256),
            dl(256, 512),
            nn.Conv2d(512, 1, 3, 1, 1),
        )

    def forward(self, om, eds):
        return self.model(torch.cat([om, eds], 1))


# =============================================================================
# 6. Loss Optimization Functions
# =============================================================================

# Hardcoded domain-specific parameter calibration for class imbalance handling
TVERSKY_PARAMS = {
    'mg': (0.3, 0.7), 'al': (0.7, 0.3), 'si': (0.5, 0.5),
    'cu': (0.3, 0.7), 'fe': (0.3, 0.7), 'sr': (0.3, 0.7),
}

def focal_loss(p_logit, t, valid_mask, gamma=2.0, alpha_pos=0.8):
    bce = F.binary_cross_entropy_with_logits(p_logit, t, reduction='none')
    p = torch.sigmoid(p_logit)
    pt = torch.where(t > 0.5, p, 1 - p)
    at = torch.where(t > 0.5, torch.full_like(t, alpha_pos), torch.full_like(t, 1 - alpha_pos))
    loss = at * (1 - pt) ** gamma * bce * valid_mask
    return loss.sum() / (valid_mask.sum() + 1e-8)

def tversky_loss(p_logit, t, valid_mask, alpha=0.3, beta=0.7, eps=1e-6):
    p = torch.sigmoid(p_logit) * valid_mask
    t = t * valid_mask
    inter = (p * t).sum(dim=(2, 3))
    fp    = (p * (1 - t)).sum(dim=(2, 3))
    fn    = ((1 - p) * t).sum(dim=(2, 3))
    return (1 - (inter + eps) / (inter + alpha * fp + beta * fn + eps)).mean()

def combined_loss(p_logit, t, valid_mask, elem_key, focal_w=0.5, tversky_w=0.5):
    alpha, beta = TVERSKY_PARAMS[elem_key]
    return (focal_w   * focal_loss(p_logit, t, valid_mask) +
            tversky_w * tversky_loss(p_logit, t, valid_mask, alpha=alpha, beta=beta))

def calculate_iou(pred_logit, target, mask, threshold=0.5):
    """Calculates Intersection over Union (IoU) strictly within the validated mask domain."""
    pred_bin   = (torch.sigmoid(pred_logit) > threshold).float() * mask
    target_bin = (target > 0.5).float() * mask
    inter = (pred_bin * target_bin).sum(dim=(2, 3))
    union = (pred_bin + target_bin).clamp(0, 1).sum(dim=(2, 3))
    return (inter + 1e-8) / (union + 1e-8)

# =============================================================================
# 7. Core Training Pipeline Execution
# =============================================================================

if __name__ == '__main__':
    """
    Main orchestration routine coordinating data partition, sequential model 
    training per chemical element, and empirical validation metric tracking.
    """

    if not os.path.exists(OM_DIR):
        print(f"[ERROR] Target OM repository path not found: {OM_DIR}")
        exit()

    raw_files = [f for f in os.listdir(OM_DIR) if f.lower().endswith(('.png', '.bmp'))]
    all_f = sorted(list(set([f.rsplit('.', 1)[0] for f in raw_files])))

    print(f"[INFO] Discovered {len(all_f)} microstructure data validation pairs.")
    if len(all_f) == 0:
        print(f"[ERROR] No valid source imagery detected inside target paths.")
        exit()
    
    # Train / Validation / Test Split
    random.Random(SEED).shuffle(all_f)
    n_train = int(len(all_f) * 0.8)
    n_val   = int(len(all_f) * 0.1)
    train_files = all_f[:n_train]
    val_files   = all_f[n_train:n_train + n_val]
    test_files  = all_f[n_train + n_val:]
    
    print(f"[INFO] Partition completed | Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")

    os.makedirs(SAVE_DIR, exist_ok=True)
    with open(os.path.join(SAVE_DIR, 'splits.json'), 'w') as f:
        json.dump({'train': train_files, 'val': val_files, 'test': test_files}, f, indent=2)

    num_workers = 16 if os.name != 'nt' else 0

    # Sequential execution targeting unique elemental maps
    for elem_idx, (key, elem_name) in enumerate(zip(elem_keys, ELEM_NAMES)):
        print(f"\n============================================================")
        print(f"[INFO] [{elem_idx+1}/{len(ELEM_NAMES)}] Executing Phase Allocation: {elem_name} ({key})")
        print(f"============================================================")

        train_loader = DataLoader(
            MetallurgyDataset(train_files, target_idx=elem_idx, is_train=True),
            batch_size=BATCH_SIZE, shuffle=True,
            num_workers=num_workers, pin_memory=True,
            persistent_workers=(num_workers > 0),
        )
        val_loader = DataLoader(
            MetallurgyDataset(val_files, target_idx=elem_idx, is_train=False),
            batch_size=4, shuffle=False,
            num_workers=max(num_workers // 2, 0),
        )

        # Baseline Data Integrity Verification
        if elem_idx == 0:
            print("\n[INFO] Validating data integrity pipelines...")
            s_om, s_eds, s_mask, s_align, s_fname = next(iter(train_loader))
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(((s_om[0].permute(1,2,0).numpy() + 1) / 2).clip(0, 1))
            axes[0].set_title(f"OM Input\n({s_fname[0]})", fontsize=14)
            axes[0].axis('off')

            mask_vis = s_align[0, 0] if elem_name == "Al" else s_mask[0, 0]
            axes[1].imshow((s_eds[0, 0] * mask_vis).numpy(), cmap='gray')
            axes[1].set_title(f"Target {elem_name}", fontsize=14)
            axes[1].axis('off')

            axes[2].imshow(s_mask[0, 0].numpy(), cmap='gray')
            axes[2].set_title("Valid Mask", fontsize=14)
            axes[2].axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(SAVE_DIR, 'data_check.png'), dpi=100)
            plt.close()
            print(f"[INFO] Diagnostics verified. Artifact saved to: {SAVE_DIR}/data_check.png")

        # Network Initialization
        LR_INIT = 0.0002
        is_al = (elem_name == "Al")

        G = Generator(out_ch=1, pretrained=True).to(device)
        D = Discriminator(in_ch=3 + 1).to(device)

        if torch.cuda.device_count() > 1:
            print(f"[INFO] Multi-GPU environment detected. Scaling to DataParallel ({torch.cuda.device_count()} GPUs)")
            G = nn.DataParallel(G)
            D = nn.DataParallel(D)

        opt_G = torch.optim.Adam(G.parameters(), LR_INIT, betas=(0.5, 0.999))
        opt_D = torch.optim.Adam(D.parameters(), LR_INIT, betas=(0.5, 0.999))

        def lr_lambda(epoch):
            return 1.0 if epoch < 100 else max(0.0, 1.0 - (epoch - 100) / (EPOCHS - 100 + 1))

        sched_G = torch.optim.lr_scheduler.LambdaLR(opt_G, lr_lambda)
        sched_D = torch.optim.lr_scheduler.LambdaLR(opt_D, lr_lambda)

        scaler_G = GradScaler()
        scaler_D = GradScaler()

        best_iou = 0.0
        lam_pix  = LAMBDAS_PIX[elem_idx]

        print(f"\n[INFO] Training loop initialized | Device: {device} | λ_pix={lam_pix}")
        print(f"{'G_GAN':>8} {'D_Loss':>8} {'G_Pix':>8} {'Train_IoU':>10} {'Val_IoU':>10}")
        print("-" * 50)

        # Optimization Iterations
        for epoch in range(EPOCHS):
            G.train()
            D.train()
            run_g_gan = run_d_loss = run_g_pix = run_iou = 0.0

            for om, eds, v_mask, align_mask, _ in train_loader:
                om         = om.to(device)
                eds        = eds.to(device)         
                v_mask     = v_mask.to(device)      
                align_mask = align_mask.to(device)    

                c_mask = align_mask if is_al else v_mask

                # -----------------------------------------------------------------
                # Generator Update
                # -----------------------------------------------------------------
                opt_G.zero_grad()
                with autocast():
                    fake_logit = G(om)                         
                    fake_sig   = torch.sigmoid(fake_logit)

                    d_fake = D(om, fake_sig * c_mask)
                    g_gan  = F.binary_cross_entropy_with_logits(d_fake, torch.ones_like(d_fake))

                    loss_pix = combined_loss(fake_logit, eds, c_mask, elem_key=key)
                    loss_G   = LAMBDA_GAN * g_gan + lam_pix * loss_pix

                scaler_G.scale(loss_G).backward()
                nn.utils.clip_grad_norm_(G.parameters(), max_norm=0.5)
                scaler_G.step(opt_G)
                scaler_G.update()

                # -----------------------------------------------------------------
                # Discriminator Update
                # -----------------------------------------------------------------
                opt_D.zero_grad()
                with autocast():
                    d_real   = D(om, eds * c_mask)
                    d_fake_d = D(om, fake_sig.detach() * c_mask)
                    loss_D   = 0.5 * (
                        F.binary_cross_entropy_with_logits(d_real,   torch.full_like(d_real, 0.9)) +
                        F.binary_cross_entropy_with_logits(d_fake_d, torch.zeros_like(d_fake_d))
                    )

                scaler_D.scale(loss_D).backward()
                scaler_D.step(opt_D)
                scaler_D.update()

                with torch.no_grad():
                    run_iou += calculate_iou(fake_logit, eds, c_mask).mean().item()
                run_g_gan += g_gan.item()
                run_g_pix += loss_pix.item()
                run_d_loss += loss_D.item()

            sched_G.step()
            sched_D.step()

            # -----------------------------------------------------------------
            # Validation & Checkpoint Strategy
            # -----------------------------------------------------------------
            if epoch % 5 == 0:
                n = len(train_loader)
                G.eval()
                val_iou_sum = 0.0
                with torch.no_grad():
                    for om_v, eds_v, v_mask_v, align_v, _ in val_loader:
                        om_v   = om_v.to(device)
                        eds_v  = eds_v.to(device)
                        m_v    = align_v.to(device) if is_al else v_mask_v.to(device)
                        logit_v = G(om_v)
                        val_iou_sum += calculate_iou(logit_v, eds_v, m_v).mean().item()

                avg_val_iou = val_iou_sum / len(val_loader)
                saved_mark = ""
                if avg_val_iou > best_iou:
                    best_iou = avg_val_iou
                    sd = G.module.state_dict() if hasattr(G, 'module') else G.state_dict()
                    torch.save(sd, os.path.join(MODEL_DIR, f"best_model_{key}_{epoch}.pth"))
                    saved_mark = " * [Saved Checkpoint]"

                print(f"[Epoch {epoch:4d}/{EPOCHS}] "
                      f"{run_g_gan/n:>8.4f} "
                      f"{run_d_loss/n:>8.4f} "
                      f"{run_g_pix/n:>8.4f} "
                      f"{run_iou/n:>10.4f} "
                      f"{avg_val_iou:>10.4f}"
                      f"{saved_mark}")

            # -----------------------------------------------------------------
            # Periodic Artifact Generation for Visual Evaluation
            # -----------------------------------------------------------------
            if epoch % 50 == 0:
                G.eval()
                with torch.no_grad():
                    sample_om, sample_eds, sample_mask, sample_align, sample_name = next(iter(val_loader))
                    sample_om = sample_om.to(device)
                    pred_sig  = torch.sigmoid(G(sample_om)).cpu()

                mask_vis = sample_align[0, 0] if is_al else sample_mask[0, 0]
                mask_vis_np = mask_vis.numpy()

                # 1. # Output OM Macrostructure
                om_img = ((sample_om[0].cpu().permute(1, 2, 0).numpy() + 1) / 2).clip(0, 1)
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.imshow(om_img)
                ax.axis('off')
                plt.savefig(os.path.join(SAVE_DIR, f'pred_{key}_epoch_{epoch:04d}_OM.png'), dpi=300, bbox_inches='tight', pad_inches=0)
                plt.close()

                # 2. Output Ground Truth Mask
                gt_img = sample_eds[0, 0].numpy() * mask_vis_np
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.imshow(gt_img, cmap='gray', vmin=0, vmax=max(1.0, gt_img.max()))
                ax.axis('off')
                plt.savefig(os.path.join(SAVE_DIR, f'pred_{key}_epoch_{epoch:04d}_GT.png'), dpi=300, bbox_inches='tight', pad_inches=0)
                plt.close()

                # 3. Output Predicted Map
                pred_img = pred_sig[0, 0].numpy() * mask_vis_np
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.imshow(pred_img, cmap='gray', vmin=0, vmax=max(1.0, pred_img.max()))
                ax.axis('off')
                plt.savefig(os.path.join(SAVE_DIR, f'pred_{key}_epoch_{epoch:04d}_Pred.png'), dpi=300, bbox_inches='tight', pad_inches=0)
                plt.close()

        # Phase Completion & Final Weights Persistence
        final_path = os.path.join(MODEL_DIR, f"last_model_{key}.pth")
        sd = G.module.state_dict() if isinstance(G, nn.DataParallel) else G.state_dict()
        torch.save(sd, final_path)

        print(f"\n[SUCCESS] Model optimization for {elem_name} completed successfully.")
        print(f"[INFO] Benchmark IoU: {best_iou:.4f} | Artifact registered at: {final_path}")
        best_ious[key] = best_iou

    # Global Process Completion Summary
    print(f"\n============================================================")
    print("[SUCCESS] Comprehensive deep learning spatial mapping pipeline finalized.")
    print("\n[INFO] Global Benchmark Synthesis (Best Validation IoU):")
    for k, v in best_ious.items():
        print(f"  {k.upper()}: {v:.4f}")
