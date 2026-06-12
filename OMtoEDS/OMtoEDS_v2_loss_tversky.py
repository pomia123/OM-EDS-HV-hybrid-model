## 6개 원소 별도 학습
## eds binary

import os, random, cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
import json
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torchvision.models import resnet34, ResNet34_Weights

import logging
logging.getLogger("albumentations.check_version").setLevel(logging.ERROR)
import albumentations as A

# ──────────────────────────────────────────────
# 0. Seed & Device
# ──────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def imread_korean(path, mode=cv2.IMREAD_COLOR):
    try:
        n = np.fromfile(path, np.uint8)
        return cv2.imdecode(n, mode)
    except:
        return None

# ──────────────────────────────────────────────
# 1. 하이퍼파라미터
# ──────────────────────────────────────────────
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

# 원소별 pixel loss 가중치 - 희소 원소 강조
LAMBDAS_PIX = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0]

# GAN loss 가중치 (pixel loss 대비 낮게)
LAMBDA_GAN = 1.0

best_ious = {key: 0.0 for key in elem_keys}

# ──────────────────────────────────────────────
# 2. 전처리 및 Augmentation
# ──────────────────────────────────────────────
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
    'mask_ref': 'mask', 'map_ref': 'mask',
    'eds0': 'mask',
})

color_aug = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    A.RandomGamma(p=0.2),
])

# ──────────────────────────────────────────────
# 3. Dataset
# ──────────────────────────────────────────────
class MetallurgyDataset(Dataset):
    def __init__(self, file_list, target_idx, is_train=True):
        self.file_names = file_list
        self.target_idx = target_idx  # 학습할 원소 인덱스 (0~5)
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

        # 해당 원소 하나만 로드
        suf = EDS_SUFFIXES[i]
        raw = imread_korean(os.path.join(EDS_DIR, f"{base_name}_{suf}.png"), 0)
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

        # Normalize to [-1, 1] for GAN
        om_t    = (torch.from_numpy(om_c).permute(2, 0, 1).float() / 127.5) - 1.0
        eds_t   = torch.from_numpy(eds_crop).float().unsqueeze(0) / 255.0  # (1, H, W)
        align_t = (torch.from_numpy(mask_c) > 0).float()
        almap_t = (torch.from_numpy(map_c)  > 0).float()
        valid_mask = (align_t * almap_t).unsqueeze(0)  # (1, H, W)
        align_only = align_t.unsqueeze(0)               # (1, H, W) Al용

        return om_t, eds_t, valid_mask, align_only, base_name


# ──────────────────────────────────────────────
# 4. Model
# ──────────────────────────────────────────────

# ── Attention modules (CBAM) ──
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

# ── Generator: ResNet34 encoder + CBAM U-Net decoder ──
class Generator(nn.Module):
    def __init__(self, out_ch=NUM_CLASSES, pretrained=True):
        super().__init__()
        bb = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)

        self.enc0 = nn.Sequential(bb.conv1, bb.bn1, bb.relu)   # /2  64ch
        self.pool = bb.maxpool                                   # /4
        self.enc1 = bb.layer1   #  64ch
        self.enc2 = bb.layer2   # 128ch
        self.enc3 = bb.layer3   # 256ch
        self.enc4 = bb.layer4   # 512ch

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
        return out   # logits (no sigmoid) — sigmoid applied in loss/inference


# ── Discriminator: PatchGAN ──
class Discriminator(nn.Module):
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


# ──────────────────────────────────────────────
# 5. Loss functions
# ──────────────────────────────────────────────

# 원소별 Tversky alpha/beta 설정
# alpha: FP 패널티, beta: FN 패널티
# Sr/Cu/Fe/Mg (희소, 1%): FN 패널티 강하게
# Si (10%): 균형
# Al (80%): FP 패널티 강하게
TVERSKY_PARAMS = {
    'mg': (0.3, 0.7),
    'al': (0.7, 0.3),
    'si': (0.5, 0.5),
    'cu': (0.3, 0.7),
    'fe': (0.3, 0.7),
    'sr': (0.3, 0.7),
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
    pred_bin   = (torch.sigmoid(pred_logit) > threshold).float() * mask
    target_bin = (target > 0.5).float() * mask
    inter = (pred_bin * target_bin).sum(dim=(2, 3))
    union = (pred_bin + target_bin).clamp(0, 1).sum(dim=(2, 3))
    return (inter + 1e-8) / (union + 1e-8)


# ──────────────────────────────────────────────
# 6. Main
# ──────────────────────────────────────────────
if __name__ == '__main__':

    # ── Data split ──
    if not os.path.exists(OM_DIR):
        print(f"🚨 에러: 경로를 찾을 수 없습니다 -> {OM_DIR}")
        exit()

    # 1. 먼저 폴더 내 파일 리스트를 가져옵니다 (이게 빠져서 NameError 발생)
    raw_files = [f for f in os.listdir(OM_DIR) if f.lower().endswith(('.png', '.bmp'))]
    
    # 2. 가져온 파일명에서 확장자를 떼고 ID 리스트를 만듭니다
    all_f = sorted(list(set([f.rsplit('.', 1)[0] for f in raw_files])))

    print(f"🔍 총 {len(all_f)}개의 데이터 세트를 발견했습니다.")

    if len(all_f) == 0:
        print(f"🚨 에러: {OM_DIR} 폴더에 .png 또는 .bmp 파일이 하나도 없습니다.")
        exit()

    # 나머지 분할 로직
    random.Random(SEED).shuffle(all_f)
    n_train = int(len(all_f) * 0.8)
    n_val   = int(len(all_f) * 0.1)
    train_files = all_f[:n_train]
    val_files   = all_f[n_train:n_train + n_val]
    test_files  = all_f[n_train + n_val:]
    
    print(f"✅ 데이터 분할 완료: Train {len(train_files)} | Val {len(val_files)} | Test {len(test_files)}")

    
    os.makedirs(SAVE_DIR, exist_ok=True)
    with open(os.path.join(SAVE_DIR, 'splits.json'), 'w') as f:
        json.dump({'train': train_files, 'val': val_files, 'test': test_files}, f, indent=2)

    num_workers = 16 if os.name != 'nt' else 0

    # ── 원소별 개별 학습 ──
    for elem_idx, (key, elem_name) in enumerate(zip(elem_keys, ELEM_NAMES)):
        print(f"\n{'='*60}")
        print(f"🔬 [{elem_idx+1}/{len(ELEM_NAMES)}] 원소 학습 시작: {elem_name} ({key})")
        print(f"{'='*60}")

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

        # ── Initial data check (첫 원소만) ──
        if elem_idx == 0:
            print("\n🧐 Checking initial training data...")
            s_om, s_eds, s_mask, s_align, s_fname = next(iter(train_loader))
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(((s_om[0].permute(1,2,0).numpy() + 1) / 2).clip(0, 1))
            axes[0].set_title(f"OM Input\n({s_fname[0]})", fontsize=14); axes[0].axis('off')
            mask_vis = s_align[0, 0] if elem_name == "Al" else s_mask[0, 0]
            axes[1].imshow((s_eds[0, 0] * mask_vis).numpy(), cmap='gray')
            axes[1].set_title(f"Target {elem_name}", fontsize=14); axes[1].axis('off')
            axes[2].imshow(s_mask[0, 0].numpy(), cmap='gray')
            axes[2].set_title("Valid Mask", fontsize=14); axes[2].axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(SAVE_DIR, 'data_check.png'), dpi=100)
            plt.close()
            print(f"  Data check saved → {SAVE_DIR}/data_check.png")

        # ── Model & Optimizer (원소마다 새로 초기화) ──
        LR_INIT = 0.0002
        is_al = (elem_name == "Al")

        G = Generator(out_ch=1, pretrained=True).to(device)
        D = Discriminator(in_ch=3 + 1).to(device)

        if torch.cuda.device_count() > 1:
            print(f"🔥 {torch.cuda.device_count()} GPUs → DataParallel")
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

        print(f"\n🚀 Training Start | Device: {device} | λ_pix={lam_pix}")
        print(f"{'G_GAN':>8} {'D_Loss':>8} {'G_Pix':>8} {'Train_IoU':>10} {'Val_IoU':>10}")
        print("-" * 50)

        for epoch in range(EPOCHS):
            G.train(); D.train()
            run_g_gan = run_d_loss = run_g_pix = run_iou = 0.0

            for om, eds, v_mask, align_mask, _ in train_loader:
                om         = om.to(device)
                eds        = eds.to(device)           # (B, 1, H, W)
                v_mask     = v_mask.to(device)        # (B, 1, H, W)
                align_mask = align_mask.to(device)    # (B, 1, H, W)

                c_mask = align_mask if is_al else v_mask

                # ── Generator update ──
                opt_G.zero_grad()
                with autocast():
                    fake_logit = G(om)                          # (B, 1, H, W)
                    fake_sig   = torch.sigmoid(fake_logit)

                    d_fake = D(om, fake_sig * c_mask)
                    g_gan  = F.binary_cross_entropy_with_logits(d_fake, torch.ones_like(d_fake))

                    loss_pix = combined_loss(fake_logit, eds, c_mask, elem_key=key)
                    loss_G   = LAMBDA_GAN * g_gan + lam_pix * loss_pix

                scaler_G.scale(loss_G).backward()
                nn.utils.clip_grad_norm_(G.parameters(), max_norm=0.5)
                scaler_G.step(opt_G); scaler_G.update()

                # ── Discriminator update ──
                opt_D.zero_grad()
                with autocast():
                    d_real   = D(om, eds * c_mask)
                    d_fake_d = D(om, fake_sig.detach() * c_mask)
                    loss_D   = 0.5 * (
                        F.binary_cross_entropy_with_logits(d_real,   torch.full_like(d_real, 0.9)) +
                        F.binary_cross_entropy_with_logits(d_fake_d, torch.zeros_like(d_fake_d))
                    )
                scaler_D.scale(loss_D).backward()
                scaler_D.step(opt_D); scaler_D.update()

                with torch.no_grad():
                    run_iou += calculate_iou(fake_logit, eds, c_mask).mean().item()
                run_g_gan += g_gan.item()
                run_g_pix += loss_pix.item()
                run_d_loss += loss_D.item()

            sched_G.step(); sched_D.step()

            # ── Validation every 5 epochs ──
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
                saved_mark  = ""
                if avg_val_iou > best_iou:
                    best_iou = avg_val_iou
                    sd = G.module.state_dict() if hasattr(G, 'module') else G.state_dict()
                    torch.save(sd, os.path.join(MODEL_DIR, f"best_model_{key}_{epoch}.pth"))
                    saved_mark = " 🌟"

                print(f"[Epoch {epoch:4d}/{EPOCHS}] "
                      f"{run_g_gan/n:>8.4f} "
                      f"{run_d_loss/n:>8.4f} "
                      f"{run_g_pix/n:>8.4f} "
                      f"{run_iou/n:>10.4f} "
                      f"{avg_val_iou:>10.4f}"
                      f"{saved_mark}")

            # ── Save prediction images every 50 epochs ──
            if epoch % 50 == 0:
                G.eval()
                with torch.no_grad():
                    sample_om, sample_eds, sample_mask, sample_align, sample_name = \
                        next(iter(val_loader))
                    sample_om = sample_om.to(device)
                    pred_sig  = torch.sigmoid(G(sample_om)).cpu()

                # 마스크 정의 (is_al 여부에 따라 선택)
                mask_vis = sample_align[0, 0] if is_al else sample_mask[0, 0]
                mask_vis_np = mask_vis.numpy()

                # 1. OM 이미지 처리 및 저장
                om_img = ((sample_om[0].cpu().permute(1, 2, 0).numpy() + 1) / 2).clip(0, 1)
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.imshow(om_img)
                ax.axis('off')
                plt.savefig(os.path.join(SAVE_DIR, f'pred_{key}_epoch_{epoch:04d}_OM.png'), dpi=300, bbox_inches='tight', pad_inches=0)
                plt.close()

                # 2. GT EDS 이미지 처리 (마스크가 검정색인 곳은 확실히 검정색 처리) 및 저장
                gt_img = sample_eds[0, 0].numpy() * mask_vis_np
                fig, ax = plt.subplots(figsize=(4, 4))
                # vmin/vmax를 지정하여 마스킹된 0 영역이 확실히 cmap의 가장 어두운 색(검정)으로 표현되도록 합니다.
                ax.imshow(gt_img, cmap='gray', vmin=0, vmax=max(1.0, gt_img.max()))
                ax.axis('off')
                plt.savefig(os.path.join(SAVE_DIR, f'pred_{key}_epoch_{epoch:04d}_GT.png'), dpi=300, bbox_inches='tight', pad_inches=0)
                plt.close()

                # 3. Pred 이미지 처리 및 저장
                pred_img = pred_sig[0, 0].numpy() * mask_vis_np
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.imshow(pred_img, cmap='gray', vmin=0, vmax=max(1.0, pred_img.max()))
                ax.axis('off')
                plt.savefig(os.path.join(SAVE_DIR, f'pred_{key}_epoch_{epoch:04d}_Pred.png'), dpi=300, bbox_inches='tight', pad_inches=0)
                plt.close()

        # ── 원소별 최종 모델 저장 ──
        final_path = os.path.join(MODEL_DIR, f"last_model_{key}.pth")
        sd = G.module.state_dict() if isinstance(G, nn.DataParallel) else G.state_dict()
        torch.save(sd, final_path)
        print(f"\n✅ {elem_name} 학습 완료. Best IoU: {best_iou:.4f} | Last model → {final_path}")
        best_ious[key] = best_iou

    print(f"\n{'='*60}")
    print("🎉 전체 학습 완료")
    print("\nBest IoU per element:")
    for k, v in best_ious.items():
        print(f"  {k.upper()}: {v:.4f}")