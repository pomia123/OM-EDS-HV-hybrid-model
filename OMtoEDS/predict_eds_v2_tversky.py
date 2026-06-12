# -*- coding: utf-8 -*-
"""
New Sample Evaluation & Integration Mapping Script (Fixed Formula Mode)
- 실측 정답(GT) 파일 없이, 사용자가 입력한 단일 {base} 데이터에 대해 'last_model' 추론만 수행합니다.
- 평가지표: 모든 원소의 분모는 전체 영역(512x512)으로 고정합니다.
  * Al 분자: 이미지 전체 영역에서의 예측 픽셀 수
  * 나머지 분자: MAP 유효 영역 내부로 제한된 예측 픽셀 수
- 시각화: 가시성을 위해 MAP 파일 유효 영역 내부에만 다중 예측 원소들을 오버랩하여 고해상도 Pred 맵을 생성합니다.
"""

import os, csv
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

# ──────────────────────────────────────────────
# [설정] 여기를 수정하여 새로운 이미지를 테스트하세요
# ──────────────────────────────────────────────
NEW_BASE_NAME = "0La-4_x500_39"

OM_DIR   = r'.\pred_data\OM_hv'
MAP_DIR  = r'.\pred_data\MAP_hv'
RESULT_DIR = r'.\pred_data\result'

MODEL_DIR  = os.path.join(RESULT_DIR, 'models_tversky')
OUTPUT_DIR = os.path.join(RESULT_DIR, 'new_sample_inference')
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALL_ELEMS   = ["Al", "Si", "Mg", "Cu", "Fe", "Sr"]
PREC_ELEMS  = ["Mg", "Cu", "Fe", "Sr"]
THRESHOLD   = 0.5
CROP_W, CROP_H = 512, 512

# matplotlib 글로벌 폰트 규격 통일
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial'] + plt.rcParams['font.sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 원소별 논문용 고대조 보색 계열 색상 (RGB 규격)
ELEM_COLORS_RGB = {
    "Al": (215, 205, 190),   # soft beige-gray
    "Si": (235, 190,  70),   # muted gold
    "Mg": (  0, 200, 255),   # cyan
    "Cu": (150,  70, 255),   # vivid purple
    "Fe": (220,  40,  40),   # crimson red
    "Sr": ( 80, 230,  80),   # lime green
}

# ──────────────────────────────────────────────
# 데이터 로드 및 정중앙 크롭 기능
# ──────────────────────────────────────────────
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
        raise FileNotFoundError(f"🚨 기초 파일 로드 실패. 파일명과 경로를 확인하세요.\nOM: {om_path}\nMAP: {map_path}")

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

# ──────────────────────────────────────────────
# 네트워크 아키텍처 (Generator 정의)
# ──────────────────────────────────────────────
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

# ──────────────────────────────────────────────
# 오버랩 맵 생성용 헬퍼 함수
# ──────────────────────────────────────────────
def create_multi_elem_overlay(om_bgr, bins, valid_mask_np, elem_names, elem_alpha=0.6):
    om_gray  = cv2.cvtColor(om_bgr, cv2.COLOR_BGR2GRAY)
    om_faded = cv2.addWeighted(om_gray, 0.4, np.full_like(om_gray, 255), 0.6, 0)
    canvas   = cv2.cvtColor(om_faded, cv2.COLOR_GRAY2RGB).astype(np.float32)

    for en in elem_names:
        # 💡 [가시성 보정] 면적 수치 계산 공식과 상관없이, 이미지에 시각화할 때는 전부 MAP 유효 영역 내로 제한
        pred = (bins[en] > 0) & (valid_mask_np > 0)
        r, g, b = ELEM_COLORS_RGB[en]
        color = np.array([r, g, b], dtype=np.float32)
        canvas[pred] = (1.0 - elem_alpha) * canvas[pred] + elem_alpha * color

    return canvas.clip(0, 255).astype(np.uint8)

def save_overlay_plot(overlay_img, elem_names, legend_ncol, filename):
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

# ──────────────────────────────────────────────
# 오직 last_model만 확정 탐색하는 로더
# ──────────────────────────────────────────────
def load_last_model(elem_name, device):
    key = elem_name.lower()
    path = os.path.join(MODEL_DIR, f"last_model_{key}.pth")
    
    if not os.path.exists(path):
        return None

    print(f"🔓 Last 모델 로드 완료 [{elem_name}]: {os.path.basename(path)}")
    G = Generator(out_ch=1).to(device)
    sd = torch.load(path, map_location=device)
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    G.load_state_dict(sd)
    return G

# ──────────────────────────────────────────────
# 메인 연산 루프
# ──────────────────────────────────────────────
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Inference Device: {device}")
    print(f"Target Sample ID: {NEW_BASE_NAME}\n")

    # 1. 데이터 로드 및 정중앙 크롭 수행
    om_t, valid_mask_t, valid_mask_np, om_raw_crop = load_and_center_crop_data(NEW_BASE_NAME)
    
    # 💡 [요청 반영] 분모는 모든 원소 공통으로 전체 영역(512x512) 고정
    total_image_pixels = float(CROP_W * CROP_H) 
    
    if valid_mask_np.sum() == 0:
        raise ValueError("🚨 MAP 이미지에 흰색 유효 영역이 존재하지 않습니다.")

    # 2. Last 모델 단독 로드
    models = {}
    for en in ALL_ELEMS:
        G = load_last_model(en, device)
        if G is not None:
            models[en] = G
        else:
            print(f"⚠️ {en} 원소의 'last' 모델 가중치 파일이 존재하지 않습니다. (Pass)")

    if not models:
        print("❌ 폴더 내에 로드할 수 있는 last_model 가중치 파일이 전혀 없습니다. 가중치 파일명을 확인하세요.")
        return

    # 3. 모델 추론 및 원소별 Area% 연산
    pred_bins = {}
    area_results = []

    print(f"\n{'='*45}")
    print(f"  원소 (Elem)  |   예측 면적 비율 (Pred %)  ")
    print(f"{'='*45}")

    for en in ALL_ELEMS:
        if en in models:
            models[en].eval()
            with torch.no_grad():
                logit = models[en](om_t.to(device))
                pred_sig = torch.sigmoid(logit).cpu().squeeze().numpy()
                
                # 💡 [요청 반영 - 분자 이원화 조건문]
                if en == "Al":
                    # Al은 이미지 전체 영역에서 threshold를 통과한 픽셀을 분자로 취함
                    pred_bin = (pred_sig > THRESHOLD).astype(np.uint8)
                else:
                    # Al을 제외한 원소는 MAP 유효 영역 내부의 예측 픽셀만 분자로 취함
                    pred_bin = ((pred_sig > THRESHOLD) * valid_mask_np).astype(np.uint8)
        else:
            pred_bin = np.zeros((CROP_H, CROP_W), dtype=np.uint8)

        pred_bins[en] = pred_bin

        # 💡 [요청 반영] 분모는 예외 없이 512x512 전체 픽셀 수로 연산
        pred_area_pct = (pred_bin.sum() / total_image_pixels) * 100

        print(f"      {en:<8} |       {pred_area_pct:8.4f} %")
        area_results.append({"Element": en, "Pred_Area(%)": round(pred_area_pct, 4)})

    print(f"{'='*45}")

    # 4. 정량 수치 결과 CSV 파일로 저장
    csv_out_path = os.path.join(OUTPUT_DIR, f"{NEW_BASE_NAME}_last_area_metrics_fixed_total_denom.csv")
    with open(csv_out_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=["Element", "Pred_Area(%)"])
        writer.writeheader()
        writer.writerows(area_results)
    print(f"✅ 예측 면적률 수치 CSV 저장 완료: {csv_out_path}")

    # 5. 다중 예측 원소 오버랩 맵 시각화 이미지 생성 (이미지는 전부 MAP 유효 영역 내부 제한 적용됨)
    active_all = [en for en in ALL_ELEMS if en in models]
    if active_all:
        pred_overlay_all = create_multi_elem_overlay(om_raw_crop, pred_bins, valid_mask_np, active_all)
        save_overlay_plot(pred_overlay_all, active_all, 3, os.path.join(OUTPUT_DIR, f"{NEW_BASE_NAME}_6elems_last_Pred.png"))

    active_prec = [en for en in PREC_ELEMS if en in models]
    if active_prec:
        pred_overlay_prec = create_multi_elem_overlay(om_raw_crop, pred_bins, valid_mask_np, active_prec)
        save_overlay_plot(pred_overlay_prec, active_prec, 2, os.path.join(OUTPUT_DIR, f"{NEW_BASE_NAME}_prec4_last_Pred.png"))

    print(f"✅ 논문용 융합 오버랩 이미지 저장 완료. 결과 폴더: {OUTPUT_DIR}")
    print("\n✨ 연산 공식 피드백이 완벽하게 반영되어 가공 처리가 끝났습니다!")

if __name__ == '__main__':
    main()