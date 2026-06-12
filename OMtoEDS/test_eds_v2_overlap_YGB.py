import os, json, random
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

# ──────────────────────────────────────────────
# 설정  (tversky 가중치 및 데이터 호환 경로 자동 지정)
# ──────────────────────────────────────────────
OM_DIR   = r'C:\Taehyun\260512\data\OM'
EDS_DIR  = r'.\data\EDS'
MASK_DIR = r'.\data\MASK'
MAP_DIR  = r'.\data\MAP'
RESULT_DIR = r'.\result'

MODEL_DIR  = os.path.join(RESULT_DIR, 'tversky', 'models_tversky')
TEST_DIR   = os.path.join(RESULT_DIR, 'test_tversky')

TARGET_ELEMS = ["Mg", "Al", "Si", "Cu", "Fe", "Sr"]
THRESHOLD  = 0.5
CROP_W, CROP_H = 512, 512

EDS_SUFFIXES = {"Mg":"01", "Al":"02", "Si":"03", "Cu":"06", "Fe":"07", "Sr":"09"}

# 폰트 패밀리를 DejaVu Sans로 지정
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans'] + plt.rcParams['font.sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ──────────────────────────────────────────────
# 유틸 및 데이터셋
# ──────────────────────────────────────────────
def imread_korean(path, mode=cv2.IMREAD_COLOR):
    try:
        n = np.fromfile(path, np.uint8)
        return cv2.imdecode(n, mode)
    except:
        return None

def preprocess_om_color(om_bgr):
    return cv2.cvtColor(om_bgr, cv2.COLOR_BGR2RGB)

class TestDataset(Dataset):
    def __init__(self, file_list, elem_name):
        self.file_list = file_list
        self.elem_name = elem_name
        self.suf = EDS_SUFFIXES[elem_name]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        base = self.file_list[idx]

        om_raw = imread_korean(os.path.join(OM_DIR,   f"{base}.png"), 1)
        g_mask = imread_korean(os.path.join(MASK_DIR, f"{base}.png"), 0)
        g_map  = imread_korean(os.path.join(MAP_DIR,  f"{base}.png"), 0)
        raw_eds = imread_korean(os.path.join(EDS_DIR, f"{base}_{self.suf}.png"), 0)
        
        h, w = g_mask.shape
        sy = (h - CROP_H) // 2
        sx = (w - CROP_W) // 2

        # 💡 배경 오버랩용으로 가공하기 위해 원래의 OM BGR 이미지도 슬라이싱해서 원본 형태로 리턴에 추가합니다.
        om_raw_crop = om_raw[sy:sy+CROP_H, sx:sx+CROP_W]
        om_c    = preprocess_om_color(om_raw)[sy:sy+CROP_H, sx:sx+CROP_W]
        mask_c  = g_mask[sy:sy+CROP_H, sx:sx+CROP_W]
        map_c   = g_map [sy:sy+CROP_H, sx:sx+CROP_W]
        eds_c   = raw_eds[sy:sy+CROP_H, sx:sx+CROP_W]

        if self.elem_name == "Al":
            th = np.percentile(eds_c, 20)
        elif self.elem_name == "Si":
            th = np.percentile(eds_c, 90)
        else:
            th = np.percentile(eds_c, 99)
        eds_bin = (eds_c > th).astype(np.uint8)

        om_t    = (torch.from_numpy(om_c).permute(2, 0, 1).float() / 127.5) - 1.0
        eds_t   = torch.from_numpy(eds_bin).float().unsqueeze(0)
        align_t = (torch.from_numpy(mask_c) > 0).float()
        almap_t = (torch.from_numpy(map_c)  > 0).float()
        valid_mask  = (align_t * almap_t).unsqueeze(0)
        align_only  = align_t.unsqueeze(0)

        return om_t, eds_t, valid_mask, align_only, base, om_raw_crop

# ──────────────────────────────────────────────
# 네트워크 아키텍처
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
# 핵심 시각화 및 연한 그레이스케일 OM 오버랩 함수
# ──────────────────────────────────────────────
def create_om_overlay_comparison(om_bgr, gt_bin, pred_bin):
    """
    OM 이미지를 은은한 연회색(Gray) 도화지로 만들고, 
    그 위에 유효 마스크 영역 내부의 Match/Miss/False 픽셀을 선명하게 오버랩합니다.
    """
    # 1. 원본 크롭 OM 이미지를 논문용 그레이스케일 배경으로 가공
    om_gray = cv2.cvtColor(om_bgr, cv2.COLOR_BGR2GRAY)
    
    # 석출상 형상이 묻히지 않으면서도 마스크 색상이 잘 돋보이도록 연하게 밝기 톤 조절 (0.4 * gray + 153)
    om_faded = cv2.addWeighted(om_gray, 0.4, np.full_like(om_gray, 255), 0.6, 0)
    canvas = cv2.cvtColor(om_faded, cv2.COLOR_GRAY2RGB) # Matplotlib에 태우기 위해 3채널 RGB화

    # 2. 오차 연산용 마스크 조건 정의
    tp_mask = (gt_bin == 1) & (pred_bin == 1)
    fn_mask = (gt_bin == 1) & (pred_bin == 0)
    fp_mask = (gt_bin == 0) & (pred_bin == 1)

    # 3. 그레이스케일 지도 위에 매핑할 유색 픽셀 도포
    canvas[tp_mask] = [40, 180, 70]   # Match -> 선명한 초록색
    canvas[fn_mask] = [230, 180, 30]  # Miss  -> 화사한 노란색
    canvas[fp_mask] = [220, 50, 50]   # False -> 크림슨 빨간색

    return canvas

def evaluate_and_visualize_overlap(model, loader, device, elem_name, is_al, save_dir, model_tag):
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

                # 단일 배치의 원본 크롭 넘파이 이미지 추출
                single_om_bgr = om_raw_crops[b].numpy()

                # 💡 검정 배경 대신 연한 그레이스케일 OM이 결합된 오버랩 맵 생성
                overlay_comparison_map = create_om_overlay_comparison(single_om_bgr, gt_bin, pred_bin)

                fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
                ax.imshow(overlay_comparison_map)
                ax.axis('off')

                # 범례 패치 구성
                patch_match = mpatches.Patch(color='#28B446', label='Match')
                patch_miss  = mpatches.Patch(color='#E6B41E', label='Miss')
                patch_false = mpatches.Patch(color='#DC3232', label='False')
                
                # 우측 하단 세로 3줄 정렬 유지 (흰색 배경 프레임 포함으로 석출상 윤곽과 간섭 최소화)
                ax.legend(handles=[patch_match, patch_miss, patch_false], 
                          loc='lower right', ncol=1, fontsize=9, 
                          frameon=True, facecolor='white', edgecolor='none',
                          handletextpad=0.5, labelspacing=0.5, borderpad=0.5)

                save_fname = os.path.join(target_save_path, f"{base_names[b]}_mask_analysis.png")
                plt.savefig(save_fname, dpi=300, bbox_inches='tight', pad_inches=0.02)
                plt.close()

# ──────────────────────────────────────────────
# 메인 제어 루프
# ──────────────────────────────────────────────
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"시각화 전용 디바이스: {device}\n")

    splits_path = os.path.join(RESULT_DIR, 'splits.json')
    if not os.path.exists(splits_path):
        raise FileNotFoundError(f"splits.json이 없습니다. 경로를 확인하세요: {splits_path}")
        
    with open(splits_path) as f:
        splits = json.load(f)
    test_files = splits['test']
    print(f"로드된 테스트 샘플 수: {len(test_files)}개\n")

    for elem_name in TARGET_ELEMS:
        key = elem_name.lower()
        is_al = (elem_name == "Al")

        best_path = os.path.join(MODEL_DIR, f"best_model_{key}.pth")
        last_path = os.path.join(MODEL_DIR, f"last_model_{key}.pth")

        models_to_test = []
        for tag, path in [("best", best_path), ("last", last_path)]:
            if os.path.exists(path):
                models_to_test.append((tag, path))

        if not models_to_test:
            print(f"⚠️ {elem_name} 모델 파일이 존재하지 않아 건너뜁니다. (경로 확인: {MODEL_DIR})")
            continue

        print(f"▶ {elem_name} 원소 논문 규격 오버랩 맵 생성 중...")
        dataset = TestDataset(test_files, elem_name)
        # 넘파이 이미지 객체를 배치 형태로 그대로 가져오기 위해 정렬 루틴 적용
        loader  = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

        for model_tag, model_path in models_to_test:
            G = Generator(out_ch=1).to(device)
            
            state_dict = torch.load(model_path, map_location=device)
            strip_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            G.load_state_dict(strip_dict)

            evaluate_and_visualize_overlap(
                G, loader, device, elem_name, is_al,
                save_dir=TEST_DIR, model_tag=model_tag
            )
            print(f"   - [{model_tag.upper()}] 정밀 오버랩 분석 이미지 저장 완료.")

    print(f"\n✨ 모든 지정 원소의 논문용 비교 시각화 데이터 저장이 완료되었습니다!")
    print(f"결과 레포트 경로: {TEST_DIR}")

if __name__ == '__main__':
    main()