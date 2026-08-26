"""
Metallurgical Descriptor Evaluation Script
- Computes Particle Size Distribution (PSD) and Cross-Element Nearest-Neighbor Spacing
  (cross-NND, the 6 precipitate-element pairs).
- Evaluated on the test set defined in splits.json.
- Deep Ensemble: each element has N_ENSEMBLE independently trained Generators.
  Each member is thresholded independently (own sigmoid map > THRESHOLD).
- PSD: per-member particle areas are pooled across members and across all
  test images (GT vs Pred distributions), then compared via KS-test and
  Wasserstein distance.
- Cross-NND: for the 4 precipitate elements (Mg, Fe, Cu, Sr) only, since it
  requires discrete particles. For each pair, each particle's distance to its
  nearest particle of the other element is computed (both directions), pooled
  across members and all test images (GT vs Pred), then compared via KS-test
  and Wasserstein distance — same aggregation style as PSD.
- MIN_PARTICLE_AREA_PX filters out connected components below this pixel
  count. Border-touching particles are not clipped. Areas/distances are in
  pixel units (no physical pixel-size conversion applied).
- All CSV and histogram (GT vs Pred) outputs are written under a single
  subfolder: TEST_DIR/metallurgical_descriptors_{model_tag}/.
"""

import os
import json
import csv
import itertools
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet34
from skimage.measure import label, regionprops
from scipy.spatial import cKDTree
from scipy.stats import ks_2samp, wasserstein_distance
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

ALL_ELEMS  = ["Mg", "Al", "Si", "Fe", "Cu", "Sr"]
PREC_ELEMS = ["Mg", "Fe", "Cu", "Sr"]   # PSD and Cross-NND target elements
THRESHOLD  = 0.5
N_ENSEMBLE = 3
CROP_W, CROP_H = 512, 512
EDS_SUFFIXES = {"Mg": "01", "Al": "02", "Si": "03", "Fe": "06", "Cu": "07", "Sr": "09"}

MIN_PARTICLE_AREA_PX = 4  # connected components smaller than this are dropped as noise

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

class MultiElemTestDataset(Dataset):
    """Loads OM images and GT binary EDS maps for all elements simultaneously."""
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

        om_c   = preprocess_om_color(om_raw)[sy:sy+CROP_H, sx:sx+CROP_W]
        mask_c = g_mask[sy:sy+CROP_H, sx:sx+CROP_W]
        map_c  = g_map [sy:sy+CROP_H, sx:sx+CROP_W]

        eds_bins = {}
        for en in self.elem_names:
            suf = EDS_SUFFIXES[en]
            raw = imread_korean(os.path.join(EDS_DIR, f"{base}_{suf}.png"), 0)
            raw = raw[sy:sy+CROP_H, sx:sx+CROP_W]

            if en == "Al":
                th = np.percentile(raw, 20)
            elif en == "Si":
                th = np.percentile(raw, 90)
            else:
                th = np.percentile(raw, 99)
            eds_bins[en] = torch.from_numpy((raw > th).astype(np.uint8)).float().unsqueeze(0)

        om_t    = (torch.from_numpy(om_c).permute(2, 0, 1).float() / 127.5) - 1.0
        align_t = (torch.from_numpy(mask_c) > 0).float()
        almap_t = (torch.from_numpy(map_c)  > 0).float()
        valid_mask = (align_t * almap_t).unsqueeze(0)

        return om_t, eds_bins, valid_mask, base

def collate_fn(batch):
    om_list, eds_list, mask_list, names = zip(*batch)
    om_t   = torch.stack(om_list)
    mask_t = torch.stack(mask_list)
    elem_names = list(eds_list[0].keys())
    eds_stacked = {en: torch.stack([b[en] for b in eds_list]) for en in elem_names}
    return om_t, eds_stacked, mask_t, list(names)

# =============================================================================
# Model Architecture (Generator with CBAM & Decoder)
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

def load_model_ensemble(elem_name, model_tag, device):
    """Loads all available Deep Ensemble member checkpoints for one element/tag."""
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

# =============================================================================
# Descriptor Computation
# =============================================================================
def particle_props(bin_mask, valid_mask):
    """Connected-component (area, centroid) pairs within the valid ROI."""
    m = (bin_mask.astype(np.uint8) & valid_mask.astype(np.uint8))
    labeled = label(m, connectivity=2)
    return [(r.area, r.centroid) for r in regionprops(labeled) if r.area >= MIN_PARTICLE_AREA_PX]

def cross_nnd(centroids_a, centroids_b):
    """Nearest-neighbor distances between two particle sets, both directions pooled together."""
    if len(centroids_a) == 0 or len(centroids_b) == 0:
        return []
    dist_a_to_b, _ = cKDTree(centroids_b).query(centroids_a, k=1)
    dist_b_to_a, _ = cKDTree(centroids_a).query(centroids_b, k=1)
    return list(dist_a_to_b) + list(dist_b_to_a)

def plot_distribution_comparison(gt_vals, pred_vals, xlabel, save_path, log_y=False):
    """Overlaid GT vs Pred histogram (density-normalized) for a distribution comparison."""
    if len(gt_vals) == 0 or len(pred_vals) == 0:
        return
    combined = np.concatenate([gt_vals, pred_vals])
    bins = np.histogram_bin_edges(combined, bins=50)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.hist(gt_vals, bins=bins, density=True, histtype="stepfilled",
            alpha=0.45, linewidth=1.5, edgecolor="#3B6FA0",
            facecolor="#5B9BD5", label="GT")
    ax.hist(pred_vals, bins=bins, density=True, histtype="stepfilled",
            alpha=0.45, linewidth=1.5, edgecolor="#C4622D",
            facecolor="#ED7D31", label="Pred")

    if log_y:
        ax.set_yscale('log')
        ax.set_ylabel("Density (log scale)", fontsize=40, labelpad=15)
    else:
        ax.set_ylabel("Density", fontsize=40, labelpad=15)

    ax.set_xlabel(xlabel, fontsize=40, labelpad=15)
    ax.tick_params(axis='both', which='major', labelsize=40, width=3, length=6)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.legend(fontsize=36)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

# =============================================================================
# Main Execution Pipeline
# =============================================================================
def main(model_tag="last"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using Device: {device}")
    print(f"[INFO] Model tag: {model_tag}\n")

    splits_path = os.path.join(RESULT_DIR, 'tversky', 'splits.json')
    if not os.path.exists(splits_path):
        raise FileNotFoundError(f"[ERROR] 'splits.json' missing at {splits_path}")
    with open(splits_path) as f:
        splits = json.load(f)
    test_files = splits['test']
    print(f"[INFO] Test set: {len(test_files)} samples\n")

    desc_dir = os.path.join(TEST_DIR, f"metallurgical_descriptors_{model_tag}")
    os.makedirs(desc_dir, exist_ok=True)

    print("[INFO] Loading ensemble models for target elements...")
    models = {en: load_model_ensemble(en, model_tag, device) for en in PREC_ELEMS}
    models = {en: m for en, m in models.items() if m}
    missing = [en for en in PREC_ELEMS if en not in models]
    if missing:
        print(f"  [WARNING] No models found for: {missing} (skipped)")
    elem_names = list(models.keys())
    prec_pairs = list(itertools.combinations(elem_names, 2))

    dataset = MultiElemTestDataset(test_files, elem_names)
    loader  = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0, collate_fn=collate_fn)

    # --- PSD accumulators: pooled across members AND all test images ---
    psd_areas_gt   = {en: [] for en in elem_names}
    psd_areas_pred = {en: [] for en in elem_names}

    # --- Cross-NND accumulators: pooled across members AND all test images ---
    nnd_gt   = {p: [] for p in prec_pairs}
    nnd_pred = {p: [] for p in prec_pairs}

    for m in models.values():
        for G in m:
            G.eval()

    with torch.no_grad():
        for om_t, eds_bins, v_mask, base_names in loader:
            om_t = om_t.to(device)
            mask_np = v_mask.cpu().numpy()[:, 0] > 0.5

            # Per-element, per-member binary predictions for this batch
            member_bins_batch = {}
            for en in elem_names:
                member_probs = [torch.sigmoid(G(om_t)).cpu().numpy()[:, 0] for G in models[en]]
                member_bins_batch[en] = [(p > THRESHOLD).astype(np.uint8) for p in member_probs]

            B = om_t.shape[0]
            for b in range(B):
                valid_b = mask_np[b]

                gt_bins_b = {en: eds_bins[en][b, 0].numpy().astype(np.uint8) for en in elem_names}
                member_bins_b = {en: [mb[b] for mb in member_bins_batch[en]] for en in elem_names}

                # Particle extraction (area + centroid together, one labeling pass per mask)
                gt_props     = {en: particle_props(gt_bins_b[en], valid_b) for en in elem_names}
                member_props = {
                    en: [particle_props(member_bins_b[en][i], valid_b) for i in range(len(models[en]))]
                    for en in elem_names
                }

                # PSD: pool particle areas across members and images
                for en in elem_names:
                    psd_areas_gt[en].extend(a for a, c in gt_props[en])
                    for i in range(len(models[en])):
                        psd_areas_pred[en].extend(a for a, c in member_props[en][i])

                # Cross-NND: pool nearest-neighbor distances across members and images
                for (e1, e2) in prec_pairs:
                    gt_c1 = [c for a, c in gt_props[e1]]
                    gt_c2 = [c for a, c in gt_props[e2]]
                    nnd_gt[(e1, e2)].extend(cross_nnd(gt_c1, gt_c2))
                    for i in range(min(len(models[e1]), len(models[e2]))):
                        m_c1 = [c for a, c in member_props[e1][i]]
                        m_c2 = [c for a, c in member_props[e2][i]]
                        nnd_pred[(e1, e2)].extend(cross_nnd(m_c1, m_c2))

    # =============================================================================
    # Export: PSD pooled raw areas + summary (count/mean/median/std, KS-test, Wasserstein)
    # =============================================================================
    psd_raw_path = os.path.join(desc_dir, f"psd_pooled_areas_{model_tag}.csv")
    with open(psd_raw_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(["element", "source", "area_px"])
        for en in elem_names:
            for a in psd_areas_gt[en]:
                writer.writerow([en, "GT", a])
            for a in psd_areas_pred[en]:
                writer.writerow([en, "Pred", a])
    print(f"[SUCCESS] PSD pooled raw particle areas saved to: {psd_raw_path}")

    psd_summary_path = os.path.join(desc_dir, f"psd_summary_{model_tag}.csv")
    with open(psd_summary_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "element", "gt_n_particles", "gt_mean_area", "gt_median_area", "gt_std_area",
            "pred_n_particles_pooled", "pred_n_particles_per_member",
            "pred_mean_area", "pred_median_area", "pred_std_area",
            "ks_statistic", "ks_pvalue", "wasserstein_distance"
        ])
        writer.writeheader()
        for en in elem_names:
            gt_a = np.array(psd_areas_gt[en])
            pred_a = np.array(psd_areas_pred[en])
            if len(gt_a) == 0 or len(pred_a) == 0:
                print(f"  [WARNING] {en}: no particles found in GT or Pred (GT n={len(gt_a)}, Pred n={len(pred_a)}). Skipping stats row.")
                continue
            ks_stat, ks_p = ks_2samp(gt_a, pred_a)
            wd = wasserstein_distance(gt_a, pred_a)
            writer.writerow({
                "element": en,
                "gt_n_particles": len(gt_a),
                "gt_mean_area": f"{gt_a.mean():.2f}",
                "gt_median_area": f"{np.median(gt_a):.2f}",
                "gt_std_area": f"{gt_a.std():.2f}",
                "pred_n_particles_pooled": len(pred_a),
                "pred_n_particles_per_member": f"{len(pred_a) / N_ENSEMBLE:.1f}",
                "pred_mean_area": f"{pred_a.mean():.2f}",
                "pred_median_area": f"{np.median(pred_a):.2f}",
                "pred_std_area": f"{pred_a.std():.2f}",
                "ks_statistic": f"{ks_stat:.4f}",
                "ks_pvalue": f"{ks_p:.4g}",
                "wasserstein_distance": f"{wd:.2f}",
            })
            plot_distribution_comparison(
                gt_a, pred_a,
                xlabel="Particle area (px)",
                save_path=os.path.join(desc_dir, f"psd_hist_{en}_{model_tag}.png"),
                log_y=True
            )
    print(f"[SUCCESS] PSD summary statistics saved to: {psd_summary_path}")
    print(f"[SUCCESS] PSD histograms saved to: {desc_dir}")

    # =============================================================================
    # Export: Cross-NND pooled raw distances + summary (KS-test, Wasserstein)
    # =============================================================================
    nnd_raw_path = os.path.join(desc_dir, f"nnd_pooled_distances_{model_tag}.csv")
    with open(nnd_raw_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(["pair", "source", "distance_px"])
        for (e1, e2) in prec_pairs:
            pair_key = f"{e1}-{e2}"
            for d in nnd_gt[(e1, e2)]:
                writer.writerow([pair_key, "GT", d])
            for d in nnd_pred[(e1, e2)]:
                writer.writerow([pair_key, "Pred", d])
    print(f"[SUCCESS] Cross-NND pooled raw distances saved to: {nnd_raw_path}")

    nnd_summary_path = os.path.join(desc_dir, f"nnd_summary_{model_tag}.csv")
    with open(nnd_summary_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "pair", "gt_n", "gt_mean_dist", "gt_median_dist", "gt_std_dist",
            "pred_n_pooled", "pred_n_per_member",
            "pred_mean_dist", "pred_median_dist", "pred_std_dist",
            "ks_statistic", "ks_pvalue", "wasserstein_distance"
        ])
        writer.writeheader()
        for (e1, e2) in prec_pairs:
            gt_d = np.array(nnd_gt[(e1, e2)])
            pred_d = np.array(nnd_pred[(e1, e2)])
            if len(gt_d) == 0 or len(pred_d) == 0:
                print(f"  [WARNING] {e1}-{e2}: no cross-pairs found in GT or Pred (GT n={len(gt_d)}, Pred n={len(pred_d)}). Skipping stats row.")
                continue
            ks_stat, ks_p = ks_2samp(gt_d, pred_d)
            wd = wasserstein_distance(gt_d, pred_d)
            writer.writerow({
                "pair": f"{e1}-{e2}",
                "gt_n": len(gt_d),
                "gt_mean_dist": f"{gt_d.mean():.2f}",
                "gt_median_dist": f"{np.median(gt_d):.2f}",
                "gt_std_dist": f"{gt_d.std():.2f}",
                "pred_n_pooled": len(pred_d),
                "pred_n_per_member": f"{len(pred_d) / N_ENSEMBLE:.1f}",
                "pred_mean_dist": f"{pred_d.mean():.2f}",
                "pred_median_dist": f"{np.median(pred_d):.2f}",
                "pred_std_dist": f"{pred_d.std():.2f}",
                "ks_statistic": f"{ks_stat:.4f}",
                "ks_pvalue": f"{ks_p:.4g}",
                "wasserstein_distance": f"{wd:.2f}",
            })
            plot_distribution_comparison(
                gt_d, pred_d,
                xlabel="Nearest-neighbor distance (px)",
                save_path=os.path.join(desc_dir, f"nnd_hist_{e1}-{e2}_{model_tag}.png"),
            )
    print(f"[SUCCESS] Cross-NND summary statistics saved to: {nnd_summary_path}")
    print(f"[SUCCESS] Cross-NND histograms saved to: {desc_dir}")

    print(f"\n[SUCCESS] Metallurgical descriptor evaluation completed.")
    print(f"[SUCCESS] All outputs saved under: {desc_dir}")


if __name__ == '__main__':
    main(model_tag="last")