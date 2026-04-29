"""step3: habitat聚类 K=3, 12维体素特征, GPU KMeans"""
import os
import sys
import json
import logging
import warnings
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from scipy.ndimage import gaussian_laplace

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
from config_v4 import (OUT_PREPROC, OUT_HABITAT, LOG_DIR,
                        HABITAT_K, HABITAT_CLUSTER_WIN, HABITAT_GLCM_LEVELS,
                        HABITAT_LOG_SIGMAS, HABITAT_KMEANS_NINIT,
                        HABITAT_KMEANS_MAXITER, RANDOM_STATE)

os.makedirs(OUT_HABITAT, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'step3_v4.log'), mode='w'),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)

import torch
import cupy as cp
from cuml.cluster import KMeans as cuKMeans
from cuml.preprocessing import StandardScaler as cuScaler
from sklearn.metrics import silhouette_score

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
log.info(f'Device: {DEVICE}')


def compute_glcm_features_3d(arr: np.ndarray, mask: np.ndarray,
                              win: int = 3, levels: int = 16) -> np.ndarray:
    """
    计算每个肿瘤体素的 3x3x3 窗口内 GLCM 纹理特征:
      - contrast, energy, homogeneity
    使用量化后的强度值，d=1 在26邻域方向取平均
    """
    # 量化到 [0, levels-1]
    vmin, vmax = arr[mask].min(), arr[mask].max()
    rng = vmax - vmin
    if rng < 1e-6:
        rng = 1.0
    arr_q = ((arr - vmin) / rng * (levels - 1)).clip(0, levels - 1).astype(np.int32)

    pad = win // 2
    arr_pad = np.pad(arr_q, pad, mode='reflect')
    mask_pad = np.pad(mask, pad, mode='constant', constant_values=False)

    coords = np.argwhere(mask)
    n_vox = len(coords)
    feats = np.zeros((n_vox, 3), dtype=np.float32)

    # 26-connectivity offsets (d=1)
    offsets = []
    for dz in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dz == 0 and dy == 0 and dx == 0:
                    continue
                offsets.append((dz, dy, dx))

    for i, (z, y, x) in enumerate(coords):
        zp, yp, xp = z + pad, y + pad, x + pad
        patch = arr_pad[zp - pad:zp + pad + 1,
                        yp - pad:yp + pad + 1,
                        xp - pad:xp + pad + 1]

        # 快速 GLCM: 累积所有方向的共生矩阵
        glcm = np.zeros((levels, levels), dtype=np.float32)
        center = patch[pad, pad, pad]
        for dz, dy, dx in offsets:
            nz, ny, nx = pad + dz, pad + dy, pad + dx
            if 0 <= nz < win and 0 <= ny < win and 0 <= nx < win:
                neighbor = patch[nz, ny, nx]
                glcm[center, neighbor] += 1

        total = glcm.sum()
        if total > 0:
            glcm /= total

        # Contrast
        ii, jj = np.meshgrid(range(levels), range(levels), indexing='ij')
        contrast = float(np.sum((ii - jj) ** 2 * glcm))
        # Energy
        energy = float(np.sum(glcm ** 2))
        # Homogeneity
        homogeneity = float(np.sum(glcm / (1.0 + np.abs(ii - jj))))

        feats[i] = [contrast, energy, homogeneity]

    return feats


def extract_12dim_features(arr: np.ndarray, mask: np.ndarray,
                            win: int = 3) -> np.ndarray:
    """
    12维体素级特征:
      原始强度(1) + LoG 4尺度(4) + GLCM纹理(3) + 局部统计(4)
    """
    coords = np.argwhere(mask)
    n_vox = len(coords)
    feats = np.zeros((n_vox, 12), dtype=np.float32)

    feats[:, 0] = arr[mask]  # 原始强度

    # LoG多尺度
    for idx, sigma in enumerate(HABITAT_LOG_SIGMAS):
        log_img = gaussian_laplace(arr, sigma=sigma).astype(np.float32)
        feats[:, 1 + idx] = log_img[mask]

    # GLCM
    glcm_feats = compute_glcm_features_3d(arr, mask, win=win,
                                           levels=HABITAT_GLCM_LEVELS)
    feats[:, 5:8] = glcm_feats

    # 局部统计 (3x3x3窗口, GPU)
    pad = win // 2
    D, H, W = arr.shape
    arr_t = torch.from_numpy(arr.astype(np.float32)).to(DEVICE)
    arr_pad = torch.nn.functional.pad(
        arr_t.unsqueeze(0).unsqueeze(0),
        (pad, pad, pad, pad, pad, pad), mode='reflect'
    ).squeeze()
    unf = arr_pad.unfold(0, win, 1).unfold(1, win, 1).unfold(2, win, 1)
    patches = unf.contiguous().view(D, H, W, win ** 3)
    mask_t = torch.from_numpy(mask).to(DEVICE)
    X_patches = patches[mask_t].cpu().numpy()

    feats[:, 8] = X_patches.mean(axis=1)   # local mean
    feats[:, 9] = X_patches.std(axis=1)    # local std
    mu = feats[:, 8:9]
    std = np.clip(feats[:, 9:10], 1e-6, None)
    centered = X_patches - mu
    feats[:, 10] = (centered ** 3).mean(axis=1) / (std.flatten() ** 3)  # skewness
    feats[:, 11] = (centered ** 4).mean(axis=1) / (std.flatten() ** 4)  # kurtosis

    return feats


def cluster_habitat(X: np.ndarray, k: int = 3) -> tuple:
    """GPU KMeans 聚类，返回 labels 和 silhouette"""
    X_gpu = cp.asarray(X.astype(np.float32))
    scaler = cuScaler()
    X_scaled = scaler.fit_transform(X_gpu)

    km = cuKMeans(n_clusters=k, random_state=RANDOM_STATE,
                  n_init=HABITAT_KMEANS_NINIT,
                  max_iter=HABITAT_KMEANS_MAXITER)
    labels_gpu = km.fit_predict(X_scaled)
    labels = cp.asnumpy(labels_gpu).astype(int)
    X_scaled_np = cp.asnumpy(X_scaled)

    # Silhouette (subsample for speed)
    n_samp = min(5000, len(labels))
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(len(labels), n_samp, replace=False)
    sil = silhouette_score(X_scaled_np[idx], labels[idx])

    return labels, float(sil)


def reorder_labels_by_intensity(labels: np.ndarray,
                                 intensities: np.ndarray) -> np.ndarray:
    """按平均强度排序: H1=最低(坏死), H2=中间(活性), H3=最高(侵袭)"""
    means = {}
    for k in range(3):
        vals = intensities[labels == k]
        means[k] = vals.mean() if len(vals) > 0 else 0
    sorted_k = sorted(means, key=means.get)
    mapping = {old: new for new, old in enumerate(sorted_k)}
    return np.vectorize(mapping.get)(labels)


def process_case(case_id: str) -> dict:
    img_path = os.path.join(OUT_PREPROC, f'{case_id}_image.nii.gz')
    lbl_path = os.path.join(OUT_PREPROC, f'{case_id}_label.nii.gz')

    if not (os.path.exists(img_path) and os.path.exists(lbl_path)):
        return {'case_id': case_id, 'status': 'missing_input'}

    # Check if done
    expected = [os.path.join(OUT_HABITAT, f'{case_id}_H{h}.nii.gz')
                for h in [1, 2, 3]]
    if all(os.path.exists(p) for p in expected):
        return {'case_id': case_id, 'status': 'skipped'}

    try:
        img_sitk = sitk.ReadImage(img_path)
        lbl_sitk = sitk.ReadImage(lbl_path)
        arr = sitk.GetArrayFromImage(img_sitk).astype(np.float32)
        mask = sitk.GetArrayFromImage(lbl_sitk).astype(bool)

        n_tumor = int(mask.sum())
        if n_tumor < 30:
            return {'case_id': case_id, 'status': 'too_small',
                    'tumor_vox': n_tumor}

        # Crop to bounding box + padding
        coords = np.argwhere(mask)
        zmin, ymin, xmin = coords.min(axis=0)
        zmax, ymax, xmax = coords.max(axis=0)
        pad = HABITAT_CLUSTER_WIN + 2
        sl = (slice(max(0, zmin - pad), min(arr.shape[0], zmax + pad + 1)),
              slice(max(0, ymin - pad), min(arr.shape[1], ymax + pad + 1)),
              slice(max(0, xmin - pad), min(arr.shape[2], xmax + pad + 1)))

        arr_crop = arr[sl]
        mask_crop = mask[sl]

        # Extract 12-dim features
        X = extract_12dim_features(arr_crop, mask_crop, win=HABITAT_CLUSTER_WIN)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Cluster
        labels, sil = cluster_habitat(X, k=HABITAT_K)

        # Reorder by intensity
        intensities = arr_crop[mask_crop]
        labels = reorder_labels_by_intensity(labels, intensities)

        # Write back to full volume
        full_cluster = np.zeros(arr.shape, dtype=np.uint8)
        tmp = np.zeros(mask_crop.shape, dtype=np.uint8)
        tmp[mask_crop] = labels + 1  # 1,2,3
        full_cluster[sl] = tmp

        # Save combined cluster map
        cluster_sitk = sitk.GetImageFromArray(full_cluster)
        cluster_sitk.CopyInformation(img_sitk)
        sitk.WriteImage(cluster_sitk,
                        os.path.join(OUT_HABITAT, f'{case_id}_habitat.nii.gz'),
                        useCompression=True)

        # Save individual H1, H2, H3 masks
        subregion_voxels = {}
        for h in [1, 2, 3]:
            sub = (full_cluster == h).astype(np.uint8)
            sub_sitk = sitk.GetImageFromArray(sub)
            sub_sitk.CopyInformation(img_sitk)
            sitk.WriteImage(sub_sitk,
                            os.path.join(OUT_HABITAT, f'{case_id}_H{h}.nii.gz'),
                            useCompression=True)
            subregion_voxels[f'H{h}'] = int(sub.sum())

        proportions = {k: round(v / n_tumor, 4)
                       for k, v in subregion_voxels.items()}

        return {
            'case_id': case_id, 'status': 'ok',
            'tumor_voxels': n_tumor,
            'silhouette': round(sil, 4),
            'subregion_voxels': subregion_voxels,
            'subregion_proportions': proportions,
        }
    except Exception as e:
        import traceback
        return {'case_id': case_id, 'status': 'error', 'msg': str(e),
                'traceback': traceback.format_exc()}


def main():
    cases = sorted([
        f.replace('_image.nii.gz', '')
        for f in os.listdir(OUT_PREPROC)
        if f.endswith('_image.nii.gz')
    ])
    log.info(f'共 {len(cases)} 例，Habitat v4 (K=3 fixed, 12-dim) ...')

    results = []
    for case_id in tqdm(cases, desc='Habitat v4'):
        r = process_case(case_id)
        results.append(r)
        if r['status'] == 'error':
            log.warning(f"  {r['case_id']}: {r.get('msg','')[:120]}")

    with open(os.path.join(LOG_DIR, 'step3_v4_summary.json'), 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    ok_results = [r for r in results if r['status'] == 'ok']
    sils = [r['silhouette'] for r in ok_results]
    log.info(f'Silhouette: mean={np.mean(sils):.3f} '
             f'median={np.median(sils):.3f} '
             f'min={np.min(sils):.3f}')

    ok = sum(1 for r in results if r['status'] == 'ok')
    skip = sum(1 for r in results if r['status'] == 'skipped')
    error = sum(1 for r in results if r['status'] == 'error')
    log.info(f'完成: ok={ok}  skipped={skip}  error={error}')


if __name__ == '__main__':
    main()
