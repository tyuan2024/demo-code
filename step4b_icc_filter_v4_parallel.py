"""step4b: ICC(3,1) 稳定性过滤, mask扰动, 多进程"""
import os
import sys
import json
import logging
import warnings
import random
import tempfile
import numpy as np
import pandas as pd
import SimpleITK as sitk
from multiprocessing import Pool, cpu_count
from radiomics import featureextractor
from scipy.ndimage import binary_erosion, binary_dilation

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
from config_v4 import (OUT_PREPROC, OUT_PERI, OUT_HABITAT, OUT_PERI_HAB,
                        OUT_FEAT_RAW, OUT_FEAT_ICC, LOG_DIR,
                        PYRADIOMICS_PARAMS, FEATURE_CLASSES, IMAGE_TYPES,
                        PRIMARY_PERI_MM, REGIONS,
                        ICC_N_CASES, ICC_N_PERTURBATIONS, ICC_THRESHOLD,
                        RANDOM_STATE)
from step4_features_v4 import build_extractor, get_mask_path

os.makedirs(OUT_FEAT_ICC, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'step4b_v4.log'), mode='w'),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)

# 并行 worker 数: 留 2 核给系统
N_WORKERS = max(1, min(cpu_count() - 2, 12))


def perturb_mask(mask_arr: np.ndarray, method: int) -> np.ndarray:
    """5种 mask 扰动"""
    rng = np.random.RandomState(RANDOM_STATE + method)
    if method == 0:
        return binary_erosion(mask_arr, iterations=1).astype(np.uint8)
    elif method == 1:
        return binary_dilation(mask_arr, iterations=1).astype(np.uint8)
    elif method == 2:
        perturbed = mask_arr.copy()
        coords = np.argwhere(perturbed > 0)
        n_drop = max(1, int(len(coords) * 0.05))
        drop_idx = rng.choice(len(coords), n_drop, replace=False)
        for idx in drop_idx:
            z, y, x = coords[idx]
            perturbed[z, y, x] = 0
        return perturbed
    elif method == 3:
        dilated = binary_dilation(mask_arr, iterations=1).astype(np.uint8)
        boundary = dilated - mask_arr
        boundary_coords = np.argwhere(boundary > 0)
        if len(boundary_coords) == 0:
            return mask_arr.copy()
        n_add = max(1, int(mask_arr.sum() * 0.05))
        add_idx = rng.choice(len(boundary_coords),
                             min(n_add, len(boundary_coords)), replace=False)
        perturbed = mask_arr.copy()
        for idx in add_idx:
            z, y, x = boundary_coords[idx]
            perturbed[z, y, x] = 1
        return perturbed
    else:
        shift = rng.choice([-1, 0, 1], size=3)
        return np.roll(np.roll(np.roll(
            mask_arr, shift[0], axis=0), shift[1], axis=1), shift[2], axis=2
        ).astype(np.uint8)


def compute_icc31(data: np.ndarray) -> float:
    """ICC(3,1) two-way mixed, consistency"""
    n, k = data.shape
    if n < 2 or k < 2:
        return 0.0
    mean_total = data.mean()
    ss_rows = k * np.sum((data.mean(axis=1) - mean_total) ** 2)
    ss_cols = n * np.sum((data.mean(axis=0) - mean_total) ** 2)
    ss_total = np.sum((data - mean_total) ** 2)
    ss_error = ss_total - ss_rows - ss_cols

    ms_rows = ss_rows / max(1, n - 1)
    ms_error = ss_error / max(1, (n - 1) * (k - 1))

    denom = ms_rows + (k - 1) * ms_error
    if denom < 1e-12:
        return 0.0
    icc = (ms_rows - ms_error) / denom
    return max(0.0, float(icc))


def _process_single_case(args: tuple) -> list[dict] | None:
    """Worker: 处理单个 case 的原始 + 5次扰动特征提取

    每个 worker 独立创建 extractor (PyRadiomics 非线程安全)
    返回 list of feature dicts, 或 None (失败)
    """
    case_id, region, worker_id = args

    # 抑制 PyRadiomics 日志 (子进程内)
    logging.getLogger('radiomics').setLevel(logging.ERROR)

    extractor = build_extractor()

    img_path = os.path.join(OUT_PREPROC, f'{case_id}_image.nii.gz')
    mask_path = get_mask_path(case_id, region)
    mask_sitk = sitk.ReadImage(mask_path)
    mask_arr = sitk.GetArrayFromImage(mask_sitk).astype(np.uint8)

    measurements = []

    # Original mask
    try:
        result = extractor.execute(img_path, mask_path)
        feats = {k: float(v) for k, v in result.items()
                 if not k.startswith('diagnostics_') and hasattr(v, '__float__')}
        measurements.append(feats)
    except Exception:
        return None

    # Perturbations
    for p in range(ICC_N_PERTURBATIONS):
        perturbed = perturb_mask(mask_arr, p)
        if perturbed.sum() < 10:
            continue
        perturbed_sitk = sitk.GetImageFromArray(perturbed)
        perturbed_sitk.CopyInformation(mask_sitk)

        # 用 tempfile 避免 worker 间文件名冲突
        tmp_fd, tmp_path = tempfile.mkstemp(
            suffix='.nii.gz',
            prefix=f'icc_{case_id}_{region}_w{worker_id}_p{p}_',
            dir=OUT_FEAT_ICC
        )
        os.close(tmp_fd)
        try:
            sitk.WriteImage(perturbed_sitk, tmp_path, useCompression=True)
            result = extractor.execute(img_path, tmp_path)
            feats = {k: float(v) for k, v in result.items()
                     if not k.startswith('diagnostics_') and hasattr(v, '__float__')}
            measurements.append(feats)
        except Exception:
            pass
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    if len(measurements) >= 3:
        return measurements
    return None


def compute_icc_for_region(region: str) -> dict:
    """计算一个区域所有特征的 ICC — 并行版"""
    # 获取有效 case 列表
    cases = sorted([
        f.replace('_image.nii.gz', '')
        for f in os.listdir(OUT_PREPROC)
        if f.endswith('_image.nii.gz')
    ])

    # 过滤有 mask 的 case
    valid_cases = []
    for cid in cases:
        mask_path = get_mask_path(cid, region)
        if os.path.exists(mask_path):
            mask_sitk = sitk.ReadImage(mask_path)
            if sitk.GetArrayFromImage(mask_sitk).sum() >= 10:
                valid_cases.append(cid)

    n_select = min(ICC_N_CASES, len(valid_cases))
    rng = random.Random(RANDOM_STATE)
    selected = rng.sample(valid_cases, n_select)
    log.info(f'  [{region}] ICC: {n_select} cases, {N_WORKERS} workers')

    # 构建 worker 参数
    task_args = [(cid, region, i) for i, cid in enumerate(selected)]

    # 并行提取
    all_measurements = []
    with Pool(processes=N_WORKERS) as pool:
        results = pool.map(_process_single_case, task_args)

    for r in results:
        if r is not None:
            all_measurements.append(r)

    log.info(f'  [{region}] {len(all_measurements)}/{n_select} cases 有效')

    # 获取特征名
    feat_names = None
    for case_meas in all_measurements:
        if case_meas:
            feat_names = sorted(case_meas[0].keys())
            break

    if not feat_names or len(all_measurements) < 5:
        log.warning(f'  [{region}] 不足以计算 ICC')
        return {}

    # 计算每个特征的 ICC
    icc_scores = {}
    for feat_name in feat_names:
        data_rows = []
        for case_meas in all_measurements:
            row = [m.get(feat_name, np.nan) for m in case_meas]
            if not any(np.isnan(r) for r in row):
                data_rows.append(row)

        if len(data_rows) < 5:
            icc_scores[feat_name] = 0.0
            continue

        min_cols = min(len(r) for r in data_rows)
        data = np.array([r[:min_cols] for r in data_rows])

        if data.std() < 1e-10:
            icc_scores[feat_name] = 1.0
            continue

        icc_scores[feat_name] = compute_icc31(data)

    return icc_scores


def main():
    log.info(f'=== Step 4b v4 ICC (parallel, {N_WORKERS} workers) ===')

    for region in REGIONS:
        icc_path = os.path.join(OUT_FEAT_ICC, f'icc_scores_{region}.json')
        stable_path = os.path.join(OUT_FEAT_ICC, f'stable_features_{region}.txt')

        if os.path.exists(icc_path):
            log.info(f'  {region}: ICC 已存在，跳过')
            continue

        log.info(f'计算 ICC: {region}')
        icc_scores = compute_icc_for_region(region)

        if not icc_scores:
            continue

        with open(icc_path, 'w') as f:
            json.dump(icc_scores, f, indent=2)

        stable = [k for k, v in icc_scores.items() if v >= ICC_THRESHOLD]
        with open(stable_path, 'w') as f:
            f.write('\n'.join(sorted(stable)))

        total = len(icc_scores)
        n_stable = len(stable)
        log.info(f'  {region}: {n_stable}/{total} features ICC>={ICC_THRESHOLD} '
                 f'({n_stable/total*100:.0f}%)')

    log.info('=== Step 4b v4 ICC (parallel) 完成 ===')


if __name__ == '__main__':
    main()
