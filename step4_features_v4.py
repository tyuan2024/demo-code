"""step4: pyradiomics 8区域特征 + 生态学特征"""
import os
import sys
import json
import logging
import warnings
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm
from radiomics import featureextractor
from scipy.ndimage import label as ndlabel, binary_dilation

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
from config_v4 import (OUT_PREPROC, OUT_PERI, OUT_HABITAT, OUT_PERI_HAB,
                        OUT_FEAT_RAW, OUT_FEAT_ECO, LOG_DIR,
                        PYRADIOMICS_PARAMS, FEATURE_CLASSES, IMAGE_TYPES,
                        PRIMARY_PERI_MM, REGIONS)

os.makedirs(OUT_FEAT_RAW, exist_ok=True)
os.makedirs(OUT_FEAT_ECO, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'step4_v4.log'), mode='w'),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)


def build_extractor():
    """构建 PyRadiomics 提取器"""
    settings = dict(PYRADIOMICS_PARAMS)
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)

    # 禁用所有，再逐个启用
    extractor.disableAllFeatures()
    for fc in FEATURE_CLASSES:
        extractor.enableFeatureClassByName(fc)

    extractor.disableAllImageTypes()
    for img_type, params in IMAGE_TYPES.items():
        extractor.enableImageTypeByName(img_type, customArgs=params)

    return extractor


def get_mask_path(case_id: str, region: str) -> str:
    if region == 'intra':
        return os.path.join(OUT_PREPROC, f'{case_id}_label.nii.gz')
    elif region == 'peri10mm':
        return os.path.join(OUT_PERI, f'{case_id}_peri{PRIMARY_PERI_MM}mm.nii.gz')
    elif region.startswith('H'):
        return os.path.join(OUT_HABITAT, f'{case_id}_{region}.nii.gz')
    elif region.startswith('periH'):
        return os.path.join(OUT_PERI_HAB, f'{case_id}_{region}.nii.gz')
    return ''


def extract_case_region(extractor, case_id: str, region: str) -> dict:
    img_path = os.path.join(OUT_PREPROC, f'{case_id}_image.nii.gz')
    mask_path = get_mask_path(case_id, region)

    if not (os.path.exists(img_path) and os.path.exists(mask_path)):
        return None

    # 检查 mask 体素数
    mask_sitk = sitk.ReadImage(mask_path)
    mask_arr = sitk.GetArrayFromImage(mask_sitk)
    n_vox = int(mask_arr.sum())
    if n_vox < 10:
        return None

    try:
        result = extractor.execute(img_path, mask_path)
        feats = {}
        for key, val in result.items():
            if key.startswith('diagnostics_'):
                continue
            feats[key] = float(val) if hasattr(val, '__float__') else val
        feats['case_id'] = case_id
        feats['region'] = region
        feats['n_voxels'] = n_vox
        return feats
    except Exception as e:
        log.warning(f'  {case_id}/{region}: {str(e)[:100]}')
        return None


def compute_ecological_features(case_id: str) -> dict:
    """计算 habitat 生态学特征"""
    habitat_path = os.path.join(OUT_HABITAT, f'{case_id}_habitat.nii.gz')
    img_path = os.path.join(OUT_PREPROC, f'{case_id}_image.nii.gz')

    if not (os.path.exists(habitat_path) and os.path.exists(img_path)):
        return None

    habitat_sitk = sitk.ReadImage(habitat_path)
    img_sitk = sitk.ReadImage(img_path)
    habitat_mask = sitk.GetArrayFromImage(habitat_sitk).astype(np.int32)
    image_arr = sitk.GetArrayFromImage(img_sitk).astype(np.float32)

    features = {'case_id': case_id}
    total_tumor = (habitat_mask > 0).sum()
    if total_tumor == 0:
        return None

    # Volume proportions
    proportions = []
    for h in [1, 2, 3]:
        vol = (habitat_mask == h).sum()
        p = vol / total_tumor
        features[f'habitat_proportion_H{h}'] = p
        proportions.append(p)

    # Shannon entropy
    props_pos = [p for p in proportions if p > 0]
    shannon = -sum(p * np.log(p) for p in props_pos) if props_pos else 0
    features['habitat_shannon_entropy'] = shannon

    # Simpson index
    simpson = 1 - sum(p ** 2 for p in proportions)
    features['habitat_simpson_index'] = simpson

    # Evenness
    features['habitat_evenness'] = shannon / np.log(3) if shannon > 0 else 0

    # Inter-habitat intensity contrasts
    means = {}
    for h in [1, 2, 3]:
        vals = image_arr[habitat_mask == h]
        means[h] = float(vals.mean()) if len(vals) > 0 else 0

    features['contrast_H1_H2'] = abs(means[1] - means[2])
    features['contrast_H1_H3'] = abs(means[1] - means[3])
    features['contrast_H2_H3'] = abs(means[2] - means[3])
    features['contrast_max'] = max(features['contrast_H1_H2'],
                                    features['contrast_H1_H3'],
                                    features['contrast_H2_H3'])

    # Boundary ratios
    for h in [1, 2, 3]:
        h_mask = (habitat_mask == h)
        dilated = binary_dilation(h_mask, iterations=1)
        boundary = dilated & ~h_mask & (habitat_mask > 0)
        features[f'habitat_boundary_ratio_H{h}'] = (
            boundary.sum() / max(1, h_mask.sum()))

    # Fragmentation
    for h in [1, 2, 3]:
        h_mask = (habitat_mask == h)
        labeled, n_comp = ndlabel(h_mask)
        if n_comp > 0:
            sizes = [(labeled == i).sum() for i in range(1, n_comp + 1)]
            features[f'habitat_fragmentation_H{h}'] = n_comp
            features[f'habitat_largest_component_ratio_H{h}'] = (
                max(sizes) / max(1, h_mask.sum()))
        else:
            features[f'habitat_fragmentation_H{h}'] = 0
            features[f'habitat_largest_component_ratio_H{h}'] = 0

    return features


def main():
    cases = sorted([
        f.replace('_image.nii.gz', '')
        for f in os.listdir(OUT_PREPROC)
        if f.endswith('_image.nii.gz')
    ])
    log.info(f'共 {len(cases)} 例，提取 PyRadiomics 特征 (8区域) ...')

    extractor = build_extractor()

    # PyRadiomics per region
    for region in REGIONS:
        out_path = os.path.join(OUT_FEAT_RAW, f'features_{region}.csv')
        if os.path.exists(out_path):
            log.info(f'  {region}: 已存在，跳过')
            continue

        log.info(f'  提取 {region} ...')
        all_feats = []
        for case_id in tqdm(cases, desc=f'  {region}'):
            feats = extract_case_region(extractor, case_id, region)
            if feats is not None:
                all_feats.append(feats)

        if all_feats:
            df = pd.DataFrame(all_feats)
            df.to_csv(out_path, index=False)
            feat_cols = [c for c in df.columns
                         if c not in ('case_id', 'region', 'n_voxels')]
            log.info(f'    {region}: {len(df)} cases, {len(feat_cols)} features')
        else:
            log.warning(f'    {region}: 无有效特征')

    # Ecological features
    eco_path = os.path.join(OUT_FEAT_ECO, 'ecological_features.csv')
    if not os.path.exists(eco_path):
        log.info('提取生态学特征 ...')
        eco_feats = []
        for case_id in tqdm(cases, desc='  Ecological'):
            feats = compute_ecological_features(case_id)
            if feats is not None:
                eco_feats.append(feats)

        if eco_feats:
            df_eco = pd.DataFrame(eco_feats)
            df_eco.to_csv(eco_path, index=False)
            log.info(f'  生态学特征: {len(df_eco)} cases, '
                     f'{len(df_eco.columns) - 1} features')
    else:
        log.info('生态学特征已存在，跳过')

    log.info('=== Step 4 v4 完成 ===')


if __name__ == '__main__':
    main()
