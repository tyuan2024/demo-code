"""step3b: 围瘤区habitat聚类 (peri10mm -> periH1/H2/H3)"""
import os
import sys
import json
import logging
import warnings
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
from config_v4 import (OUT_PREPROC, OUT_PERI, OUT_PERI_HAB, LOG_DIR,
                        HABITAT_K, HABITAT_CLUSTER_WIN, PRIMARY_PERI_MM,
                        RANDOM_STATE)
from step3_habitat_v4 import extract_12dim_features, cluster_habitat, reorder_labels_by_intensity

os.makedirs(OUT_PERI_HAB, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'step3b_v4.log'), mode='w'),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)


def process_case(case_id: str) -> dict:
    img_path = os.path.join(OUT_PREPROC, f'{case_id}_image.nii.gz')
    peri_path = os.path.join(OUT_PERI, f'{case_id}_peri{PRIMARY_PERI_MM}mm.nii.gz')

    if not (os.path.exists(img_path) and os.path.exists(peri_path)):
        return {'case_id': case_id, 'status': 'missing_input'}

    expected = [os.path.join(OUT_PERI_HAB, f'{case_id}_periH{h}.nii.gz')
                for h in [1, 2, 3]]
    if all(os.path.exists(p) for p in expected):
        return {'case_id': case_id, 'status': 'skipped'}

    try:
        img_sitk = sitk.ReadImage(img_path)
        peri_sitk = sitk.ReadImage(peri_path)
        arr = sitk.GetArrayFromImage(img_sitk).astype(np.float32)
        peri_mask = sitk.GetArrayFromImage(peri_sitk).astype(bool)

        n_peri = int(peri_mask.sum())
        if n_peri < 30:
            return {'case_id': case_id, 'status': 'too_small',
                    'peri_vox': n_peri}

        # Crop
        coords = np.argwhere(peri_mask)
        zmin, ymin, xmin = coords.min(axis=0)
        zmax, ymax, xmax = coords.max(axis=0)
        pad = HABITAT_CLUSTER_WIN + 2
        sl = (slice(max(0, zmin - pad), min(arr.shape[0], zmax + pad + 1)),
              slice(max(0, ymin - pad), min(arr.shape[1], ymax + pad + 1)),
              slice(max(0, xmin - pad), min(arr.shape[2], xmax + pad + 1)))

        arr_crop = arr[sl]
        mask_crop = peri_mask[sl]

        X = extract_12dim_features(arr_crop, mask_crop, win=HABITAT_CLUSTER_WIN)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        labels, sil = cluster_habitat(X, k=HABITAT_K)
        intensities = arr_crop[mask_crop]
        labels = reorder_labels_by_intensity(labels, intensities)

        full_cluster = np.zeros(arr.shape, dtype=np.uint8)
        tmp = np.zeros(mask_crop.shape, dtype=np.uint8)
        tmp[mask_crop] = labels + 1
        full_cluster[sl] = tmp

        subregion_voxels = {}
        for h in [1, 2, 3]:
            sub = (full_cluster == h).astype(np.uint8)
            sub_sitk = sitk.GetImageFromArray(sub)
            sub_sitk.CopyInformation(img_sitk)
            sitk.WriteImage(sub_sitk,
                            os.path.join(OUT_PERI_HAB, f'{case_id}_periH{h}.nii.gz'),
                            useCompression=True)
            subregion_voxels[f'periH{h}'] = int(sub.sum())

        return {
            'case_id': case_id, 'status': 'ok',
            'peri_voxels': n_peri,
            'silhouette': round(sil, 4),
            'subregion_voxels': subregion_voxels,
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
    log.info(f'共 {len(cases)} 例，Peri-Habitat v4 ...')

    results = []
    for case_id in tqdm(cases, desc='PeriHabitat v4'):
        r = process_case(case_id)
        results.append(r)
        if r['status'] == 'error':
            log.warning(f"  {r['case_id']}: {r.get('msg','')[:120]}")

    with open(os.path.join(LOG_DIR, 'step3b_v4_summary.json'), 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    ok = sum(1 for r in results if r['status'] == 'ok')
    skip = sum(1 for r in results if r['status'] == 'skipped')
    error = sum(1 for r in results if r['status'] == 'error')
    log.info(f'完成: ok={ok}  skipped={skip}  error={error}')


if __name__ == '__main__':
    main()
