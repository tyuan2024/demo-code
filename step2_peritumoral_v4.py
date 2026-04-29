"""step2: 围瘤区掩膜 5/10/15mm"""
import os
import sys
import json
import logging
import multiprocessing as mp
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from config_v4 import OUT_PREPROC, OUT_PERI, LOG_DIR, PERI_MARGINS_MM

os.makedirs(OUT_PERI, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'step2_v4.log'), mode='w'),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)


def process_case(case_id: str) -> dict:
    lbl_path = os.path.join(OUT_PREPROC, f'{case_id}_label.nii.gz')
    if not os.path.exists(lbl_path):
        return {'case_id': case_id, 'status': 'missing'}

    all_exist = all(
        os.path.exists(os.path.join(OUT_PERI, f'{case_id}_peri{m}mm.nii.gz'))
        for m in PERI_MARGINS_MM
    )
    if all_exist:
        return {'case_id': case_id, 'status': 'skipped'}

    try:
        label = sitk.ReadImage(lbl_path, sitk.sitkUInt8)
        spacing = label.GetSpacing()
        results_mm = {}

        for margin_mm in PERI_MARGINS_MM:
            radius = [max(1, int(round(margin_mm / spacing[i]))) for i in range(3)]
            dilated = sitk.BinaryDilate(label, radius)
            peri = sitk.And(dilated, sitk.Not(label))
            peri = sitk.Cast(peri, sitk.sitkUInt8)

            out_path = os.path.join(OUT_PERI, f'{case_id}_peri{margin_mm}mm.nii.gz')
            sitk.WriteImage(peri, out_path, useCompression=True)
            n_vox = int(sitk.GetArrayFromImage(peri).sum())
            results_mm[f'peri{margin_mm}mm'] = n_vox

        return {'case_id': case_id, 'status': 'ok', 'voxels': results_mm}
    except Exception as e:
        return {'case_id': case_id, 'status': 'error', 'msg': str(e)}


def main():
    cases = sorted([
        f.replace('_label.nii.gz', '')
        for f in os.listdir(OUT_PREPROC)
        if f.endswith('_label.nii.gz')
    ])
    log.info(f'共 {len(cases)} 例，生成围瘤区掩膜 ...')

    n_workers = min(8, mp.cpu_count())
    results = []
    with mp.Pool(processes=n_workers) as pool:
        for r in tqdm(pool.imap_unordered(process_case, cases),
                      total=len(cases), desc='Step2 v4'):
            results.append(r)

    with open(os.path.join(LOG_DIR, 'step2_v4_summary.json'), 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    ok = sum(1 for r in results if r['status'] == 'ok')
    skip = sum(1 for r in results if r['status'] == 'skipped')
    log.info(f'完成: ok={ok}  skipped={skip}')


if __name__ == '__main__':
    main()
