"""step1: 预处理 — resample 1mm + zscore float32"""
import os
import sys
import json
import logging
import multiprocessing as mp
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from config_v4 import (DATA_DIR, LABEL_DIR, OUT_PREPROC, LOG_DIR,
                        TARGET_SPACING, INTERP_IMAGE, INTERP_MASK,
                        N4_SHRINK_FACTOR, N4_FITTING_LEVELS,
                        N4_CONVERGENCE_THR, N4_MAX_ITERATIONS)

os.makedirs(OUT_PREPROC, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'step1_v4.log'), mode='w'),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)

INTERP_MAP = {
    'bspline': sitk.sitkBSpline,
    'linear':  sitk.sitkLinear,
    'nearest': sitk.sitkNearestNeighbor,
}


def n4_correction(image: sitk.Image, mask: sitk.Image) -> sitk.Image:
    image_f32 = sitk.Cast(image, sitk.sitkFloat32)
    shrunk_img = sitk.Shrink(image_f32, [N4_SHRINK_FACTOR] * 3)
    body_mask = sitk.BinaryThreshold(shrunk_img, lowerThreshold=1.0)
    body_mask = sitk.Cast(body_mask, sitk.sitkUInt8)

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations(N4_FITTING_LEVELS)
    corrector.SetConvergenceThreshold(N4_CONVERGENCE_THR)
    corrector.SetNumberOfControlPoints([4, 4, 4])

    try:
        corrector.Execute(shrunk_img, body_mask)
        log_bias = corrector.GetLogBiasFieldAsImage(image_f32)
        corrected = image_f32 / sitk.Exp(log_bias)
        return corrected
    except Exception as e:
        log.warning(f'N4 failed, using original: {e}')
        return image_f32


def resample(image: sitk.Image,
             target_spacing=(1.0, 1.0, 1.0),
             interpolator=sitk.sitkBSpline,
             default_value: float = 0.0) -> sitk.Image:
    orig_spacing = image.GetSpacing()
    orig_size = image.GetSize()
    new_size = [
        int(round(orig_size[i] * orig_spacing[i] / target_spacing[i]))
        for i in range(3)
    ]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(default_value)
    return resampler.Execute(image)


def zscore_norm_float32(image: sitk.Image, mask: sitk.Image) -> sitk.Image:
    """Z-score → clip [-3,3] → 保留 float32"""
    arr = sitk.GetArrayFromImage(image).astype(np.float32)
    marr = sitk.GetArrayFromImage(mask).astype(bool)

    roi_vals = arr[marr]
    mu = float(roi_vals.mean())
    std = float(roi_vals.std())
    if std < 1e-6:
        std = 1.0

    arr_norm = (arr - mu) / std
    arr_norm = np.clip(arr_norm, -3.0, 3.0).astype(np.float32)

    out = sitk.GetImageFromArray(arr_norm)
    out.CopyInformation(image)
    return out


def process_case(case_id: str) -> dict:
    img_path = os.path.join(DATA_DIR, f'{case_id}.nii')
    lbl_path = os.path.join(LABEL_DIR, f'{case_id}.nii')
    out_img = os.path.join(OUT_PREPROC, f'{case_id}_image.nii.gz')
    out_lbl = os.path.join(OUT_PREPROC, f'{case_id}_label.nii.gz')

    if os.path.exists(out_img) and os.path.exists(out_lbl):
        return {'case_id': case_id, 'status': 'skipped'}

    try:
        image = sitk.ReadImage(img_path, sitk.sitkFloat32)
        label = sitk.ReadImage(lbl_path, sitk.sitkUInt8)
        orig_spacing = image.GetSpacing()
        orig_size = image.GetSize()

        # N4对CT基本没效果(CT没有bias field)，保留是因为跑完了不想重跑
        image_n4 = n4_correction(image, label)
        image_res = resample(image_n4, TARGET_SPACING, INTERP_MAP[INTERP_IMAGE])
        label_res = resample(label, TARGET_SPACING, INTERP_MAP[INTERP_MASK], default_value=0)
        label_res = sitk.Cast(label_res > 0, sitk.sitkUInt8)

        image_norm = zscore_norm_float32(image_res, label_res)

        n_tumor_vox = int(sitk.GetArrayFromImage(label_res).sum())
        sitk.WriteImage(image_norm, out_img, useCompression=True)
        sitk.WriteImage(label_res, out_lbl, useCompression=True)

        return {
            'case_id': case_id, 'status': 'ok',
            'orig_spacing': list(orig_spacing),
            'orig_size': list(orig_size),
            'new_size': list(image_norm.GetSize()),
            'tumor_voxels': n_tumor_vox,
        }
    except Exception as e:
        import traceback
        return {'case_id': case_id, 'status': 'error', 'msg': str(e),
                'tb': traceback.format_exc()}


def main():
    cases = sorted([
        f.replace('.nii', '')
        for f in os.listdir(DATA_DIR)
        if f.endswith('.nii')
    ])
    log.info(f'共 {len(cases)} 例，开始预处理 v4 (N4+Resample+Zscore→float32) ...')

    n_workers = min(12, mp.cpu_count())
    log.info(f'并行 workers: {n_workers}')

    results = []
    with mp.Pool(processes=n_workers) as pool:
        for r in tqdm(pool.imap_unordered(process_case, cases),
                      total=len(cases), desc='Step1 v4'):
            results.append(r)
            if r['status'] == 'error':
                log.warning(f"  FAILED {r['case_id']}: {r.get('msg','')[:120]}")

    summary_path = os.path.join(LOG_DIR, 'step1_v4_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    ok = sum(1 for r in results if r['status'] == 'ok')
    skip = sum(1 for r in results if r['status'] == 'skipped')
    error = sum(1 for r in results if r['status'] == 'error')
    log.info(f'完成: ok={ok}  skipped={skip}  error={error}')


if __name__ == '__main__':
    main()
