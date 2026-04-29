"""step4c: DL特征 ResNet50+DenseNet121 -> PCA50"""
import os
import sys
import json
import logging
import warnings
import pickle
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
from config_v4 import (OUT_PREPROC, OUT_FEAT_DL, LOG_DIR,
                        DL_PCA_COMPONENTS, RANDOM_STATE)
from data_split_v4 import get_or_create_split

os.makedirs(OUT_FEAT_DL, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'step4c_v4.log'), mode='w'),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DLFeatureExtractor:
    def __init__(self):
        # ResNet50
        self.resnet = models.resnet50(weights='IMAGENET1K_V2')
        self.resnet.fc = torch.nn.Identity()
        self.resnet.eval().to(DEVICE)

        # DenseNet121
        self.densenet = models.densenet121(weights='IMAGENET1K_V1')
        self.densenet.classifier = torch.nn.Identity()
        self.densenet.eval().to(DEVICE)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def _extract_slices(self, image_arr: np.ndarray,
                        mask_arr: np.ndarray,
                        model: torch.nn.Module) -> np.ndarray:
        """提取最大肿瘤切片 + 相邻切片的特征"""
        slice_areas = mask_arr.sum(axis=(1, 2))
        best_z = int(np.argmax(slice_areas))
        slices_z = [max(0, best_z - 1), best_z,
                    min(image_arr.shape[0] - 1, best_z + 1)]

        features_per_slice = []
        for z in slices_z:
            sl = image_arr[z]
            sl_mask = mask_arr[z]

            coords = np.argwhere(sl_mask > 0)
            if len(coords) == 0:
                continue
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            pad = 20
            y_min = max(0, y_min - pad)
            x_min = max(0, x_min - pad)
            y_max = min(sl.shape[0], y_max + pad + 1)
            x_max = min(sl.shape[1], x_max + pad + 1)

            crop = sl[y_min:y_max, x_min:x_max]
            if crop.size == 0:
                continue

            # Normalize to 0-255
            cmin, cmax = crop.min(), crop.max()
            if cmax - cmin < 1e-6:
                crop_norm = np.zeros_like(crop, dtype=np.uint8)
            else:
                crop_norm = ((crop - cmin) / (cmax - cmin) * 255).astype(np.uint8)

            crop_rgb = np.stack([crop_norm] * 3, axis=-1)

            tensor = self.transform(crop_rgb).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                feat = model(tensor).cpu().numpy().flatten()
            features_per_slice.append(feat)

        if not features_per_slice:
            return None
        return np.mean(features_per_slice, axis=0).astype(np.float32)

    def extract_case(self, case_id: str) -> dict:
        img_path = os.path.join(OUT_PREPROC, f'{case_id}_image.nii.gz')
        lbl_path = os.path.join(OUT_PREPROC, f'{case_id}_label.nii.gz')

        img_sitk = sitk.ReadImage(img_path)
        lbl_sitk = sitk.ReadImage(lbl_path)
        image_arr = sitk.GetArrayFromImage(img_sitk).astype(np.float32)
        mask_arr = sitk.GetArrayFromImage(lbl_sitk).astype(np.uint8)

        resnet_feat = self._extract_slices(image_arr, mask_arr, self.resnet)
        densenet_feat = self._extract_slices(image_arr, mask_arr, self.densenet)

        return {
            'case_id': case_id,
            'resnet50': resnet_feat,
            'densenet121': densenet_feat,
        }


def main():
    cases = sorted([
        f.replace('_image.nii.gz', '')
        for f in os.listdir(OUT_PREPROC)
        if f.endswith('_image.nii.gz')
    ])
    log.info(f'共 {len(cases)} 例，DL 特征提取 ...')

    raw_resnet_path = os.path.join(OUT_FEAT_DL, 'resnet50_raw_2048.csv')
    pca_resnet_path = os.path.join(OUT_FEAT_DL, 'resnet50_pca50.csv')
    pca_dense_path = os.path.join(OUT_FEAT_DL, 'densenet121_pca50.csv')

    if all(os.path.exists(p) for p in [pca_resnet_path, pca_dense_path]):
        log.info('DL 特征已存在，跳过')
        return

    extractor = DLFeatureExtractor()

    resnet_feats = {}
    densenet_feats = {}

    for case_id in tqdm(cases, desc='DL Features'):
        try:
            result = extractor.extract_case(case_id)
            if result['resnet50'] is not None:
                resnet_feats[case_id] = result['resnet50']
            if result['densenet121'] is not None:
                densenet_feats[case_id] = result['densenet121']
        except Exception as e:
            log.warning(f'  {case_id}: {str(e)[:100]}')

    log.info(f'ResNet50: {len(resnet_feats)} cases, '
             f'DenseNet121: {len(densenet_feats)} cases')

    # Save raw ResNet50
    common_cases = sorted(set(resnet_feats.keys()) & set(densenet_feats.keys()))
    if not common_cases:
        log.error('无有效 DL 特征')
        return

    resnet_mat = np.array([resnet_feats[c] for c in common_cases])
    dense_mat = np.array([densenet_feats[c] for c in common_cases])

    # Raw ResNet50
    df_raw = pd.DataFrame(resnet_mat,
                           columns=[f'resnet_{i}' for i in range(resnet_mat.shape[1])])
    df_raw.insert(0, 'case_id', common_cases)
    df_raw.to_csv(raw_resnet_path, index=False)

    # PCA (fit on training set only)
    split = get_or_create_split()
    train_ids = set(split['train'])
    train_mask = np.array([c in train_ids for c in common_cases])

    n_comp = min(DL_PCA_COMPONENTS, resnet_mat.shape[0] - 1,
                 resnet_mat.shape[1])

    # ResNet50 PCA
    pca_resnet = PCA(n_components=n_comp, random_state=RANDOM_STATE)
    pca_resnet.fit(resnet_mat[train_mask])
    resnet_pca = pca_resnet.transform(resnet_mat)
    log.info(f'ResNet50 PCA: {n_comp} components, '
             f'explained var={pca_resnet.explained_variance_ratio_.sum():.3f}')

    df_rpca = pd.DataFrame(resnet_pca,
                            columns=[f'resnet_pca_{i}' for i in range(n_comp)])
    df_rpca.insert(0, 'case_id', common_cases)
    df_rpca.to_csv(pca_resnet_path, index=False)

    # DenseNet121 PCA
    n_comp_d = min(DL_PCA_COMPONENTS, dense_mat.shape[0] - 1,
                   dense_mat.shape[1])
    pca_dense = PCA(n_components=n_comp_d, random_state=RANDOM_STATE)
    pca_dense.fit(dense_mat[train_mask])
    dense_pca = pca_dense.transform(dense_mat)
    log.info(f'DenseNet121 PCA: {n_comp_d} components, '
             f'explained var={pca_dense.explained_variance_ratio_.sum():.3f}')

    df_dpca = pd.DataFrame(dense_pca,
                            columns=[f'densenet_pca_{i}' for i in range(n_comp_d)])
    df_dpca.insert(0, 'case_id', common_cases)
    df_dpca.to_csv(pca_dense_path, index=False)

    # Save PCA models
    with open(os.path.join(OUT_FEAT_DL, 'pca_model.pkl'), 'wb') as f:
        pickle.dump({'resnet50_pca': pca_resnet, 'densenet121_pca': pca_dense}, f)

    log.info('=== Step 4c v4 DL 特征完成 ===')


if __name__ == '__main__':
    main()
