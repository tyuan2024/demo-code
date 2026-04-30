"""config_v4.py — 全局配置"""
import os

# ---- 路径 ----
BASE_DIR      = '/root/autodl-tmp/影像/fudanzhongshan3'
DATA_DIR      = os.path.join(BASE_DIR, 'data')
LABEL_DIR     = os.path.join(BASE_DIR, 'label')
CLINICAL_PATH = os.path.join(BASE_DIR, '临床信息.xlsx')

PROJECT_DIR   = '/root/autodl-tmp/影像/habitat_project_v2'
OUT_V4        = os.path.join(PROJECT_DIR, 'output_v4')
OUT_PREPROC   = os.path.join(OUT_V4, 'preprocessed')
OUT_PERI      = os.path.join(OUT_V4, 'peritumoral')
OUT_HABITAT   = os.path.join(OUT_V4, 'habitat_masks')
OUT_PERI_HAB  = os.path.join(OUT_V4, 'peri_habitat_masks')
OUT_FEATURES  = os.path.join(OUT_V4, 'features')
OUT_FEAT_RAW  = os.path.join(OUT_FEATURES, 'raw')
OUT_FEAT_ICC  = os.path.join(OUT_FEATURES, 'icc')
OUT_FEAT_ECO  = os.path.join(OUT_FEATURES, 'ecological')
OUT_FEAT_DL   = os.path.join(OUT_FEATURES, 'dl_features')
OUT_FEAT_SEL  = os.path.join(OUT_FEATURES, 'selected')
OUT_MODELS    = os.path.join(OUT_V4, 'models')
OUT_FIGURES   = os.path.join(OUT_V4, 'figures')
ALL_FIGURE    = os.path.join(PROJECT_DIR, 'all figure')
LOG_DIR       = os.path.join(OUT_V4, 'logs')

# ---- 预处理 ----
TARGET_SPACING = (1.0, 1.0, 1.0)
INTERP_IMAGE   = 'bspline'
INTERP_MASK    = 'nearest'
N4_SHRINK_FACTOR    = 4
N4_FITTING_LEVELS   = [50, 50, 50, 50]
N4_CONVERGENCE_THR  = 1e-6
N4_MAX_ITERATIONS   = 50

# ---- 围瘤区 ----
PERI_MARGINS_MM = [5, 10, 15]
PRIMARY_PERI_MM = 10

# ---- Habitat 聚类 ----
HABITAT_K = 3                       # 固定 K=3
HABITAT_CLUSTER_WIN = 3             # 3x3x3 窗口
HABITAT_GLCM_LEVELS = 16           # GLCM 量化级数
HABITAT_LOG_SIGMAS = [1.0, 2.0, 3.0, 5.0]  # LoG 尺度
HABITAT_KMEANS_NINIT = 20
HABITAT_KMEANS_MAXITER = 500
RANDOM_STATE = 42

# ---- 特征提取 ----
PYRADIOMICS_PARAMS = {
    'binWidth': 25,
    'resampledPixelSpacing': None,
    'interpolator': None,
    'normalize': False,
    'minimumROIDimensions': 1,
    'minimumROISize': 10,
    'geometryTolerance': 1e-6,
    'correctMask': True,
}
FEATURE_CLASSES = ['shape', 'firstorder', 'glcm', 'glrlm', 'glszm', 'gldm']
IMAGE_TYPES = {
    'Original': {},
    'LoG': {'sigma': [2.0, 3.0, 5.0]},
    'Wavelet': {},
}

# ---- ICC ----
ICC_N_CASES = 50
ICC_N_PERTURBATIONS = 5
ICC_THRESHOLD = 0.75

# ---- DL特征 ----
DL_PCA_COMPONENTS = 50

# ---- 特征筛选 ----
MWU_ALPHA = 0.05
CORR_THRESHOLD = 0.90
MRMR_CANDIDATES = 40
MAX_FEATURES = 20

# ---- 结局变量 ----
ENDPOINT_COL  = 'RFS-endpoint'
TIME_COL      = 'RFS-RFS'
CUTOFF_MONTHS = 24
ID_COL        = '住院号'

# ---- 数据划分 ----
TEST_RATIO   = 0.10
VAL_RATIO    = 0.10

# ---- 建模 ----
OPTUNA_N_TRIALS = 200
CV_FOLDS        = 5
CV_REPEATS      = 3

# ---- 配色 ----
NPG_COLORS = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488',
              '#F39B7F', '#8491B4', '#91D1C2', '#DC0000', '#7E6148']

# ---- 提取区域 ----
REGIONS = ['intra', 'peri10mm', 'H1', 'H2', 'H3', 'periH1', 'periH2', 'periH3']
