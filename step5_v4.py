"""step5: 特征筛选 ICC->方差->MWU->Pearson->mRMR->LASSO"""
import os
import sys
import json
import logging
import warnings
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
from config_v4 import (OUT_FEAT_RAW, OUT_FEAT_ICC, OUT_FEAT_ECO, OUT_FEAT_SEL,
                        LOG_DIR, RANDOM_STATE, REGIONS,
                        MWU_ALPHA, CORR_THRESHOLD, MRMR_CANDIDATES, MAX_FEATURES)
from data_split_v4 import load_labels, get_or_create_split

os.makedirs(OUT_FEAT_SEL, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'step5_v4.log'), mode='w'),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)


def load_icc_stable(region: str) -> set:
    path = os.path.join(OUT_FEAT_ICC, f'stable_features_{region}.txt')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return set(line.strip() for line in f if line.strip())


def univariate_filter(X: pd.DataFrame, y: pd.Series,
                      alpha: float = 0.05) -> list:
    kept = []
    for col in X.columns:
        g0 = X.loc[y == 0, col].dropna().values
        g1 = X.loc[y == 1, col].dropna().values
        if len(g0) < 3 or len(g1) < 3:
            continue
        try:
            _, pval = mannwhitneyu(g0, g1, alternative='two-sided')
        except Exception:
            pval = 1.0
        if pval < alpha:
            kept.append(col)
    return kept


def pearson_dedup(X: pd.DataFrame, threshold: float = 0.90) -> list:
    cols = list(X.columns)
    if len(cols) <= 1:
        return cols
    corr = X.corr(method='pearson').abs()
    to_drop = set()
    for i in range(len(cols)):
        if cols[i] in to_drop:
            continue
        for j in range(i + 1, len(cols)):
            if cols[j] in to_drop:
                continue
            if corr.loc[cols[i], cols[j]] > threshold:
                to_drop.add(cols[j])
    return [c for c in cols if c not in to_drop]


def mrmr_filter(X: pd.DataFrame, y: pd.Series, n_select: int) -> list:
    cols = list(X.columns)
    if len(cols) <= n_select:
        return cols
    f_scores, _ = f_classif(X.values, y.values)
    f_scores = np.nan_to_num(f_scores)
    corr_mat = np.abs(np.corrcoef(X.values.T))

    selected = []
    remaining = list(range(len(cols)))
    best_idx = int(np.argmax(f_scores))
    selected.append(best_idx)
    remaining.remove(best_idx)

    for _ in range(min(n_select - 1, len(remaining))):
        scores = []
        for idx in remaining:
            rel = f_scores[idx]
            red = np.mean([corr_mat[idx, s] for s in selected])
            scores.append(rel - red)
        best = remaining[int(np.argmax(scores))]
        selected.append(best)
        remaining.remove(best)
    return [cols[i] for i in selected]


def lasso_filter(X: pd.DataFrame, y: pd.Series) -> tuple:
    """ElasticNet CV, 返回 (selected_features, coefficients)"""
    if X.shape[1] <= 3:
        return list(X.columns), {c: 1.0 for c in X.columns}

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    n_cv = min(10, max(3, X.shape[0] // 5))

    en = ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                      n_alphas=80, cv=n_cv, max_iter=10000,
                      random_state=RANDOM_STATE, n_jobs=-1)
    en.fit(X_s, y)

    coef = pd.Series(en.coef_, index=X.columns)
    selected = list(coef[coef != 0].index)

    if len(selected) > MAX_FEATURES:
        selected = list(coef.abs().nlargest(MAX_FEATURES).index)
    elif len(selected) < 5:
        selected = list(coef.abs().nlargest(min(MAX_FEATURES, len(X.columns))).index)

    coefficients = {c: float(coef[c]) for c in selected}
    log.info(f'    LASSO: alpha={en.alpha_:.5f} l1_ratio={en.l1_ratio_:.2f} '
             f'-> {len(selected)} features')
    return selected, coefficients


def select_features_pipeline(X_train: pd.DataFrame, y_train: pd.Series,
                              region: str, icc_stable: set = None) -> tuple:
    """完整特征筛选管线，返回 (selected_features, lasso_coefficients)"""
    X = X_train.copy()

    # 0. ICC filter
    if icc_stable is not None:
        icc_cols = [c for c in X.columns if c in icc_stable]
        if icc_cols:
            X = X[icc_cols]
            log.info(f'    ICC: {len(icc_cols)}/{X_train.shape[1]}')
        else:
            log.warning(f'    ICC: 无匹配特征，跳过 ICC')

    # 1. 方差过滤
    X = X.loc[:, X.std() > 0.01]
    log.info(f'    方差: {X.shape[1]}')

    if X.shape[1] == 0:
        return [], {}

    # 2. MWU
    kept = univariate_filter(X, y_train, alpha=MWU_ALPHA)
    if len(kept) < 10:
        f_scores, _ = f_classif(X.values, y_train.values)
        top_idx = np.argsort(np.nan_to_num(f_scores))[-50:]
        kept = [X.columns[i] for i in top_idx]
    X = X[kept]
    log.info(f'    MWU(p<{MWU_ALPHA}): {len(kept)}')

    if X.shape[1] == 0:
        return [], {}

    # 3. Pearson 去冗余
    kept = pearson_dedup(X, threshold=CORR_THRESHOLD)
    X = X[kept]
    log.info(f'    Pearson(r>{CORR_THRESHOLD}): {len(kept)}')

    # 4. mRMR
    n_mrmr = min(MRMR_CANDIDATES, len(kept))
    kept = mrmr_filter(X, y_train, n_select=n_mrmr)
    X = X[kept]
    log.info(f'    mRMR: {len(kept)}')

    # 5. LASSO
    selected, coefficients = lasso_filter(X, y_train)
    log.info(f'    最终: {len(selected)} features')
    return selected, coefficients


def main():
    labels_df = load_labels()
    split = get_or_create_split()
    ids_train = split['train']

    log.info(f'标签: {len(labels_df)}例  pos={labels_df["label"].sum()}')
    log.info(f'训练集: {len(ids_train)}例')

    all_results = []
    all_coefficients = {}

    # 各区域独立筛选
    for region in REGIONS:
        feat_path = os.path.join(OUT_FEAT_RAW, f'features_{region}.csv')
        if not os.path.exists(feat_path):
            log.warning(f'  {region}: 特征文件不存在')
            continue

        log.info(f'筛选: {region}')
        df = pd.read_csv(feat_path)
        df['case_id'] = df['case_id'].astype(str)
        merged = df.merge(labels_df[['case_id', 'label']], on='case_id', how='inner')

        train_mask = merged['case_id'].isin(ids_train)
        train_df = merged[train_mask]
        y_train = train_df['label']
        feat_cols = [c for c in df.columns
                     if c not in ('case_id', 'region', 'n_voxels')]
        X_train = train_df[feat_cols].select_dtypes(include=[np.number])

        if len(X_train) < 30:
            log.warning(f'  {region}: 训练样本不足 ({len(X_train)})')
            continue

        icc_stable = load_icc_stable(region)
        selected, coefficients = select_features_pipeline(
            X_train, y_train, region, icc_stable)

        if not selected:
            all_results.append({'region': region, 'status': 'no_features'})
            continue

        out_df = merged[['case_id', 'label'] + selected]
        out_df.to_csv(os.path.join(OUT_FEAT_SEL, f'selected_{region}.csv'),
                      index=False)
        all_coefficients[region] = coefficients
        all_results.append({
            'region': region, 'status': 'ok',
            'n_features': len(selected), 'features': selected,
        })

    # Habitat combined = H1+H2+H3 + ecological
    log.info('构建 Habitat Combined 特征 ...')
    habitat_dfs = []
    for h in ['H1', 'H2', 'H3']:
        p = os.path.join(OUT_FEAT_RAW, f'features_{h}.csv')
        if os.path.exists(p):
            df = pd.read_csv(p)
            df['case_id'] = df['case_id'].astype(str)
            feat_cols = [c for c in df.columns
                         if c not in ('case_id', 'region', 'n_voxels')]
            df_r = df[['case_id'] + feat_cols].rename(
                columns={c: f'{h}_{c}' for c in feat_cols})
            habitat_dfs.append(df_r)

    if len(habitat_dfs) >= 2:
        comb = habitat_dfs[0]
        for df_r in habitat_dfs[1:]:
            comb = comb.merge(df_r, on='case_id', how='inner')

        # 加入 ecological
        eco_path = os.path.join(OUT_FEAT_ECO, 'ecological_features.csv')
        if os.path.exists(eco_path):
            eco_df = pd.read_csv(eco_path)
            eco_df['case_id'] = eco_df['case_id'].astype(str)
            comb = comb.merge(eco_df, on='case_id', how='inner')

        merged = comb.merge(labels_df[['case_id', 'label']],
                            on='case_id', how='inner')
        train_mask = merged['case_id'].isin(ids_train)
        train_df = merged[train_mask]
        y_train = train_df['label']
        feat_cols = [c for c in comb.columns if c != 'case_id']
        X_train = train_df[feat_cols].select_dtypes(include=[np.number])

        # Ecological 特征跳过 ICC (天然稳定)
        eco_cols = [c for c in X_train.columns
                    if c.startswith('habitat_') or c.startswith('contrast_')]
        icc_stable_h = set()
        for h in ['H1', 'H2', 'H3']:
            stable = load_icc_stable(h)
            if stable:
                icc_stable_h.update(f'{h}_{s}' for s in stable)
        icc_stable_h.update(eco_cols)

        selected, coefficients = select_features_pipeline(
            X_train, y_train, 'habitat_combined', icc_stable_h)

        if selected:
            out_df = merged[['case_id', 'label'] + selected]
            out_df.to_csv(os.path.join(OUT_FEAT_SEL,
                                       'selected_habitat_combined.csv'),
                          index=False)
            all_coefficients['habitat_combined'] = coefficients
            all_results.append({
                'region': 'habitat_combined', 'status': 'ok',
                'n_features': len(selected), 'features': selected,
            })

    # 保存汇总
    with open(os.path.join(OUT_FEAT_SEL, 'selection_summary.json'), 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    with open(os.path.join(OUT_FEAT_SEL, 'lasso_coefficients.json'), 'w') as f:
        json.dump(all_coefficients, f, indent=2)

    log.info('=== Step 5 v4 特征筛选完成 ===')
    for r in all_results:
        if r.get('status') == 'ok':
            log.info(f"  {r['region']}: {r['n_features']} features")


if __name__ == '__main__':
    main()
