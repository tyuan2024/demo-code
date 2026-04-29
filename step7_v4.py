"""step7: 生存分析 Cox-LASSO + RSF + KM + TD-AUC"""
import os
import sys
import json
import logging
import warnings
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
from config_v4 import (OUT_FEAT_SEL, OUT_MODELS, LOG_DIR,
                        ENDPOINT_COL, TIME_COL, RANDOM_STATE)
from data_split_v4 import load_survival_data, get_or_create_split

os.makedirs(OUT_MODELS, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'step7_v4.log'), mode='w'),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)


def compute_rad_score_from_model(model_name: str, surv_df: pd.DataFrame) -> pd.Series:
    """从step6模型算rad-score"""
    model_path = os.path.join(OUT_MODELS, f'{model_name}_model_v4.pkl')
    if not os.path.exists(model_path):
        return None

    # 找对应的特征文件
    region_map = {
        'Habitat_Radiomics': 'habitat_combined',
        'Radiomics_Intra': 'intra',
        'Radiomics_Peri': 'peri10mm',
    }
    region = region_map.get(model_name, model_name.lower())
    feat_path = os.path.join(OUT_FEAT_SEL, f'selected_{region}.csv')
    if not os.path.exists(feat_path):
        return None

    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    clf = model_data['clf']
    scaler = model_data['scaler']
    imputer = model_data.get('imputer')

    feat_df = pd.read_csv(feat_path)
    feat_df['case_id'] = feat_df['case_id'].astype(str)
    feat_cols = [c for c in feat_df.columns if c not in ('case_id', 'label')]

    X = feat_df[feat_cols].values
    if imputer is not None:
        X = imputer.transform(X)
    X = scaler.transform(X)
    probs = clf.predict_proba(X)[:, 1]

    return pd.DataFrame({'case_id': feat_df['case_id'], 'rad_score': probs})


def cox_lasso_cv(train_df: pd.DataFrame, features: list,
                 time_col: str, event_col: str) -> CoxPHFitter:
    """Cox-LASSO, 5折CV选penalizer"""
    best_c = 0.5
    best_pen = 0.1

    for pen in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]:
        c_indices = []
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        y_binary = (train_df[event_col] > 0).astype(int)

        for tr_idx, va_idx in kf.split(train_df, y_binary):
            tr_fold = train_df.iloc[tr_idx]
            va_fold = train_df.iloc[va_idx]

            try:
                cph = CoxPHFitter(penalizer=pen, l1_ratio=0.5)
                cph.fit(tr_fold[features + [time_col, event_col]],
                        duration_col=time_col, event_col=event_col)
                risk = cph.predict_partial_hazard(va_fold[features]).values.flatten()
                c_idx = concordance_index_censored(
                    va_fold[event_col].astype(bool),
                    va_fold[time_col], risk)[0]
                c_indices.append(c_idx)
            except Exception:
                c_indices.append(0.5)

        mean_c = np.mean(c_indices)
        if mean_c > best_c:
            best_c = mean_c
            best_pen = pen

    log.info(f'  Cox best penalizer={best_pen}, CV C-index={best_c:.4f}')

    cph_final = CoxPHFitter(penalizer=best_pen, l1_ratio=0.5)
    cph_final.fit(train_df[features + [time_col, event_col]],
                  duration_col=time_col, event_col=event_col)
    return cph_final


def find_optimal_cutpoint(risk_scores, times, events):
    """搜索最优cutpoint (log-rank最小p)"""
    best_p = 1.0
    best_cut = np.median(risk_scores)

    for pct in range(25, 76, 5):
        cut = np.percentile(risk_scores, pct)
        high = risk_scores >= cut
        low = risk_scores < cut
        if high.sum() < 10 or low.sum() < 10:
            continue
        try:
            lr = logrank_test(times[high], times[low],
                              events[high], events[low])
            if lr.p_value < best_p:
                best_p = lr.p_value
                best_cut = cut
        except Exception:
            pass

    return float(best_cut), float(best_p)


def time_dependent_auc(risk_scores, times, events, eval_times=[12, 24, 36]):
    """TD-AUC at 12/24/36月"""
    try:
        from sksurv.metrics import cumulative_dynamic_auc
        y_surv = np.array([(bool(e), t) for e, t in zip(events, times)],
                          dtype=[('event', bool), ('time', float)])
        auc_vals, mean_auc = cumulative_dynamic_auc(
            y_surv, y_surv, risk_scores, eval_times)
        return {f'AUC_{t}m': round(float(a), 4)
                for t, a in zip(eval_times, auc_vals)}
    except Exception as e:
        log.warning(f'  Time-dependent AUC failed: {e}')
        return {}


def main():
    surv_df = load_survival_data()
    split = get_or_create_split()
    ids_train, ids_test = split['train'], split['test']

    log.info(f'生存数据: {len(surv_df)}例, '
             f'事件率={surv_df[ENDPOINT_COL].mean():.2%}')

    # 尝试 Habitat rad-score，否则用最佳 radiomics
    rad_df = None
    for model_name in ['Habitat_Radiomics', 'Radiomics_Intra', 'Radiomics_Peri']:
        rad_df = compute_rad_score_from_model(model_name, surv_df)
        if rad_df is not None:
            log.info(f'Rad-score from: {model_name}')
            break

    if rad_df is not None:
        surv_df = surv_df.merge(rad_df, on='case_id', how='left')
        surv_df['rad_score'] = surv_df['rad_score'].fillna(
            surv_df['rad_score'].median())
    else:
        log.warning('No rad-score available')
        surv_df['rad_score'] = 0.5

    # 临床特征
    cand = ['性别', '年龄', '肿瘤最大径', '肿瘤数目', '卫星灶', '肿瘤分级',
            '肝硬化', 'MVI', '乙肝史', 'AFP甲胎', 'BCLC分期',
            'TB总胆红素', 'ALB白蛋白', 'PLT血小板']
    avail = [c for c in cand if c in surv_df.columns]
    for col in avail:
        surv_df[col] = pd.to_numeric(surv_df[col], errors='coerce')
    avail = [c for c in avail if surv_df[c].notna().sum() > 10]
    for col in avail:
        med = surv_df[col].median()
        if pd.isna(med):
            med = 0
        surv_df[col] = surv_df[col].fillna(med).replace([np.inf, -np.inf], 0)

    features = avail + ['rad_score']
    train_df = surv_df[surv_df['case_id'].isin(ids_train)].copy()
    test_df = surv_df[surv_df['case_id'].isin(ids_test)].copy()

    # 标准化
    scaler = StandardScaler()
    train_scaled = pd.DataFrame(
        scaler.fit_transform(train_df[features]),
        columns=features, index=train_df.index)
    train_scaled[TIME_COL] = train_df[TIME_COL].values
    train_scaled[ENDPOINT_COL] = train_df[ENDPOINT_COL].values

    test_scaled = pd.DataFrame(
        scaler.transform(test_df[features]),
        columns=features, index=test_df.index)
    test_scaled[TIME_COL] = test_df[TIME_COL].values
    test_scaled[ENDPOINT_COL] = test_df[ENDPOINT_COL].values

    # 去零方差
    zero_var = [c for c in features if train_scaled[c].std() < 1e-8]
    if zero_var:
        features = [c for c in features if c not in zero_var]
        train_scaled = train_scaled.drop(columns=zero_var)
        test_scaled = test_scaled.drop(columns=zero_var)

    # 确保无 NaN
    for col in features:
        for df_ in [train_scaled, test_scaled]:
            df_[col] = df_[col].fillna(0).replace([np.inf, -np.inf], 0)

    results = {}

    # ── Cox PH (LASSO CV) ──
    log.info('Cox PH (LASSO CV) ...')
    try:
        cph = cox_lasso_cv(train_scaled, features, TIME_COL, ENDPOINT_COL)
        train_risk = cph.predict_partial_hazard(train_scaled[features]).values.flatten()
        test_risk = cph.predict_partial_hazard(test_scaled[features]).values.flatten()

        c_train = concordance_index_censored(
            train_df[ENDPOINT_COL].astype(bool), train_df[TIME_COL], train_risk)[0]
        c_test = concordance_index_censored(
            test_df[ENDPOINT_COL].astype(bool), test_df[TIME_COL], test_risk)[0]

        results['Cox_LASSO'] = {
            'C_index_train': round(c_train, 4),
            'C_index_test': round(c_test, 4),
        }
        log.info(f'  Cox C-index: train={c_train:.4f}  test={c_test:.4f}')

        with open(os.path.join(OUT_MODELS, 'cox_model_v4.pkl'), 'wb') as f:
            pickle.dump({'cph': cph, 'scaler': scaler, 'features': features}, f)
    except Exception as e:
        log.warning(f'  Cox failed: {e}')
        results['Cox_LASSO'] = {'C_index_train': 0.5, 'C_index_test': 0.5}
        test_risk = surv_df.loc[surv_df['case_id'].isin(ids_test), 'rad_score'].values

    # ── RSF ──
    log.info('Random Survival Forest ...')
    X_train_rsf = train_df[features].values
    X_test_rsf = test_df[features].values

    y_train_surv = np.array(
        [(bool(e), t) for e, t in
         zip(train_df[ENDPOINT_COL], train_df[TIME_COL])],
        dtype=[('event', bool), ('time', float)])
    y_test_surv = np.array(
        [(bool(e), t) for e, t in
         zip(test_df[ENDPOINT_COL], test_df[TIME_COL])],
        dtype=[('event', bool), ('time', float)])

    rsf = RandomSurvivalForest(
        n_estimators=200, max_depth=4, min_samples_split=15,
        min_samples_leaf=10, max_features='sqrt',
        random_state=RANDOM_STATE, n_jobs=-1)
    rsf.fit(X_train_rsf, y_train_surv)

    rsf_c_train = rsf.score(X_train_rsf, y_train_surv)
    rsf_c_test = rsf.score(X_test_rsf, y_test_surv)
    results['RSF'] = {
        'C_index_train': round(rsf_c_train, 4),
        'C_index_test': round(rsf_c_test, 4),
    }
    log.info(f'  RSF C-index: train={rsf_c_train:.4f}  test={rsf_c_test:.4f}')

    with open(os.path.join(OUT_MODELS, 'rsf_model_v4.pkl'), 'wb') as f:
        pickle.dump({'rsf': rsf, 'features': features}, f)

    # ── Time-dependent AUC ──
    log.info('Time-dependent AUC ...')
    td_auc_train = time_dependent_auc(
        test_risk if len(test_risk) == len(test_df) else
        surv_df.loc[surv_df['case_id'].isin(ids_test), 'rad_score'].values,
        test_df[TIME_COL].values, test_df[ENDPOINT_COL].values)
    results['time_dependent_auc'] = td_auc_train
    log.info(f'  TD-AUC: {td_auc_train}')

    # ── KM with optimal cutpoint ──
    log.info('Kaplan-Meier ...')
    all_risk = surv_df['rad_score'].values
    all_times = surv_df[TIME_COL].values
    all_events = surv_df[ENDPOINT_COL].values

    best_cut, best_p = find_optimal_cutpoint(all_risk, all_times, all_events)
    log.info(f'  Optimal cutpoint={best_cut:.4f}, p={best_p:.2e}')

    surv_df['risk_group'] = np.where(all_risk >= best_cut, 'High', 'Low')

    km_data = {'cutpoint': best_cut, 'logrank_p': best_p, 'groups': {}}
    for grp in ['High', 'Low']:
        sub = surv_df[surv_df['risk_group'] == grp]
        kmf = KaplanMeierFitter()
        kmf.fit(sub[TIME_COL], sub[ENDPOINT_COL], label=grp)
        km_data['groups'][grp] = {
            'n': len(sub), 'events': int(sub[ENDPOINT_COL].sum()),
            'median_survival': (float(kmf.median_survival_time_)
                                if kmf.median_survival_time_ != np.inf else None),
        }
    results['KM'] = km_data

    # Save
    with open(os.path.join(OUT_MODELS, 'survival_results_v4.json'), 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    with open(os.path.join(OUT_MODELS, 'time_dependent_auc_v4.json'), 'w') as f:
        json.dump(td_auc_train, f, indent=2)

    log.info('=== Step 7 v4 生存分析完成 ===')


if __name__ == '__main__':
    main()
