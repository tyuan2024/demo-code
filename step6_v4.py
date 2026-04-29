"""step6: 分类建模 10个模型, ElasticNet + SMOTE + Optuna"""
import os
import sys
import json
import logging
import warnings
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (roc_auc_score, accuracy_score,
                             confusion_matrix, f1_score)
from imblearn.over_sampling import SMOTE
import optuna

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, os.path.dirname(__file__))
from config_v4 import (OUT_FEAT_SEL, OUT_FEAT_DL, OUT_MODELS, LOG_DIR,
                        RANDOM_STATE, OPTUNA_N_TRIALS, CV_FOLDS, CV_REPEATS,
                        MAX_FEATURES)
from data_split_v4 import (load_labels, load_clinical_features,
                            get_or_create_split)

os.makedirs(OUT_MODELS, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'step6_v4.log'), mode='w'),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    auc_val = roc_auc_score(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sen = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return dict(AUC=round(auc_val, 4), ACC=round(acc, 4),
                SEN=round(float(sen), 4), SPEC=round(float(spec), 4),
                PPV=round(float(ppv), 4), NPV=round(float(npv), 4),
                F1=round(f1, 4))


def build_model(X_train, y_train, X_val, y_val, X_test, y_test,
                model_name: str) -> dict:
    """Optuna调参 ElasticNet, SMOTE在CV内"""
    cv = RepeatedStratifiedKFold(n_splits=CV_FOLDS, n_repeats=CV_REPEATS,
                                  random_state=RANDOM_STATE)

    min_pos = int(y_train.sum())
    smote_k = min(5, max(1, min_pos - 1))

    def objective(trial):
        C = trial.suggest_float('C', 1e-3, 10.0, log=True)
        l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)

        aucs = []
        for tr_idx, va_idx in cv.split(X_train, y_train):
            X_t, X_v = X_train[tr_idx], X_train[va_idx]
            y_t, y_v = y_train[tr_idx], y_train[va_idx]

            imputer = SimpleImputer(strategy='median')
            X_t = imputer.fit_transform(X_t)
            X_v = imputer.transform(X_v)

            scaler = StandardScaler()
            X_t = scaler.fit_transform(X_t)
            X_v = scaler.transform(X_v)

            try:
                smote = SMOTE(random_state=RANDOM_STATE,
                              k_neighbors=smote_k)
                X_t, y_t = smote.fit_resample(X_t, y_t)
            except Exception:
                pass

            clf = LogisticRegression(
                C=C, penalty='elasticnet', l1_ratio=l1_ratio,
                solver='saga', max_iter=5000,
                class_weight='balanced', random_state=RANDOM_STATE)
            try:
                clf.fit(X_t, y_t)
                prob = clf.predict_proba(X_v)[:, 1]
                aucs.append(roc_auc_score(y_v, prob))
            except Exception:
                aucs.append(0.5)

        return np.mean(aucs)

    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=OPTUNA_N_TRIALS, show_progress_bar=False)

    best = study.best_params
    log.info(f'  [{model_name}] CV AUC={study.best_value:.4f} '
             f'C={best["C"]:.4f} l1={best["l1_ratio"]:.2f}')

    # Final model
    imputer = SimpleImputer(strategy='median')
    X_tr_imp = imputer.fit_transform(X_train)
    X_va_imp = imputer.transform(X_val)
    X_te_imp = imputer.transform(X_test)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr_imp)
    X_va_s = scaler.transform(X_va_imp)
    X_te_s = scaler.transform(X_te_imp)

    try:
        smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=smote_k)
        X_tr_sm, y_tr_sm = smote.fit_resample(X_tr_s, y_train)
    except Exception:
        X_tr_sm, y_tr_sm = X_tr_s, y_train

    clf = LogisticRegression(
        C=best['C'], penalty='elasticnet', l1_ratio=best['l1_ratio'],
        solver='saga', max_iter=5000,
        class_weight='balanced', random_state=RANDOM_STATE)
    clf.fit(X_tr_sm, y_tr_sm)

    prob_train = clf.predict_proba(X_tr_s)[:, 1]
    prob_val = clf.predict_proba(X_va_s)[:, 1]
    prob_test = clf.predict_proba(X_te_s)[:, 1]

    metrics = {
        'train': compute_metrics(y_train, prob_train),
        'val': compute_metrics(y_val, prob_val),
        'test': compute_metrics(y_test, prob_test),
    }

    model_path = os.path.join(OUT_MODELS, f'{model_name}_model_v4.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump({'clf': clf, 'scaler': scaler, 'imputer': imputer,
                     'params': best}, f)

    return {
        'model_name': model_name,
        'cv_auc': round(study.best_value, 4),
        'metrics': metrics,
        'probs': {
            'train': prob_train.tolist(),
            'val': prob_val.tolist(),
            'test': prob_test.tolist(),
        },
    }


def load_region_features(region: str, ids_train, ids_val, ids_test):
    """加载选中特征并按 split 分组"""
    path = os.path.join(OUT_FEAT_SEL, f'selected_{region}.csv')
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df['case_id'] = df['case_id'].astype(str)
    feat_cols = [c for c in df.columns if c not in ('case_id', 'label')]

    df_tr = df[df['case_id'].isin(ids_train)]
    df_va = df[df['case_id'].isin(ids_val)]
    df_te = df[df['case_id'].isin(ids_test)]

    if len(df_tr) < 30 or len(df_te) < 10:
        return None

    return {
        'X_train': df_tr[feat_cols].values,
        'X_val': df_va[feat_cols].values,
        'X_test': df_te[feat_cols].values,
        'y_train': df_tr['label'].values,
        'y_val': df_va['label'].values,
        'y_test': df_te['label'].values,
        'feat_cols': feat_cols,
        'case_ids': {
            'train': df_tr['case_id'].tolist(),
            'val': df_va['case_id'].tolist(),
            'test': df_te['case_id'].tolist(),
        },
    }


def compute_rad_score(clf, scaler, X, imputer=None):
    """计算 Rad-score (模型预测概率)"""
    if imputer is not None:
        X = imputer.transform(X)
    X_s = scaler.transform(X)
    return clf.predict_proba(X_s)[:, 1]


def main():
    labels_df = load_labels()
    clinical_df, clinical_cols = load_clinical_features(labels_df)
    split = get_or_create_split()
    ids_train, ids_val, ids_test = split['train'], split['val'], split['test']

    log.info(f'Split: train={len(ids_train)} val={len(ids_val)} '
             f'test={len(ids_test)}')

    all_results = {}
    all_probs = {}

    # 1. Clinical
    log.info('Building: Clinical')
    clin_tr = clinical_df[clinical_df['case_id'].isin(ids_train)]
    clin_va = clinical_df[clinical_df['case_id'].isin(ids_val)]
    clin_te = clinical_df[clinical_df['case_id'].isin(ids_test)]

    result = build_model(
        clin_tr[clinical_cols].values, clin_tr['label'].values,
        clin_va[clinical_cols].values, clin_va['label'].values,
        clin_te[clinical_cols].values, clin_te['label'].values,
        'Clinical')
    all_results['Clinical'] = result
    all_probs['Clinical'] = result['probs']
    log.info(f"  Clinical: test AUC={result['metrics']['test']['AUC']}")

    # 2-5. Region models
    region_map = {
        'Radiomics_Intra': 'intra',
        'Radiomics_Peri': 'peri10mm',
        'Habitat_Radiomics': 'habitat_combined',
    }

    # Build Peri_Habitat from periH1+periH2+periH3 combined
    # First build individual region models
    for name, region in region_map.items():
        data = load_region_features(region, ids_train, ids_val, ids_test)
        if data is None:
            log.warning(f'  {name}: 数据不足，跳过')
            continue
        log.info(f'Building: {name} ({data["X_train"].shape[1]} features)')
        result = build_model(
            data['X_train'], data['y_train'],
            data['X_val'], data['y_val'],
            data['X_test'], data['y_test'], name)
        all_results[name] = result
        all_probs[name] = result['probs']
        log.info(f"  {name}: test AUC={result['metrics']['test']['AUC']}")

    # 5. Peri_Habitat
    peri_h_regions = ['periH1', 'periH2', 'periH3']
    peri_h_datas = [load_region_features(r, ids_train, ids_val, ids_test)
                    for r in peri_h_regions]
    if all(d is not None for d in peri_h_datas):
        X_ph_tr = np.hstack([d['X_train'] for d in peri_h_datas])
        X_ph_va = np.hstack([d['X_val'] for d in peri_h_datas])
        X_ph_te = np.hstack([d['X_test'] for d in peri_h_datas])
        log.info(f'Building: Peri_Habitat ({X_ph_tr.shape[1]} features)')
        result = build_model(
            X_ph_tr, peri_h_datas[0]['y_train'],
            X_ph_va, peri_h_datas[0]['y_val'],
            X_ph_te, peri_h_datas[0]['y_test'], 'Peri_Habitat')
        all_results['Peri_Habitat'] = result
        all_probs['Peri_Habitat'] = result['probs']
        log.info(f"  Peri_Habitat: test AUC={result['metrics']['test']['AUC']}")
    else:
        log.warning('  Peri_Habitat: 数据不足，跳过')

    # 6. DL Features
    dl_path = os.path.join(OUT_FEAT_DL, 'resnet50_pca50.csv')
    if os.path.exists(dl_path):
        log.info('Building: DL_Features')
        dl_df = pd.read_csv(dl_path)
        dl_df['case_id'] = dl_df['case_id'].astype(str)
        dl_df = dl_df.merge(labels_df[['case_id', 'label']],
                            on='case_id', how='inner')
        dl_feats = [c for c in dl_df.columns
                    if c not in ('case_id', 'label')]

        dl_tr = dl_df[dl_df['case_id'].isin(ids_train)]
        dl_va = dl_df[dl_df['case_id'].isin(ids_val)]
        dl_te = dl_df[dl_df['case_id'].isin(ids_test)]

        if len(dl_tr) >= 30:
            result = build_model(
                dl_tr[dl_feats].values, dl_tr['label'].values,
                dl_va[dl_feats].values, dl_va['label'].values,
                dl_te[dl_feats].values, dl_te['label'].values,
                'DL_Features')
            all_results['DL_Features'] = result
            all_probs['DL_Features'] = result['probs']
            log.info(f"  DL_Features: test AUC="
                     f"{result['metrics']['test']['AUC']}")

    # 7-10. Nomogram
    # Rad-score from best radiomics model (excluding Habitat)
    rad_models_non_habitat = {k: v for k, v in all_results.items()
                              if k in ('Radiomics_Intra', 'Radiomics_Peri')}
    rad_models_all = {k: v for k, v in all_results.items()
                      if k not in ('Clinical', 'DL_Features', 'Peri_Habitat')}

    # Helper: build nomogram from rad-score + clinical
    def build_nomogram(combo_name, rad_name):
        if rad_name not in all_results:
            log.warning(f'  {combo_name}: {rad_name} not available, skip')
            return
        rad_data = pickle.load(
            open(os.path.join(OUT_MODELS, f'{rad_name}_model_v4.pkl'), 'rb'))
        rad_region = region_map.get(rad_name, rad_name)
        feat_data = load_region_features(rad_region, ids_train, ids_val, ids_test)
        if feat_data is None:
            return

        rad_tr = compute_rad_score(rad_data['clf'], rad_data['scaler'],
                                   feat_data['X_train'],
                                   rad_data.get('imputer')).reshape(-1, 1)
        rad_va = compute_rad_score(rad_data['clf'], rad_data['scaler'],
                                   feat_data['X_val'],
                                   rad_data.get('imputer')).reshape(-1, 1)
        rad_te = compute_rad_score(rad_data['clf'], rad_data['scaler'],
                                   feat_data['X_test'],
                                   rad_data.get('imputer')).reshape(-1, 1)

        clin_tr_a = clin_tr[clin_tr['case_id'].isin(
            feat_data['case_ids']['train'])].sort_values('case_id')
        clin_va_a = clin_va[clin_va['case_id'].isin(
            feat_data['case_ids']['val'])].sort_values('case_id')
        clin_te_a = clin_te[clin_te['case_id'].isin(
            feat_data['case_ids']['test'])].sort_values('case_id')

        X_combo_tr = np.hstack([rad_tr, clin_tr_a[clinical_cols].values])
        X_combo_va = np.hstack([rad_va, clin_va_a[clinical_cols].values])
        X_combo_te = np.hstack([rad_te, clin_te_a[clinical_cols].values])

        log.info(f'Building: {combo_name} ({X_combo_tr.shape[1]} features)')
        result = build_model(
            X_combo_tr, feat_data['y_train'],
            X_combo_va, feat_data['y_val'],
            X_combo_te, feat_data['y_test'], combo_name)
        all_results[combo_name] = result
        all_probs[combo_name] = result['probs']
        log.info(f"  {combo_name}: test AUC={result['metrics']['test']['AUC']}")

    # 7. Rad_Clinical: best non-habitat radiomics + clinical
    if rad_models_non_habitat:
        best_rad_name = max(rad_models_non_habitat,
                            key=lambda k: rad_models_non_habitat[k]['cv_auc'])
        log.info(f'Best non-habitat radiomics: {best_rad_name}')
        build_nomogram('Rad_Clinical', best_rad_name)
    elif rad_models_all:
        # Fallback: use best overall
        best_rad_name = max(rad_models_all,
                            key=lambda k: rad_models_all[k]['cv_auc'])
        log.info(f'Best radiomics (fallback): {best_rad_name}')
        build_nomogram('Rad_Clinical', best_rad_name)

    # 8. Habitat_Clinical: habitat rad-score + clinical
    build_nomogram('Habitat_Clinical', 'Habitat_Radiomics')

    # 9. DL_Rad_Clinical: DL-score + best rad-score + clinical
    if 'DL_Features' in all_results and rad_models_all:
        best_rad_for_dl = max(rad_models_all,
                              key=lambda k: rad_models_all[k]['cv_auc'])
        dl_model_data = pickle.load(
            open(os.path.join(OUT_MODELS, 'DL_Features_model_v4.pkl'), 'rb'))
        rad_model_data = pickle.load(
            open(os.path.join(OUT_MODELS, f'{best_rad_for_dl}_model_v4.pkl'), 'rb'))

        # DL scores
        dl_df_full = pd.read_csv(dl_path)
        dl_df_full['case_id'] = dl_df_full['case_id'].astype(str)
        dl_feats_all = [c for c in dl_df_full.columns if c not in ('case_id', 'label')]
        dl_tr_a = dl_df_full[dl_df_full['case_id'].isin(ids_train)].sort_values('case_id')
        dl_va_a = dl_df_full[dl_df_full['case_id'].isin(ids_val)].sort_values('case_id')
        dl_te_a = dl_df_full[dl_df_full['case_id'].isin(ids_test)].sort_values('case_id')

        dl_score_tr = compute_rad_score(dl_model_data['clf'], dl_model_data['scaler'],
                                        dl_tr_a[dl_feats_all].values,
                                        dl_model_data.get('imputer')).reshape(-1, 1)
        dl_score_va = compute_rad_score(dl_model_data['clf'], dl_model_data['scaler'],
                                        dl_va_a[dl_feats_all].values,
                                        dl_model_data.get('imputer')).reshape(-1, 1)
        dl_score_te = compute_rad_score(dl_model_data['clf'], dl_model_data['scaler'],
                                        dl_te_a[dl_feats_all].values,
                                        dl_model_data.get('imputer')).reshape(-1, 1)

        # Rad scores
        rad_region = region_map.get(best_rad_for_dl, best_rad_for_dl)
        feat_data = load_region_features(rad_region, ids_train, ids_val, ids_test)
        if feat_data is not None:
            rad_tr = compute_rad_score(rad_model_data['clf'], rad_model_data['scaler'],
                                       feat_data['X_train'],
                                       rad_model_data.get('imputer')).reshape(-1, 1)
            rad_va = compute_rad_score(rad_model_data['clf'], rad_model_data['scaler'],
                                       feat_data['X_val'],
                                       rad_model_data.get('imputer')).reshape(-1, 1)
            rad_te = compute_rad_score(rad_model_data['clf'], rad_model_data['scaler'],
                                       feat_data['X_test'],
                                       rad_model_data.get('imputer')).reshape(-1, 1)

            # Align all by case_id (use intersection)
            common_tr = sorted(set(dl_tr_a['case_id']) & set(feat_data['case_ids']['train'])
                               & set(clin_tr['case_id']))
            common_va = sorted(set(dl_va_a['case_id']) & set(feat_data['case_ids']['val'])
                               & set(clin_va['case_id']))
            common_te = sorted(set(dl_te_a['case_id']) & set(feat_data['case_ids']['test'])
                               & set(clin_te['case_id']))

            # Reindex
            dl_tr_idx = dl_tr_a[dl_tr_a['case_id'].isin(common_tr)].sort_values('case_id').index
            dl_va_idx = dl_va_a[dl_va_a['case_id'].isin(common_va)].sort_values('case_id').index
            dl_te_idx = dl_te_a[dl_te_a['case_id'].isin(common_te)].sort_values('case_id').index

            dl_s_tr = compute_rad_score(dl_model_data['clf'], dl_model_data['scaler'],
                                        dl_tr_a.loc[dl_tr_idx, dl_feats_all].values,
                                        dl_model_data.get('imputer')).reshape(-1, 1)
            dl_s_va = compute_rad_score(dl_model_data['clf'], dl_model_data['scaler'],
                                        dl_va_a.loc[dl_va_idx, dl_feats_all].values,
                                        dl_model_data.get('imputer')).reshape(-1, 1)
            dl_s_te = compute_rad_score(dl_model_data['clf'], dl_model_data['scaler'],
                                        dl_te_a.loc[dl_te_idx, dl_feats_all].values,
                                        dl_model_data.get('imputer')).reshape(-1, 1)

            # Rad scores for common cases
            feat_tr_ids = feat_data['case_ids']['train']
            feat_va_ids = feat_data['case_ids']['val']
            feat_te_ids = feat_data['case_ids']['test']
            r_tr_mask = [i for i, cid in enumerate(feat_tr_ids) if cid in common_tr]
            r_va_mask = [i for i, cid in enumerate(feat_va_ids) if cid in common_va]
            r_te_mask = [i for i, cid in enumerate(feat_te_ids) if cid in common_te]

            rad_s_tr = rad_tr[r_tr_mask]
            rad_s_va = rad_va[r_va_mask]
            rad_s_te = rad_te[r_te_mask]

            clin_c_tr = clin_tr[clin_tr['case_id'].isin(common_tr)].sort_values('case_id')
            clin_c_va = clin_va[clin_va['case_id'].isin(common_va)].sort_values('case_id')
            clin_c_te = clin_te[clin_te['case_id'].isin(common_te)].sort_values('case_id')

            X_drc_tr = np.hstack([dl_s_tr, rad_s_tr, clin_c_tr[clinical_cols].values])
            X_drc_va = np.hstack([dl_s_va, rad_s_va, clin_c_va[clinical_cols].values])
            X_drc_te = np.hstack([dl_s_te, rad_s_te, clin_c_te[clinical_cols].values])

            y_drc_tr = clin_c_tr['label'].values
            y_drc_va = clin_c_va['label'].values
            y_drc_te = clin_c_te['label'].values

            log.info(f'Building: DL_Rad_Clinical ({X_drc_tr.shape[1]} features)')
            result = build_model(X_drc_tr, y_drc_tr, X_drc_va, y_drc_va,
                                 X_drc_te, y_drc_te, 'DL_Rad_Clinical')
            all_results['DL_Rad_Clinical'] = result
            all_probs['DL_Rad_Clinical'] = result['probs']
            log.info(f"  DL_Rad_Clinical: test AUC={result['metrics']['test']['AUC']}")

    # 10. Habitat_DL_Clinical: habitat-score + DL-score + clinical
    if 'DL_Features' in all_results and 'Habitat_Radiomics' in all_results:
        hab_model_data = pickle.load(
            open(os.path.join(OUT_MODELS, 'Habitat_Radiomics_model_v4.pkl'), 'rb'))
        hab_feat = load_region_features('habitat_combined', ids_train, ids_val, ids_test)

        if hab_feat is not None:
            hab_tr = compute_rad_score(hab_model_data['clf'], hab_model_data['scaler'],
                                       hab_feat['X_train'],
                                       hab_model_data.get('imputer')).reshape(-1, 1)
            hab_va = compute_rad_score(hab_model_data['clf'], hab_model_data['scaler'],
                                       hab_feat['X_val'],
                                       hab_model_data.get('imputer')).reshape(-1, 1)
            hab_te = compute_rad_score(hab_model_data['clf'], hab_model_data['scaler'],
                                       hab_feat['X_test'],
                                       hab_model_data.get('imputer')).reshape(-1, 1)

            common_tr = sorted(set(dl_tr_a['case_id']) & set(hab_feat['case_ids']['train'])
                               & set(clin_tr['case_id']))
            common_va = sorted(set(dl_va_a['case_id']) & set(hab_feat['case_ids']['val'])
                               & set(clin_va['case_id']))
            common_te = sorted(set(dl_te_a['case_id']) & set(hab_feat['case_ids']['test'])
                               & set(clin_te['case_id']))

            dl_hdl_tr_idx = dl_tr_a[dl_tr_a['case_id'].isin(common_tr)].sort_values('case_id').index
            dl_hdl_va_idx = dl_va_a[dl_va_a['case_id'].isin(common_va)].sort_values('case_id').index
            dl_hdl_te_idx = dl_te_a[dl_te_a['case_id'].isin(common_te)].sort_values('case_id').index

            dl_h_tr = compute_rad_score(dl_model_data['clf'], dl_model_data['scaler'],
                                        dl_tr_a.loc[dl_hdl_tr_idx, dl_feats_all].values,
                                        dl_model_data.get('imputer')).reshape(-1, 1)
            dl_h_va = compute_rad_score(dl_model_data['clf'], dl_model_data['scaler'],
                                        dl_va_a.loc[dl_hdl_va_idx, dl_feats_all].values,
                                        dl_model_data.get('imputer')).reshape(-1, 1)
            dl_h_te = compute_rad_score(dl_model_data['clf'], dl_model_data['scaler'],
                                        dl_te_a.loc[dl_hdl_te_idx, dl_feats_all].values,
                                        dl_model_data.get('imputer')).reshape(-1, 1)

            h_tr_ids = hab_feat['case_ids']['train']
            h_va_ids = hab_feat['case_ids']['val']
            h_te_ids = hab_feat['case_ids']['test']
            h_tr_mask = [i for i, cid in enumerate(h_tr_ids) if cid in common_tr]
            h_va_mask = [i for i, cid in enumerate(h_va_ids) if cid in common_va]
            h_te_mask = [i for i, cid in enumerate(h_te_ids) if cid in common_te]

            hab_s_tr = hab_tr[h_tr_mask]
            hab_s_va = hab_va[h_va_mask]
            hab_s_te = hab_te[h_te_mask]

            clin_h_tr = clin_tr[clin_tr['case_id'].isin(common_tr)].sort_values('case_id')
            clin_h_va = clin_va[clin_va['case_id'].isin(common_va)].sort_values('case_id')
            clin_h_te = clin_te[clin_te['case_id'].isin(common_te)].sort_values('case_id')

            X_hdl_tr = np.hstack([hab_s_tr, dl_h_tr, clin_h_tr[clinical_cols].values])
            X_hdl_va = np.hstack([hab_s_va, dl_h_va, clin_h_va[clinical_cols].values])
            X_hdl_te = np.hstack([hab_s_te, dl_h_te, clin_h_te[clinical_cols].values])

            y_hdl_tr = clin_h_tr['label'].values
            y_hdl_va = clin_h_va['label'].values
            y_hdl_te = clin_h_te['label'].values

            log.info(f'Building: Habitat_DL_Clinical ({X_hdl_tr.shape[1]} features)')
            result = build_model(X_hdl_tr, y_hdl_tr, X_hdl_va, y_hdl_va,
                                 X_hdl_te, y_hdl_te, 'Habitat_DL_Clinical')
            all_results['Habitat_DL_Clinical'] = result
            all_probs['Habitat_DL_Clinical'] = result['probs']
            log.info(f"  Habitat_DL_Clinical: test AUC={result['metrics']['test']['AUC']}")

    # save results
    summary = {}
    for name, res in all_results.items():
        summary[name] = {
            'cv_auc': res['cv_auc'],
            'metrics': res['metrics'],
        }
    with open(os.path.join(OUT_MODELS, 'results_summary_v4.json'), 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(os.path.join(OUT_MODELS, 'probabilities_v4.json'), 'w') as f:
        json.dump(all_probs, f)

    y_tr = clin_tr['label'].values
    y_va = clin_va['label'].values
    y_te = clin_te['label'].values
    with open(os.path.join(OUT_MODELS, 'labels_split_v4.json'), 'w') as f:
        json.dump({'train': y_tr.tolist(), 'val': y_va.tolist(),
                   'test': y_te.tolist()}, f)

    log.info('=== Step 6 v4 建模完成 ===')
    for name, res in all_results.items():
        m = res['metrics']
        log.info(f"  {name:25s}: train={m['train']['AUC']:.3f}  "
                 f"val={m['val']['AUC']:.3f}  test={m['test']['AUC']:.3f}")


if __name__ == '__main__':
    main()
