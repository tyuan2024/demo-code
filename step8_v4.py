"""step8: 出图 (Nature style, PNG 300dpi + PDF)"""
import os
import sys
import json
import logging
import warnings
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.calibration import calibration_curve

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
from config_v4 import (OUT_MODELS, OUT_FIGURES, ALL_FIGURE, LOG_DIR,
                        NPG_COLORS, ENDPOINT_COL, TIME_COL)
from data_split_v4 import load_labels, load_survival_data, get_or_create_split

os.makedirs(OUT_FIGURES, exist_ok=True)
os.makedirs(ALL_FIGURE, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'step8_v4.log'), mode='w'),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)

# Nature style
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 8,
    'axes.linewidth': 0.6,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.2,
    'grid.linestyle': '--',
    'grid.linewidth': 0.4,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'lines.linewidth': 1.2,
    'legend.framealpha': 0.8,
    'legend.edgecolor': 'none',
})


def save_fig(fig, name):
    for ext in ['png', 'pdf']:
        path = os.path.join(OUT_FIGURES, f'{name}.{ext}')
        fig.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0.05)
        # Sync to all figure/
        dst = os.path.join(ALL_FIGURE, f'{name}.{ext}')
        import shutil
        shutil.copy2(path, dst)
    plt.close(fig)
    log.info(f'  Saved: {name}')


def load_results():
    path = os.path.join(OUT_MODELS, 'results_summary_v4.json')
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def load_probs():
    path = os.path.join(OUT_MODELS, 'probabilities_v4.json')
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def load_labels_split():
    path = os.path.join(OUT_MODELS, 'labels_split_v4.json')
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def load_survival_results():
    path = os.path.join(OUT_MODELS, 'survival_results_v4.json')
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


# --- Fig 2: ROC ---
def fig2_roc():
    results = load_results()
    probs = load_probs()
    labels = load_labels_split()
    if not results or not probs or not labels:
        return

    fig, axes = plt.subplots(1, 3, figsize=(7.08, 2.5))
    splits = ['train', 'val', 'test']
    titles = ['Training', 'Validation', 'Test']

    model_names = list(probs.keys())
    colors = NPG_COLORS[:len(model_names)]

    for ax, split, title in zip(axes, splits, titles):
        y_true = np.array(labels[split])
        for i, name in enumerate(model_names):
            if split not in probs[name]:
                continue
            y_prob = np.array(probs[name][split])
            if len(y_prob) != len(y_true):
                continue
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc = roc_auc_score(y_true, y_prob)
            ax.plot(fpr, tpr, color=colors[i % len(colors)],
                    label=f'{name} ({auc:.3f})', linewidth=1.0)

        ax.plot([0, 1], [0, 1], 'k--', linewidth=0.5, alpha=0.3)
        ax.set_xlabel('1 - Specificity')
        ax.set_ylabel('Sensitivity')
        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.legend(fontsize=5.5, loc='lower right')
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)

    fig.tight_layout()
    save_fig(fig, 'Fig2_ROC_curves')


# --- Fig 3: Calibration ---
def fig3_calibration():
    probs = load_probs()
    labels = load_labels_split()
    if not probs or not labels:
        return

    fig, ax = plt.subplots(figsize=(3.54, 3.15))
    y_test = np.array(labels['test'])
    colors = NPG_COLORS

    for i, (name, prob_data) in enumerate(probs.items()):
        y_prob = np.array(prob_data['test'])
        if len(y_prob) != len(y_test):
            continue
        try:
            fraction_pos, mean_pred = calibration_curve(
                y_test, y_prob, n_bins=8, strategy='uniform')
            ax.plot(mean_pred, fraction_pos, 's-',
                    color=colors[i % len(colors)],
                    label=name, markersize=3, linewidth=1.0)
        except Exception:
            pass

    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.5, alpha=0.3)
    ax.set_xlabel('Predicted probability')
    ax.set_ylabel('Observed frequency')
    ax.set_title('Calibration (Test)', fontsize=9, fontweight='bold')
    ax.legend(fontsize=5.5)
    fig.tight_layout()
    save_fig(fig, 'Fig3_calibration')


# --- Fig 4: DCA ---
def fig4_dca():
    probs = load_probs()
    labels = load_labels_split()
    if not probs or not labels:
        return

    fig, ax = plt.subplots(figsize=(3.54, 3.15))
    y_test = np.array(labels['test'])
    thresholds = np.linspace(0.01, 0.99, 100)
    prevalence = y_test.mean()

    # All / None baselines
    ax.plot(thresholds, prevalence - thresholds * (1 - prevalence) / (1 - thresholds + 1e-10),
            'k-', linewidth=0.5, label='Treat All')
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--', label='Treat None')

    colors = NPG_COLORS
    for i, (name, prob_data) in enumerate(probs.items()):
        y_prob = np.array(prob_data['test'])
        if len(y_prob) != len(y_test):
            continue
        net_benefits = []
        for t in thresholds:
            tp = ((y_prob >= t) & (y_test == 1)).sum()
            fp = ((y_prob >= t) & (y_test == 0)).sum()
            n = len(y_test)
            nb = tp / n - fp / n * t / (1 - t + 1e-10)
            net_benefits.append(nb)
        ax.plot(thresholds, net_benefits, color=colors[i % len(colors)],
                label=name, linewidth=1.0)

    ax.set_xlabel('Threshold probability')
    ax.set_ylabel('Net benefit')
    ax.set_title('Decision Curve Analysis', fontsize=9, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1, max(0.5, prevalence + 0.1))
    ax.legend(fontsize=5, loc='upper right')
    fig.tight_layout()
    save_fig(fig, 'Fig4_DCA')


# --- Fig 5: Model Comparison ---
def fig5_model_comparison():
    results = load_results()
    if not results:
        return

    models = list(results.keys())
    splits = ['train', 'val', 'test']
    aucs = {s: [results[m]['metrics'][s]['AUC'] for m in models] for s in splits}

    fig, ax = plt.subplots(figsize=(7.08, 3.15))
    x = np.arange(len(models))
    width = 0.25
    colors_bar = [NPG_COLORS[0], NPG_COLORS[1], NPG_COLORS[3]]

    for i, (split, color) in enumerate(zip(splits, colors_bar)):
        bars = ax.bar(x + i * width, aucs[split], width,
                      label=split.capitalize(), color=color, alpha=0.85)
        for bar, val in zip(bars, aucs[split]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=5)

    ax.set_ylabel('AUC')
    ax.set_title('Model Performance Comparison', fontsize=9, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(models, rotation=30, ha='right', fontsize=6)
    ax.legend(fontsize=7)
    ax.set_ylim(0, 1.15)
    fig.tight_layout()
    save_fig(fig, 'Fig5_model_comparison')


# --- Fig 6: SHAP ---
def fig6_shap():
    """SHAP蜂群图"""
    try:
        import shap
    except ImportError:
        log.warning('  SHAP not installed, skipping Fig6')
        return

    results = load_results()
    if 'Habitat_Radiomics' not in results:
        return

    model_path = os.path.join(OUT_MODELS, 'Habitat_Radiomics_model_v4.pkl')
    if not os.path.exists(model_path):
        return

    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    clf = model_data['clf']
    scaler = model_data['scaler']

    from config_v4 import OUT_FEAT_SEL
    feat_path = os.path.join(OUT_FEAT_SEL, 'selected_habitat_combined.csv')
    if not os.path.exists(feat_path):
        return

    df = pd.read_csv(feat_path)
    feat_cols = [c for c in df.columns if c not in ('case_id', 'label')]
    X = scaler.transform(df[feat_cols].values)
    X_df = pd.DataFrame(X, columns=feat_cols)

    explainer = shap.LinearExplainer(clf, X_df)
    shap_values = explainer.shap_values(X_df)

    fig, ax = plt.subplots(figsize=(3.54, 4.72))
    shap.summary_plot(shap_values, X_df, max_display=20,
                      show=False, plot_size=None)
    plt.title('SHAP Feature Importance', fontsize=9, fontweight='bold')
    fig = plt.gcf()
    save_fig(fig, 'Fig6_SHAP')


# --- Fig 8: KM ---
def fig8_km():
    surv_results = load_survival_results()
    if 'KM' not in surv_results:
        return

    surv_df = load_survival_data()
    split = get_or_create_split()

    # Recompute rad-score for KM
    km_data = surv_results['KM']
    cutpoint = km_data.get('cutpoint', 0.5)

    # Load rad-score
    from config_v4 import OUT_FEAT_SEL
    for model_name in ['Habitat_Radiomics', 'Radiomics_Intra']:
        model_path = os.path.join(OUT_MODELS, f'{model_name}_model_v4.pkl')
        region_map = {'Habitat_Radiomics': 'habitat_combined',
                      'Radiomics_Intra': 'intra'}
        region = region_map.get(model_name)
        feat_path = os.path.join(OUT_FEAT_SEL, f'selected_{region}.csv')
        if os.path.exists(model_path) and os.path.exists(feat_path):
            with open(model_path, 'rb') as f:
                md = pickle.load(f)
            feat_df = pd.read_csv(feat_path)
            feat_df['case_id'] = feat_df['case_id'].astype(str)
            feat_cols = [c for c in feat_df.columns if c not in ('case_id', 'label')]
            X = md['scaler'].transform(feat_df[feat_cols].values)
            probs = md['clf'].predict_proba(X)[:, 1]
            rad_df = pd.DataFrame({'case_id': feat_df['case_id'], 'rad_score': probs})
            surv_df = surv_df.merge(rad_df, on='case_id', how='left')
            surv_df['rad_score'] = surv_df['rad_score'].fillna(surv_df['rad_score'].median())
            break

    if 'rad_score' not in surv_df.columns:
        return

    surv_df['risk_group'] = np.where(surv_df['rad_score'] >= cutpoint, 'High', 'Low')

    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test

    fig, ax = plt.subplots(figsize=(3.54, 3.54))
    colors_km = [NPG_COLORS[0], NPG_COLORS[2]]

    for grp, color in zip(['High', 'Low'], colors_km):
        sub = surv_df[surv_df['risk_group'] == grp]
        kmf = KaplanMeierFitter()
        kmf.fit(sub[TIME_COL], sub[ENDPOINT_COL], label=f'{grp} risk')
        kmf.plot_survival_function(ax=ax, ci_show=True, color=color,
                                    linewidth=1.2)

    high = surv_df[surv_df['risk_group'] == 'High']
    low = surv_df[surv_df['risk_group'] == 'Low']
    lr = logrank_test(high[TIME_COL], low[TIME_COL],
                      high[ENDPOINT_COL], low[ENDPOINT_COL])

    ax.set_xlabel('Time (months)')
    ax.set_ylabel('Recurrence-free survival')
    ax.set_title('Kaplan-Meier Curves', fontsize=9, fontweight='bold')
    ax.text(0.95, 0.95, f'Log-rank p={lr.p_value:.2e}',
            transform=ax.transAxes, ha='right', va='top', fontsize=7)
    ax.legend(fontsize=7)
    fig.tight_layout()
    save_fig(fig, 'Fig8_KM_curves')


# --- Fig 10: TD-AUC ---
def fig10_td_auc():
    path = os.path.join(OUT_MODELS, 'time_dependent_auc_v4.json')
    if not os.path.exists(path):
        return

    with open(path) as f:
        td_auc = json.load(f)

    if not td_auc:
        return

    fig, ax = plt.subplots(figsize=(3.54, 3.15))
    times = [12, 24, 36]
    aucs = [td_auc.get(f'AUC_{t}m', 0) for t in times]

    ax.plot(times, aucs, 'o-', color=NPG_COLORS[0], linewidth=1.5, markersize=6)
    for t, a in zip(times, aucs):
        ax.annotate(f'{a:.3f}', (t, a), textcoords='offset points',
                    xytext=(0, 10), ha='center', fontsize=7)

    ax.set_xlabel('Time (months)')
    ax.set_ylabel('AUC')
    ax.set_title('Time-dependent AUC', fontsize=9, fontweight='bold')
    ax.set_xticks(times)
    ax.set_ylim(0.4, 1.0)
    fig.tight_layout()
    save_fig(fig, 'Fig10_time_dependent_AUC')


# --- Supplementary ---
def figS2_icc_distribution():
    from config_v4 import OUT_FEAT_ICC, REGIONS
    fig, axes = plt.subplots(2, 4, figsize=(7.08, 3.54))
    axes = axes.flatten()

    for i, region in enumerate(REGIONS):
        icc_path = os.path.join(OUT_FEAT_ICC, f'icc_scores_{region}.json')
        if not os.path.exists(icc_path):
            axes[i].set_visible(False)
            continue
        with open(icc_path) as f:
            icc_scores = json.load(f)
        values = list(icc_scores.values())
        axes[i].hist(values, bins=30, color=NPG_COLORS[i % len(NPG_COLORS)],
                     alpha=0.7, edgecolor='white', linewidth=0.3)
        axes[i].axvline(0.75, color='red', linestyle='--', linewidth=0.8)
        axes[i].set_title(region, fontsize=7)
        axes[i].set_xlabel('ICC', fontsize=6)
        axes[i].tick_params(labelsize=5)

    fig.suptitle('ICC Distribution per Region', fontsize=9, fontweight='bold')
    fig.tight_layout()
    save_fig(fig, 'FigS2_ICC_distribution')


def figS4_confusion_matrices():
    results = load_results()
    probs = load_probs()
    labels = load_labels_split()
    if not results or not probs or not labels:
        return

    from sklearn.metrics import confusion_matrix as cm_func
    models = list(probs.keys())
    n = len(models)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(7.08, 2.0 * nrows))
    if nrows == 1:
        axes = [axes] if ncols == 1 else list(axes)
    else:
        axes = axes.flatten()

    y_test = np.array(labels['test'])
    for i, name in enumerate(models):
        y_prob = np.array(probs[name]['test'])
        if len(y_prob) != len(y_test):
            axes[i].set_visible(False)
            continue
        y_pred = (y_prob >= 0.5).astype(int)
        cm = cm_func(y_test, y_pred, labels=[0, 1])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                    xticklabels=['Neg', 'Pos'], yticklabels=['Neg', 'Pos'],
                    cbar=False, annot_kws={'size': 7})
        axes[i].set_title(name, fontsize=6)
        axes[i].tick_params(labelsize=5)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Confusion Matrices (Test Set)', fontsize=9, fontweight='bold')
    fig.tight_layout()
    save_fig(fig, 'FigS4_confusion_matrices')


def main():
    log.info('生成 v4 图表 ...')

    fig2_roc()
    fig3_calibration()
    fig4_dca()
    fig5_model_comparison()
    fig6_shap()
    fig8_km()
    fig10_td_auc()

    # Supplementary
    figS2_icc_distribution()
    figS4_confusion_matrices()

    # Count generated figures
    figs = [f for f in os.listdir(ALL_FIGURE)
            if f.endswith('.png') and f.startswith('Fig')]
    log.info(f'=== Step 8 v4 完成: {len(figs)} figures in all figure/ ===')


if __name__ == '__main__':
    main()
