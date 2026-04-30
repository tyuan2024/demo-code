"""数据划分 v4 — 复用v3 split"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, os.path.dirname(__file__))
from config_v4 import (CLINICAL_PATH, ENDPOINT_COL, TIME_COL,
                        CUTOFF_MONTHS, OUT_MODELS, RANDOM_STATE,
                        TEST_RATIO, VAL_RATIO)


def load_labels() -> pd.DataFrame:
    df = pd.read_excel(CLINICAL_PATH, header=0, skiprows=[1])
    df.insert(0, 'case_id', range(1, len(df) + 1))
    df['case_id'] = df['case_id'].astype(str)
    df[ENDPOINT_COL] = pd.to_numeric(df[ENDPOINT_COL], errors='coerce')
    df[TIME_COL] = pd.to_numeric(df[TIME_COL], errors='coerce')
    df = df.dropna(subset=[TIME_COL, ENDPOINT_COL])
    df[ENDPOINT_COL] = df[ENDPOINT_COL].astype(int)

    def make_binary(row):
        t, e = row[TIME_COL], row[ENDPOINT_COL]
        if e == 1 and t <= CUTOFF_MONTHS:
            return 1
        elif t >= CUTOFF_MONTHS:
            return 0
        return np.nan

    df['label'] = df.apply(make_binary, axis=1)
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    return df


def _clean_clinical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """对分类/混合格式临床变量做proper encoding"""
    import re

    # 1. 性别: 男=1, 女=0
    if '性别' in df.columns:
        df['性别'] = df['性别'].map({'男': 1, '女': 0})

    # 2. CNLC分期: ordinal Ia=1, Ib=2, IIa=3, IIb=4, IIIa=5
    if 'CNLC分期' in df.columns:
        cnlc_map = {'Ia': 1, 'Ib': 2, 'IIa': 3, 'IIb': 4, 'IIIa': 5, 'IIIb': 6}
        df['CNLC分期'] = df['CNLC分期'].map(cnlc_map)

    # 3. BCLC分期: ordinal 0=0, A=1, B=2, C=3, D=4
    if 'BCLC分期' in df.columns:
        bclc_map = {0: 0, '0': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4}
        df['BCLC分期'] = df['BCLC分期'].map(bclc_map)

    # 4. 肿瘤分级: 统一为数值 (1/1.5/2/2.5/3/3.5/4)
    if '肿瘤分级' in df.columns:
        def parse_grade(v):
            if pd.isna(v):
                return np.nan
            s = str(v).strip()
            # Roman numeral mapping
            roman = {'I': 1, 'II': 2, 'III': 3, 'IV': 4}
            # Handle "II~III", "II-III" style
            for sep in ['~', '-', '～']:
                if sep in s:
                    parts = s.split(sep)
                    nums = []
                    for p in parts:
                        p = p.strip()
                        if p in roman:
                            nums.append(roman[p])
                        else:
                            try:
                                nums.append(float(p))
                            except ValueError:
                                pass
                    if len(nums) == 2:
                        return (nums[0] + nums[1]) / 2
            # Handle "2，3" style (Chinese comma)
            for sep in ['，', ',']:
                if sep in s:
                    parts = s.split(sep)
                    nums = []
                    for p in parts:
                        p = p.strip()
                        if p in roman:
                            nums.append(roman[p])
                        else:
                            try:
                                nums.append(float(p))
                            except ValueError:
                                pass
                    if len(nums) == 2:
                        return (nums[0] + nums[1]) / 2
            # Single roman
            if s in roman:
                return float(roman[s])
            # Single number
            try:
                return float(s)
            except ValueError:
                return np.nan
        df['肿瘤分级'] = df['肿瘤分级'].apply(parse_grade)

    # 5. 异常凝血酶原: "无"→0, ">75000"→75000
    if '异常凝血酶原' in df.columns:
        def parse_pivka(v):
            if pd.isna(v):
                return np.nan
            s = str(v).strip()
            if s == '无':
                return 0.0
            m = re.match(r'[>＞](\d+)', s)
            if m:
                return float(m.group(1))
            try:
                return float(s)
            except ValueError:
                return np.nan
        df['异常凝血酶原'] = df['异常凝血酶原'].apply(parse_pivka)

    # 6. AFP甲胎: "无"→0, ">60500"→60500
    if 'AFP甲胎' in df.columns:
        def parse_afp(v):
            if pd.isna(v):
                return np.nan
            s = str(v).strip()
            if s == '无':
                return 0.0
            m = re.match(r'[>＞](\d+)', s)
            if m:
                return float(m.group(1))
            try:
                return float(s)
            except ValueError:
                return np.nan
        df['AFP甲胎'] = df['AFP甲胎'].apply(parse_afp)

    # 7. CEA癌胚: "无"→0, "<0.2"→0.1 (half LOD)
    if 'CEA癌胚' in df.columns:
        def parse_cea(v):
            if pd.isna(v):
                return np.nan
            s = str(v).strip()
            if s == '无':
                return 0.0
            m = re.match(r'[<＜]([0-9.]+)', s)
            if m:
                return float(m.group(1)) / 2
            try:
                return float(s)
            except ValueError:
                return np.nan
        df['CEA癌胚'] = df['CEA癌胚'].apply(parse_cea)

    # 8. CA199: "无"→0, "<0.6"→0.3
    if 'CA199' in df.columns:
        def parse_ca199(v):
            if pd.isna(v):
                return np.nan
            s = str(v).strip()
            if s == '无':
                return 0.0
            m = re.match(r'[<＜]([0-9.]+)', s)
            if m:
                return float(m.group(1)) / 2
            try:
                return float(s)
            except ValueError:
                return np.nan
        df['CA199'] = df['CA199'].apply(parse_ca199)

    # 9. INR国际标准: "1.1." → 1.1
    if 'INR国际标准' in df.columns:
        def parse_inr(v):
            if pd.isna(v):
                return np.nan
            s = str(v).strip().rstrip('.')
            try:
                return float(s)
            except ValueError:
                return np.nan
        df['INR国际标准'] = df['INR国际标准'].apply(parse_inr)

    return df


def load_clinical_features(labels_df: pd.DataFrame):
    df_full = pd.read_excel(CLINICAL_PATH, header=0, skiprows=[1])
    df_full.insert(0, 'case_id', range(1, len(df_full) + 1))
    df_full['case_id'] = df_full['case_id'].astype(str)

    # Clean categorical/mixed columns BEFORE numeric conversion
    df_full = _clean_clinical_columns(df_full)

    cand = [
        '性别', '年龄', '肿瘤最大径', '肿瘤数目', '卫星灶', '肿瘤分级',
        '肝硬化', 'MVI', '乙肝史', 'AFP甲胎', 'CEA癌胚', 'CA199',
        'BCLC分期', 'CNLC分期',
        'PV癌栓（1：是；0：否）', '胆管癌栓（1：是；0：否）',
        'TB总胆红素', 'ALB白蛋白', 'ALT丙氨酸', 'AST门冬氨酸',
        'GGT谷氨酰', 'PT凝血酶原时间', 'INR国际标准',
        'PLT血小板', 'WBC白细胞', 'Hb血红蛋白',
        'L淋巴细胞', 'N中性粒细胞', '异常凝血酶原',
    ]
    avail = [c for c in cand if c in df_full.columns]
    merged = labels_df[['case_id', 'label']].merge(
        df_full[['case_id'] + avail], on='case_id', how='inner')

    for col in avail:
        merged[col] = pd.to_numeric(merged[col], errors='coerce')
        col_median = merged[col].median()
        if pd.notna(col_median) and merged[col].isna().any():
            merged[col] = merged[col].fillna(col_median)

    # Sanity check: no all-NaN columns
    all_nan_cols = [c for c in avail if merged[c].isna().all()]
    if all_nan_cols:
        raise ValueError(f"Clinical columns still all-NaN after encoding: {all_nan_cols}")

    return merged, avail


def get_or_create_split():
    """8:1:1 分层划分；已有则复用"""
    split_path = os.path.join(OUT_MODELS, 'data_split_v4.json')
    if os.path.exists(split_path):
        with open(split_path) as f:
            return json.load(f)

    labels_df = load_labels()
    ids_all = labels_df['case_id'].values
    y_all = labels_df['label'].values

    ids_trainval, ids_test, y_trainval, _ = train_test_split(
        ids_all, y_all, test_size=TEST_RATIO,
        stratify=y_all, random_state=RANDOM_STATE)
    val_frac = VAL_RATIO / (1 - TEST_RATIO)
    ids_train, ids_val, _, _ = train_test_split(
        ids_trainval, y_trainval, test_size=val_frac,
        stratify=y_trainval, random_state=RANDOM_STATE)

    split = {
        'train': ids_train.tolist(),
        'val': ids_val.tolist(),
        'test': ids_test.tolist(),
    }
    os.makedirs(OUT_MODELS, exist_ok=True)
    with open(split_path, 'w') as f:
        json.dump(split, f)
    return split


def load_survival_data():
    df = pd.read_excel(CLINICAL_PATH, header=0, skiprows=[1])
    df.insert(0, 'case_id', range(1, len(df) + 1))
    df['case_id'] = df['case_id'].astype(str)
    df[ENDPOINT_COL] = pd.to_numeric(df[ENDPOINT_COL], errors='coerce')
    df[TIME_COL] = pd.to_numeric(df[TIME_COL], errors='coerce')
    df = df.dropna(subset=[TIME_COL, ENDPOINT_COL])
    df[ENDPOINT_COL] = df[ENDPOINT_COL].astype(int)
    df[TIME_COL] = df[TIME_COL].astype(float)
    return df
