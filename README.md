```bash
cd scripts/
nohup python run_v4.py > ../output_v4/logs/run_v4.log 2>&1 &
```
所有参数在 `config_v4.py` 里改。

## Pipeline

| 脚本 | 干什么 |
|------|--------|
| step1_preprocess_v4.py | resample到1mm + zscore(clip ±3) |
| step2_peritumoral_v4.py | 膨胀生成围瘤区(5/10/15mm) |
| step3_habitat_v4.py | 12维体素特征 + KMeans K=3 |
| step3b_peri_habitat_v4.py | 围瘤区也做habitat聚类 |
| step4_features_v4.py | pyradiomics 8个区域 + 生态学特征 |
| step4b_icc_filter_v4_parallel.py | mask扰动算ICC，留>=0.75的 |
| step4c_dl_features_v4.py | ResNet50/DenseNet121 -> PCA 50维 |
| step5_v4.py | 方差->MWU->去冗余->mRMR->LASSO |
| step6_v4.py | ElasticNet建模(Optuna调参) |
| step7_v4.py | Cox-LASSO + RSF + KM曲线 |
| step8_v4.py | 出图 |

## 各步骤说明

**预处理(step1):** B-spline重采样到1mm各向同性，基于肿瘤ROI做zscore标准化，clip到[-3,3]输出float32。代码里保留了N4偏场校正，但CT其实不需要这步，对结果没影响就没删。

**围瘤区(step2):** 二值膨胀减去原始mask得到围瘤环，膨胀半径按spacing换算物理距离。主分析用10mm。

**Habitat聚类(step3/3b):** 每个体素提取12维特征(原始强度+LoG 4尺度+GLCM 3纹理+局部统计4维)，GPU KMeans聚成3类，按强度排序H1<H2<H3。围瘤区同样处理。GLCM在3x3x3窗口内算，26方向取平均。

**特征提取(step4):** 8个区域(intra/peri10mm/H1/H2/H3/periH1/periH2/periH3)分别用pyradiomics提取。binWidth=25，特征类shape/firstorder/glcm/glrlm/glszm/gldm，图像类型Original+LoG(2,3,5)+Wavelet。另外算了habitat间的生态学特征(体积比/Shannon熵/Simpson/均匀度/强度对比/边界比率等)。

**ICC过滤(step4b):** 随机50例，每例5种mask扰动(腐蚀/膨胀/随机丢弃5%/边界加5%/平移)，算ICC(3,1)，阈值0.75。多进程跑的。

**DL特征(step4c):** ResNet50(2048维)和DenseNet121(1024维)的倒数第二层特征，PCA降到50维。PCA只在训练集fit。输入是最大肿瘤切片附近的ROI裁剪。

**特征筛选(step5):** 只在训练集上做。ICC过滤 -> 方差(std>0.01) -> MWU(p<0.05) -> Pearson去冗余(|r|>0.9) -> mRMR(top40) -> ElasticNet CV -> 最多留20个。

**建模(step6):** 10个模型(Clinical / Intra / Peri / Habitat / PeriHabitat / DL / Rad+Clinical / Habitat+Clinical / DL+Rad+Clinical / Habitat+DL+Clinical)。ElasticNet逻辑回归，Optuna 200 trials调C和l1_ratio，5折x3次CV，SMOTE在CV内部做。

**生存分析(step7):** Cox-LASSO 5折CV选penalizer，RSF 200棵树max_depth=4。算了12/24/36月的time-dependent AUC。KM曲线用log-rank搜最优cutpoint。

**出图(step8):** NPG配色，L形坐标轴，白底。Fig2 ROC / Fig3校准 / Fig4 DCA / Fig5模型对比 / Fig6 SHAP / Fig8 KM / Fig10 TD-AUC / FigS2 ICC分布 / FigS4混淆矩阵。

## 数据划分

train 65% / val 15% / test 20%，分层抽样。复用v3的split保证可比。结果存在 `output_v4/models/data_split_v4.json`。

## 输出

```
output_v4/
├── preprocessed/       # 预处理后nii.gz
├── peritumoral/        # 围瘤区mask
├── habitat_masks/      # H1/H2/H3 mask
├── peri_habitat_masks/ # 围瘤区habitat mask
├── features/
│   ├── raw/            # 各区域原始特征csv
│   ├── icc/            # ICC分数
│   ├── ecological/     # 生态学特征
│   ├── dl_features/    # DL特征
│   └── selected/       # 筛选后特征
├── models/             # pkl模型 + json结果
├── figures/            # png + pdf
└── logs/               # 日志
