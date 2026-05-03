# HDSAD：跨语料词向量漂移分析（目录与复现说明）

本仓库在**人民日报**与**微博**两路中文词向量上，实现自空间对齐、词级语义漂移度量、联合主成分表示、统一聚类标签下的群体层指标，至嵌入空间几何层指标的完整分析链。输入向量与公开语料格式可与 [Chinese Word Vectors（Embedding/Chinese-Word-Vectors）](https://github.com/Embedding/Chinese-Word-Vectors) 项目所发布资源及说明对照；下游流程以 Gensim `KeyedVectors`（`.kv`）为工作格式。数值产出集中于 `result/`，图形产出集中于 `figures/`；源代码位于 `src/`，路径由 `src/paths.py` 统一解析为相对仓库根的相对路径。在非仓库根作为当前工作目录运行时，可设置环境变量 `HDSAD_ROOT` 指向仓库根路径。

---

## 一、目录结构

### 1.1 数值结果（`result/`）

| 路径 | 内容 |
|------|------|
| `result/alignment/` | KeyedVectors（`.kv`）及对齐相关文本产物；约定文件名包括 `renmin_fast.kv`、`weibo_fast.kv`、`weibo_aligned_v2.kv` 等。 |
| `result/word/` | 词级漂移表（如 `shift_results_v2.csv`）及统计元数据（如 `shift_meta.json`）。 |
| `result/joint_pca/` | 联合 PCA 表示（如 `semantic_space_pca.npz`）。 |
| `result/cluster/` | 聚类标签与嵌入（`semantic_embedding_cluster.npz`）、群体层指标 CSV、簇–词表（`cluster_words_by_category.csv`）等。 |
| `result/geometric/` | 几何层指标对应的 CSV 与 NPZ（文件名前缀 `geometric_space_`）。 |

### 1.2 图形输出（`figures/`）

| 路径 | 内容 |
|------|------|
| `figures/alignment/` | 对齐与锚点实验示意图（由 `src/alignment` 下脚本生成）。 |
| `figures/word/` | 词级漂移分布及示例词图。 |
| `figures/cluster/` | 群体层指标可视化。 |
| `figures/geometric/` | 几何层指标可视化。 |

### 1.3 源代码（`src/`）

| 组件 | 功能摘要 |
|------|----------|
| `paths.py` | 定义 `result/*`、`figures/*` 等相对路径段及常用文件名常量。 |
| `alignment/transfer_format.py` | 将外部 word2vec 文本格式转为 `.kv`；输入由 `--renmin-txt`、`--weibo-txt` 指定。 |
| `alignment/utils.py` | 对齐实验共用例程与设备配置；自 `result/alignment/` 加载基向量。 |
| `alignment/spatial_alignment_v2.py` | 正交 Procrustes 与 CSLS 迭代；输出对齐后微博向量及 `figures/alignment/` 中诊断图。 |
| `alignment/anchor_size_eval.py` | 锚点规模敏感性曲线及 `Q`、`Q^TQ` 可视化。 |
| `word/compute_shift.py` | 词级漂移分数及显著性标记；写入 `result/word/`。 |
| `word/plot_figures.py` | 基于 `result/word/` 中结果生成 `figures/word/` 中图形。 |
| `word/verify_alignment.py` | 锚点集上的对齐性质检验；可选依赖 `result/alignment/renmin_anchor_sorted.kv`。 |
| `joint_pca/semantic_space_pca.py` | 两域 L2 行归一化后联合 PCA（50 维）；输出 `result/joint_pca/semantic_space_pca.npz`。 |
| `cluster/semantic_embedding_cluster.py` | 全量 KMeans；输出 `result/cluster/semantic_embedding_cluster.npz`。 |
| `cluster/cluster_centroid_drift.py` | 簇质心跨源 L2 漂移。 |
| `cluster/cluster_intra_compactness.py` | 类内紧致度（均方距离至子集质心）。 |
| `cluster/cluster_separability_metrics.py` | 簇间质心距离矩阵与轮廓系数（可子采样）。 |
| `cluster/cluster_drift_visualization.py` | 读取群体层 CSV，写入 `figures/cluster/`。 |
| `cluster/semantic_cluster_visualization.py` | 聚类结果探索性二维投影及簇–词表导出（默认交互显示）。 |
| `geometric/geometric_space_density_metrics.py` | 点对距离直方统计。 |
| `geometric/geometric_space_covariance_metrics.py` | 协方差谱与广义方差。 |
| `geometric/geometric_space_procrustes_metrics.py` | 正交普鲁克残差。 |
| `geometric/geometric_space_visualization.py` | 几何层图形输出至 `figures/geometric/`。 |

## 二、运行环境与依赖

分析脚本基于 **Python 3.9+**。依赖库见仓库根目录 `requirements.txt`，可在虚拟环境中执行：

```text
pip install -r requirements.txt
```

其中 `torch` 用于对齐脚本中的张量运算与 SVD（CPU 与 GPU 均可）；`gensim` 用于词向量读写与格式转换；`scikit-learn` 用于 PCA、KMeans 与轮廓系数等；`matplotlib`、`pandas` 用于图形与表格读写。对 `src/` 下全部模块作语法检查可在仓库根执行：

```text
python -m compileall -q src
```

---

## 三、建议计算顺序

以下命令均在仓库根目录执行。前提为 `result/alignment/` 中已具备 `renmin_fast.kv`、`weibo_fast.kv`，并完成对齐得到 `weibo_aligned_v2.kv`（若已满足可省略步骤 1）。

1. `python src/alignment/spatial_alignment_v2.py`  
2. `python src/word/compute_shift.py`；`python src/word/plot_figures.py`  
3. `python src/joint_pca/semantic_space_pca.py`  
4. `python src/cluster/semantic_embedding_cluster.py`  
5. `python src/cluster/cluster_centroid_drift.py`  
6. `python src/cluster/cluster_intra_compactness.py`  
7. `python src/cluster/cluster_separability_metrics.py`  
8. `python src/cluster/cluster_drift_visualization.py`  
9. 按需：`python src/geometric/geometric_space_density_metrics.py`、`python src/geometric/geometric_space_covariance_metrics.py`、`python src/geometric/geometric_space_procrustes_metrics.py`  
10. `python src/geometric/geometric_space_visualization.py`  

---
