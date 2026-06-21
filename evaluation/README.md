# Word2Vec 实验评估

## 文件说明

| 文件 | 功能 | 产出 |
|------|------|------|
| `run_all.py` | 一键运行前 4 项评估 | - |
| `efficiency_analysis.py` | 训练效率对比（时间、Loss 曲线） | `time_comparison.png`, `loss_curves.png` |
| `quality_eval.py` | 词向量质量评估（35 对英+40 对中相似度 + 77 对英+57 对中类比） | `quality_eval.json` |
| `visualization.py` | t-SNE / PCA / 热力图可视化 | `*_tsne.png`, `*_pca.png`, `*_heatmap.png` |
| `cross_lingual.py` | 中英文对比分析 | `cross_lingual.json`, `cross_lingual_comparison.png` |
| `hyperparameter_analysis.py` | 超参数消融（dim/window/negative） | `hyperparameter_analysis.json`, `hyperparameter_analysis.png` |
| `corpus_scaling.py` | 语料规模实验（10%/30%/50%/100%） | `corpus_scaling.json`, `corpus_scaling.png` |

## 使用方法

```bash
cd myWord2vec

# 运行所有评估（效率 + 质量 + 可视化 + 跨语言）
python evaluation/run_all.py

# 单独运行某项
python evaluation/efficiency_analysis.py
python evaluation/quality_eval.py
python evaluation/visualization.py
python evaluation/cross_lingual.py
python evaluation/hyperparameter_analysis.py
python evaluation/corpus_scaling.py
```

## 输出

所有结果保存到 `evaluation/results/` 目录：
- `*.png` - 可视化图表
- `*.json` - 统计数据

## 评估指标

- **词相似度**：Spearman 相关系数（模型余弦相似度 vs 人工评分）
- **类比推理**：准确率（a - b + c = d，50 组覆盖 8 个类别）
- **训练效率**：总训练时间、Loss 收敛曲线
- **可视化**：t-SNE/PCA 降维、相似度热力图
