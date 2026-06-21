# 实验结果使用说明

> 本说明供撰写论文的同学参考，帮助快速找到并使用实验可视化结果和数据。

---

## 目录

- [输出文件总览](#输出文件总览)
- [训练效率相关](#训练效率相关)
- [语义质量评估](#语义质量评估)
- [可视化图表](#可视化图表)
- [超参数消融实验](#超参数消融实验)
- [语料规模实验](#语料规模实验)
- [跨语言对比](#跨语言对比)
- [数据引用格式](#数据引用格式)

---

## 输出文件总览

所有文件位于 `evaluation/results/` 目录：

```
evaluation/results/
├── efficiency_stats.json          # 训练时间统计
├── loss_curves.png                # Loss 收敛曲线图
├── time_comparison.png            # 三种实现训练时间对比图
│
├── quality_eval.json              # 词相似度 + 类比推理详细数据
│
├── gensim_tsne.png                # Gensim 版 t-SNE 可视化
├── gensim_pca.png                 # Gensim 版 PCA 可视化
├── gensim_heatmap.png             # Gensim 版相似度热力图
│
├── chinese_tsne.png               # 中文 t-SNE 可视化
├── chinese_pca.png                # 中文 PCA 可视化
├── chinese_heatmap.png            # 中文相似度热力图
│
├── hyperparameter_analysis.json   # 超参数消融数据
├── hyperparameter_analysis.png    # 超参数消融图表
│
├── corpus_scaling.json            # 语料规模实验数据
├── corpus_scaling.png             # 语料规模图表
│
├── cross_lingual.json             # 跨语言对比数据
└── cross_lingual_comparison.png   # 跨语言对比图表
```

---

## 训练效率相关

### 图片

| 文件 | 内容 | 建议用途 |
|------|------|---------|
| `time_comparison.png` | 三种实现的训练时间柱状图 | 论文 4.2 节，展示效率差异 |
| `loss_curves.png` | NumPy 和 PyTorch 的 Loss 收敛曲线 | 论文 4.2 节，展示收敛性 |

### 数据

效率分析从各版本的 `output/training_log.jsonl` 读取训练过程数据（NumPy/PyTorch 逐 batch，Gensim 逐 epoch）。

`efficiency_stats.json` 包含训练时间的结构化数据。

**训练时间汇总（实际值以重新训练后为准）：**

| 实现 | 训练时间 | 相对 Gensim |
|------|---------|------------|
| NumPy | 3,551s (59 min) | ×159 |
| PyTorch | 4,964s (83 min) | ×223 |
| Gensim | 22s | ×1（基准） |

---

## 语义质量评估

### 数据

`quality_eval.json` 包含每个版本的完整评估数据：

```json
{
  "NumPy": {
    "vocab_size": 36280,
    "embedding_dim": 100,
    "similarity_correlation": 0.4273,   // Spearman 相关系数 (35 对英文)
    "analogy_accuracy": 0.1948,         // 类比准确率 (77 对英文)
    "similarity_results": [...],        // 词相似度详细结果
    "analogy_results": [...]            // 类比推理详细结果
  },
  "PyTorch": { ... },
  "Gensim": { ... },
  "Chinese": { ... }                    // 40 对中文相似度 + 57 对中文类比
}
```

**汇总数据：**

| 版本 | 词表大小 | Spearman 相关系数 | 类比准确率 |
|------|---------|-------------------|-------------------|
| NumPy | 36,280 | 0.4273 | 19.48% (15/77) |
| PyTorch | 36,280 | 0.4599 | 20.78% (16/77) |
| Gensim | 36,280 | 0.4414 | 25.97% (20/77) |
| Chinese (10%) | 494,845 | 0.3927 | 33.33% (17/51) |
| Chinese (全量) | 2,237,325 | 0.4145 | 32.69% (17/52) |

---

## 可视化图表

### Gensim 版本（推荐用于论文）

| 文件 | 说明 |
|------|------|
| `gensim_tsne.png` | t-SNE 降维，展示语义聚类结构 |
| `gensim_pca.png` | PCA 降维，展示全局分布 |
| `gensim_heatmap.png` | 高频词两两相似度热力图 |

### 中文版本

| 文件 | 说明 |
|------|------|
| `chinese_tsne.png` | 中文词向量 t-SNE 可视化 |
| `chinese_pca.png` | 中文词向量 PCA 可视化 |
| `chinese_heatmap.png` | 中文高频词相似度热力图 |

---

## 超参数消融实验

### 图片

`hyperparameter_analysis.png` 包含三张子图：
- 左图：向量维度 (50/100/200) 对质量的影响
- 中图：窗口大小 (3/5/10) 对质量的影响
- 右图：负采样数 (3/5/10) 对质量的影响

### 数据

`hyperparameter_analysis.json` 结构：

```json
{
  "embedding_dim": [
    {"dim": 50, "similarity_correlation": 0.0566, "analogy_accuracy": 0.6667, "train_time": 115.0},
    {"dim": 100, ...},
    {"dim": 200, ...}
  ],
  "window_size": [...],
  "negative_sampling": [...]
}
```

**汇总：**

| 向量维度 | 相似度相关 | 类比准确率 | 训练时间 |
|---------|-----------|-----------|---------|
| 50 | 0.4710 | 22.08% | 9.2s |
| 100 | 0.4352 | 23.38% | 9.9s |
| 200 | 0.4515 | 25.97% | 13.2s |

| 窗口大小 | 相似度相关 | 类比准确率 | 训练时间 |
|---------|-----------|-----------|---------|
| 3 | 0.5148 | 22.08% | 8.0s |
| 5 | 0.5150 | 23.38% | 9.9s |
| 10 | 0.4266 | 24.68% | 16.0s |

| 负采样数 | 相似度相关 | 类比准确率 | 训练时间 |
|---------|-----------|-----------|---------|
| 3 | 0.4947 | 27.27% | 8.1s |
| 5 | 0.4807 | 20.78% | 10.0s |
| 10 | 0.4512 | 23.38% | 16.4s |

---

## 语料规模实验

### 图片

`corpus_scaling.png` 包含三张子图：
- 左图：词表大小随语料比例的变化
- 中图：相似度相关和类比准确率随语料比例的变化
- 右图：训练时间随语料比例的变化

### 数据

`corpus_scaling.json` 结构：

```json
[
  {"proportion": 0.1, "n_words": 1700520, "vocab_size": 19461, "train_time": 9.2, ...},
  {"proportion": 0.3, ...},
  {"proportion": 0.5, ...},
  {"proportion": 1.0, ...}
]
```

**汇总：**

| 语料比例 | 词数 | 词表大小 | 相似度相关 | 类比准确率 | 训练时间 |
|---------|------|---------|-----------|-----------|---------|
| 10% | 1.7M | 19,461 | 0.4245 | 9.21% | 3.1s |
| 30% | 5.1M | 36,773 | 0.5790 | 23.38% | 9.9s |
| 50% | 8.5M | 48,792 | 0.5449 | 29.87% | 17.5s |
| 100% | 17.0M | 71,290 | 0.5179 | 29.87% | 36.8s |

---

## 跨语言对比

### 图片

`cross_lingual_comparison.png`：英文与中文语义聚类平均相似度对比柱状图。

### 数据

`cross_lingual.json` 包含英文和中文的语义聚类数据。

**英文语义聚类：**

| 语义类别 | 平均相似度 | 标准差 | 示例词 |
|---------|-----------|-------|--------|
| 王室 | 0.681 | 0.022 | king, queen, prince |
| 国家 | 0.627 | 0.039 | france, germany, japan |
| 技术 | 0.634 | 0.066 | computer, software, keyboard |
| 动物 | 0.583 | 0.048 | dog, cat, bird |

---

## 数据引用格式

在论文中引用实验数据时，建议格式：

> 本实验在 text8 语料（约 1700 万词子集）上训练了三种 Word2Vec 实现。Gensim 仅需 22 秒完成训练，比 NumPy（59 分钟）快 159 倍，比 PyTorch（83 分钟）快 223 倍。在 77 组英文类比推理任务中，Gensim 以 25.97\% 准确率最优。词相似度任务上三个英文模型 Spearman ρ 为 0.427--0.460。中文模型使用原生中文测试集（40 对相似度 + 57 组类比），Spearman ρ 为 0.393，类比准确率 33.33\%。

图片引用时，注意路径需要调整为论文项目中的实际路径（如 `figures/time_comparison.png`）。

---

## 注意事项

1. **图片分辨率**：所有 PNG 图片均为 150 DPI，适合论文插入
2. **JSON 数据**：可直接用 Python 的 `json.load()` 读取，或复制到 Excel 中分析
3. **中文编码**：JSON 文件使用 UTF-8 编码，注意编辑器的编码设置
4. **重新生成**：如需重新生成结果，运行 `python evaluation/run_all.py`
