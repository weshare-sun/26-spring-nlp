# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Word2Vec 实现与对比项目（NLP 课程作业）。三种独立实现：NumPy 手撸、PyTorch、Gensim baseline。支持英文 text8 和中文维基百科语料。

## Commands

```bash
# 训练（三种实现完全独立，互不依赖，均在 text8 5000 句子集上）
python -m numpy_version.train       # NumPy 版，约 59 分钟
python -m torch_version.train       # PyTorch 版（CPU），约 83 分钟
python -m gensim_version.train      # Gensim 版，约 22 秒

# 评估（运行全部 4 个阶段：效率/质量/可视化/跨语言）
python evaluation/run_all.py

# 单独评估
python evaluation/quality_eval.py        # 英文 35 对相似度 + 77 对类比；中文 40 对相似度 + 57 对类比
python evaluation/efficiency_analysis.py # 从 JSONL 日志读取训练时间 + Loss 曲线
python evaluation/visualization.py       # t-SNE / PCA / 热力图
python evaluation/hyperparameter_analysis.py  # 超参数消融(dim/window/negative)，复用 quality_eval 测试集
python evaluation/corpus_scaling.py      # 语料规模实验(10%/30%/50%/100%)，复用 quality_eval 测试集
python evaluation/cross_lingual.py       # 中英文对比

# 中文训练（默认 10% 子集；--full 用全量语料）
python train_chinese_wiki.py             # 约 3 分钟（10% 子集）
python train_chinese_wiki.py --full      # 约 38 分钟（全量 527K 行）
```

## Code Quality

**评估指标说明**：
- 词相似度使用 **Spearman 相关系数**（基于排名），而非 Pearson（线性关系）
- 使用 `scipy.stats.spearmanr` 计算

**性能优化**：
- 类比评估使用向量化计算，O(n) → O(1)
- 相似词查找使用反向索引，O(n) → O(1)
- workers 参数使用 `cpu_count()` 动态获取

## Architecture

- **三个版本完全独立**：`numpy_version/`、`torch_version/`、`gensim_version/` 各自包含完整 pipeline（preprocess → model → train → evaluate），删除任意一个不影响其他。
- **共享模块**：`common/download_data.py`（text8 下载）、`common/download_chinese.py`（中文语料工具）。
- **评估体系**：`evaluation/` 目录包含 6 个独立评估脚本，`run_all.py` 串联前 4 个。所有结果输出到 `evaluation/results/`（JSON + PNG）。`hyperparameter_analysis.py` 和 `corpus_scaling.py` 复用 `quality_eval.py` 的完整测试集（35 对相似度 + 77 对类比）。
- **训练日志**：三个训练脚本在各自 `output/training_log.jsonl` 写结构化日志（NumPy/PyTorch 逐 batch，Gensim 逐 epoch）。`efficiency_analysis.py` 从这些日志读取数据。
- **训练产出**：每个版本的 `output/` 目录存放 `vocab.json` + `word_vectors.npy`，评估脚本从这些文件加载。中文模型输出到 `chinese_wiki_output/`（10% 子集）或 `chinese_wiki_full_output/`（全量）。当前评估默认指向全量目录。
- **NumPy 版本的性能瓶颈**：`numpy_version/model.py` 的 `backward` 方法使用纯 Python for 循环更新梯度，是训练慢的根本原因。

## Key Parameters

三种实现共用相同默认超参：`embedding_dim=100, window=5, negative=5, min_count=5, sg=1(Skip-gram), epochs=5`。三者均默认 `max_sentences=5000`（text8 子集）。workers 统一用 `cpu_count()` 动态获取。

中文模型参数：`embedding_dim=100, window=5, negative=5, min_count=3, sg=1, epochs=5`（min_count 降低以适应中文分词后的长尾词）。

## Chinese Training

中文训练流程（`train_chinese_wiki.py`）：
1. 读取 `data/zhwiki_raw_10pct.txt`（10% 子集，~53K 行）或 `data/zhwiki_raw.txt`（全量，527K 行，1.8GB）
2. 多进程 jieba 分词（`cpu_count()` workers）→ `data/zhwiki_seg_10pct.txt` 或 `zhwiki_seg_full.txt`
3. Gensim `corpus_file` 模式训练（真正多线程并行），带 epoch 级 loss 回调
4. 输出到 `chinese_wiki_output/`（10%）或 `chinese_wiki_full_output/`（全量）

完整中文维基百科 dump 在 `data/zhwiki-latest-pages-articles.xml.bz2`（3.3GB），提取的纯文本在 `data/zhwiki_raw.txt`。

## Dependencies

```bash
pip install torch gensim numpy matplotlib scikit-learn jieba -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## Paper

论文 LaTeX 源码在 `paper/` 目录，Overleaf 编译需设为 XeLaTeX。图片引用路径为 `figures/`，需将 `evaluation/results/*.png` 上传到 Overleaf 的 `figures/` 文件夹。
