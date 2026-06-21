# Word2Vec 复现教程

> 大学生 NLP 作业：从零复现 Word2Vec

---

## 目录

- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [方案一：纯 NumPy 手撸](#方案一纯-numpy-手撸)
- [方案二：PyTorch 实现](#方案二pytorch-实现)
- [方案三：Gensim 对比](#方案三gensim-对比)
- [核心原理](#核心原理)
- [实验与可视化](#实验与可视化)
- [常见问题 FAQ](#常见问题-faq)

---

## 项目结构

```
myWord2vec/
│
├── common/                     # 共享工具
│   ├── download_data.py        #   下载 text8 数据集
│   └── download_chinese.py     #   中文语料工具
│
├── data/                       # 共享数据 (运行后生成)
│   ├── text8                   #   text8 英文语料
│   └── zhwiki-latest-pages-articles.xml.bz2  # 中文维基百科 dump
│
├── numpy_version/              # ← 方案一: 纯 NumPy
│   ├── preprocess.py           #   数据预处理 + 词表构建
│   ├── model.py                #   Word2Vec 模型 + 负采样器
│   ├── train.py                #   训练入口
│   ├── evaluate.py             #   评估: 相似词 / 词类比
│   └── output/                 #   训练产物
│
├── torch_version/              # ← 方案二: PyTorch
│   ├── model.py                #   Skip-gram 模型
│   ├── preprocess.py           #   数据预处理
│   ├── train.py                #   训练入口
│   ├── evaluate.py             #   评估
│   └── output/
│
├── gensim_version/             # ← 方案三: Gensim baseline
│   ├── train.py                #   训练入口
│   ├── evaluate.py             #   评估
│   └── output/
│
├── evaluation/                 # 实验评估体系
│   ├── run_all.py              #   一键运行前 4 项评估
│   ├── efficiency_analysis.py  #   训练效率对比
│   ├── quality_eval.py         #   词相似度(35对) + 类比推理(50对)
│   ├── visualization.py        #   t-SNE / PCA / 热力图
│   ├── cross_lingual.py        #   中英文对比
│   ├── hyperparameter_analysis.py  # 超参数消融
│   ├── corpus_scaling.py       #   语料规模实验
│   └── results/                #   评估结果 (JSON + PNG)
│
├── train_chinese_wiki.py       # 中文维基百科训练脚本
├── chinese_train.py            # 中文样本语料训练
├── paper/                      # 论文 LaTeX 源码
└── tutorial.md                 # 本文档
```

**三套方案完全独立，删除任意一个文件夹不影响其余方案运行。**

---

## 快速开始

```bash
# 1. 下载数据 (三个方案共用)
python common/download_data.py

# 2. 任选一个方案运行

# 方案一: NumPy
python -m numpy_version.train
python -m numpy_version.evaluate

# 方案二: PyTorch
python -m torch_version.train
python -m torch_version.evaluate

# 方案三: Gensim
pip install gensim
python -m gensim_version.train
python -m gensim_version.evaluate
```

> **注意**: 所有命令都在 `myWord2vec/` 目录下执行。

---

## 方案一：纯 NumPy 手撸

**推荐用于交作业**，最能展示你对算法的理解。

### 文件说明

| 文件 | 作用 |
|------|------|
| `preprocess.py` | `TextPreprocessor` 类: 构建词表、生成 (center, context) 训练对 |
| `model.py` | `Word2Vec` 类: 前向传播 (计算 loss) + 反向传播 (梯度更新) |
| `model.py` | `NegativeSampler` 类: 按 `freq^0.75` 分布采样负样本 |
| `train.py` | 训练入口: 超参数、训练循环、保存模型 |
| `evaluate.py` | 加载词向量，测试相似词和词类比 |

### 核心流程

```
1. load_text8()          → 读取语料，切分为句子列表
2. build_vocab()         → 统计词频，构建 word↔idx 映射
3. get_training_pairs()  → 滑动窗口生成 (center, context) 对
4. 训练循环:
   for each batch:
       negatives = sampler.sample(B)       # 负采样
       loss = model.forward(...)           # 前向: 计算 loss
       model.backward(...)                 # 反向: 更新 W_in, W_out
5. 保存 W_in 作为词向量
```

### 关键实现: 反向传播

```python
# 正样本梯度
pos_score = sigmoid(v_c · u_o) - 1       # (B,)
grad_u_pos = pos_score * v_c              # 对 W_out[context]
grad_v_pos = pos_score * u_o              # 对 W_in[center]

# 负样本梯度
neg_score = sigmoid(v_c · u_k)            # (B, K)
grad_u_neg = neg_score * v_c              # 对 W_out[neg]
grad_v_neg = Σ_k(neg_score * u_k)         # 对 W_in[center]

# SGD 更新
W_in[idx]  -= lr * grad
W_out[idx] -= lr * grad
```

---

## 方案二：PyTorch 实现

用自动微分，不需要手写反向传播。

### 与方案一的区别

| | NumPy 版 | PyTorch 版 |
|---|---------|-----------|
| 反向传播 | 手写 `backward()` | `loss.backward()` 自动计算 |
| 参数更新 | 手动 `W -= lr * grad` | `optimizer.step()` |
| GPU 加速 | 不支持 | 支持 (自动检测) |
| 代码量 | 多 | 少 |

### 文件说明

| 文件 | 作用 |
|------|------|
| `model.py` | `SkipGramNegSampling(nn.Module)` 模型定义 |
| `preprocess.py` | 复用 numpy 版预处理，额外转为 `torch.Tensor` |
| `train.py` | 训练循环，使用 `Adam` 优化器 |
| `evaluate.py` | 复用 numpy 版评估逻辑 |

---

## 方案三：Gensim 对比

用 Gensim 库训练作为 baseline，验证你自己实现的正确性。

```python
from gensim.models import Word2Vec

model = Word2Vec(
    sentences=sentences,
    vector_size=100,    # 词向量维度
    window=5,           # 窗口大小
    min_count=5,        # 最低词频
    sg=1,               # 1=Skip-gram, 0=CBOW
    negative=5,         # 负样本数
    epochs=5,
)

# 测试
model.wv.most_similar('king', topn=5)
model.wv.most_similar(positive=['king', 'woman'], negative=['man'], topn=3)
```

---

## 核心原理

### Word2Vec 两种模式

```
CBOW:  上下文 → 预测中心词
       [The, sat, on, the] → cat

Skip-gram: 中心词 → 预测上下文
       cat → [The, sat, on, the]
```

### 模型结构

```
      输入 (one-hot, V维)     隐藏层 (N维)     输出 (V维)
           V  ──────────>   N  ──────────>   V
               W_in (V×N)      W_out (N×V)

V = 词表大小
N = 词向量维度 (通常 100~300)
```

### 负采样

将 softmax 多分类转为多个二分类：

```
L = -log σ(u_o^T v_c) - Σ_{k=1}^{K} E[log σ(-u_k^T v_c)]

σ = sigmoid 函数
u_o = 正样本 (真实上下文词) 的输出向量
u_k = 负样本 (随机采样词) 的输出向量
v_c = 中心词的输入向量
K = 负样本个数 (通常 5~15)
```

**采样分布**: `P(w) ∝ f(w)^0.75`
- `f(w)` = 词频
- `^0.75` 的作用: 提高低频词被采样的概率，平衡高频/低频词

### 学习率衰减

```
lr = lr_start * (1 - progress) + lr_min * progress
progress = epoch / total_epochs
```
前期大学习率快速收敛，后期小学习率精细调整。

---

## 实验与可视化

### 一键运行评估

```bash
# 运行全部评估（效率 + 质量 + 可视化 + 跨语言）
python evaluation/run_all.py

# 单独运行
python evaluation/quality_eval.py        # 词相似度 + 类比推理
python evaluation/efficiency_analysis.py # 训练时间 + Loss 曲线
python evaluation/visualization.py       # t-SNE / PCA / 热力图
python evaluation/hyperparameter_analysis.py  # 超参数消融
python evaluation/corpus_scaling.py      # 语料规模实验
```

### 中文训练

```bash
# 需要先下载中文维基百科 dump 到 data/ 目录
# 然后运行（自动提取 + jieba 分词 + Gensim 训练）
python train_chinese_wiki.py
```

### 评估结果

所有结果保存到 `evaluation/results/`：
- `quality_eval.json` - 词相似度和类比推理详细结果
- `hyperparameter_analysis.png` - 超参数敏感性图表
- `corpus_scaling.png` - 语料规模影响图表
- `*_tsne.png` / `*_pca.png` / `*_heatmap.png` - 可视化图表

---

## 常见问题 FAQ

### Q: 训练太慢？

```
- 减少数据量: max_sentences=500 (快速验证)
- 减小词表:   min_count=20
- 减小维度:   embedding_dim=50
- 加大 batch: batch_size=1024
```

### Q: loss 不下降？

```
- 检查学习率 (太大震荡，太小不动)
- 检查数据是否正确加载 (print 几条看看)
- 先用小数据集 (max_sentences=100) 跑通
```

### Q: 为什么取 W_in 作为词向量？

```
- 论文惯例取 W_in (输入矩阵)
- 也可以取 (W_in + W_out) / 2
- 实验中可对比两种方式
```

### Q: 评估时词不在词表中？

```
- text8 是英文小写，确保查询的词是小写
- min_count 太大会过滤掉很多词，调小试试
```

---

## 参考文献

1. Mikolov et al., *Efficient Estimation of Word Representations in Vector Space*, 2013
2. Mikolov et al., *Distributed Representations of Words and Phrases and their Compositionality*, 2013
3. CS224N Lecture 1-2: Word Vectors (Stanford)
4. Jay Alammar, *The Illustrated Word2Vec*: https://jalammar.github.io/illustrated-word2vec/
