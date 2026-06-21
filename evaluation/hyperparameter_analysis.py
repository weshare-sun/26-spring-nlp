"""
超参数敏感性分析
- 向量维度 (embedding_dim)
- 窗口大小 (window_size)
- 负采样数 (negative)
评估复用 quality_eval 的完整测试集（35 对相似度 + 77 组类比）
"""
import sys
import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 复用 quality_eval 的完整评估函数（避免样本不足导致统计无效）
from evaluation.quality_eval import evaluate_word_similarity, evaluate_word_analogy

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')


def train_with_params(embedding_dim=100, window_size=5, num_epochs=5, negative=5, max_sentences=5000):
    """使用指定参数训练 Gensim 模型"""
    from gensim.models import Word2Vec
    from common.download_data import download_text8

    data_path = download_text8('data')

    # 加载数据
    with open(data_path, 'r', encoding='utf-8') as f:
        words = f.read().strip().split()

    sentences = []
    chunk_size = 1000
    for i in range(0, len(words), chunk_size):
        chunk = words[i:i + chunk_size]
        if chunk:
            sentences.append(chunk)
        if max_sentences and len(sentences) >= max_sentences:
            break

    # 训练
    t0 = time.time()
    model = Word2Vec(
        sentences=sentences,
        vector_size=embedding_dim,
        window=window_size,
        min_count=5,
        sg=1,
        negative=negative,
        epochs=num_epochs,
        workers=os.cpu_count(),
    )
    train_time = time.time() - t0

    # 获取词向量
    word2idx = {w: i for i, w in enumerate(model.wv.index_to_key)}
    embeddings = np.zeros((len(word2idx), embedding_dim), dtype=np.float32)
    for w, i in word2idx.items():
        embeddings[i] = model.wv[w]

    return model, word2idx, embeddings, train_time


def evaluate_model(embeddings, word2idx):
    """使用 quality_eval 的完整测试集评估（35 对相似度 + 77 组类比）"""
    _, correlation = evaluate_word_similarity(embeddings, word2idx)
    _, accuracy = evaluate_word_analogy(embeddings, word2idx)
    return correlation, accuracy


def analyze_embedding_dim():
    """分析向量维度的影响"""
    print('\n[超参数] 分析向量维度...')
    dims = [50, 100, 200]
    results = []

    for dim in dims:
        print(f'  训练 dim={dim}...')
        _, word2idx, embeddings, train_time = train_with_params(embedding_dim=dim)
        corr, acc = evaluate_model(embeddings, word2idx)
        results.append({
            'dim': dim,
            'similarity_correlation': round(corr, 4) if corr else None,
            'analogy_accuracy': round(acc, 4),
            'train_time': round(train_time, 1),
            'vocab_size': len(word2idx),
        })
        corr_str = f'{corr:.4f}' if corr else 'N/A'
        print(f'    相似度: {corr_str}, 类比: {acc:.2%}, 时间: {train_time:.1f}s')

    return results


def analyze_window_size():
    """分析窗口大小的影响"""
    print('\n[超参数] 分析窗口大小...')
    windows = [3, 5, 10]
    results = []

    for w in windows:
        print(f'  训练 window={w}...')
        _, word2idx, embeddings, train_time = train_with_params(window_size=w)
        corr, acc = evaluate_model(embeddings, word2idx)
        results.append({
            'window': w,
            'similarity_correlation': round(corr, 4) if corr else None,
            'analogy_accuracy': round(acc, 4),
            'train_time': round(train_time, 1),
        })
        corr_str = f'{corr:.4f}' if corr else 'N/A'
        print(f'    相似度: {corr_str}, 类比: {acc:.2%}, 时间: {train_time:.1f}s')

    return results


def analyze_negative_sampling():
    """分析负采样数的影响"""
    print('\n[超参数] 分析负采样数...')
    negatives = [3, 5, 10]
    results = []

    for neg in negatives:
        print(f'  训练 negative={neg}...')
        _, word2idx, embeddings, train_time = train_with_params(negative=neg)
        corr, acc = evaluate_model(embeddings, word2idx)
        results.append({
            'negative': neg,
            'similarity_correlation': round(corr, 4) if corr else None,
            'analogy_accuracy': round(acc, 4),
            'train_time': round(train_time, 1),
        })
        corr_str = f'{corr:.4f}' if corr else 'N/A'
        print(f'    相似度: {corr_str}, 类比: {acc:.2%}, 时间: {train_time:.1f}s')

    return results


def plot_hyperparameter_results(dim_results, window_results, neg_results):
    """绘制超参数分析图表（含相似度相关 + 类比准确率 + 训练时间）"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    def _plot_ablation(ax, values, labels, corrs, accs, times, title, xlabel):
        x = range(len(values))
        ax.plot(x, corrs, 'o-', color='#FF6B6B', linewidth=2, markersize=10, label='相似度相关')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Spearman 相关系数', color='#FF6B6B')
        ax.tick_params(axis='y', labelcolor='#FF6B6B')
        ax_twin = ax.twinx()
        ax_twin.plot(x, times, 's--', color='#4ECDC4', linewidth=2, markersize=8, alpha=0.5, label='训练时间')
        ax_twin.set_ylabel('训练时间 (s)', color='#4ECDC4')
        ax_twin.tick_params(axis='y', labelcolor='#4ECDC4')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        for i, (c, a) in enumerate(zip(corrs, accs)):
            ax.annotate(f'类比{a:.0f}%', (i, c), textcoords="offset points",
                        xytext=(10, -12 if i == 1 else 10), fontsize=8,
                        color='#E67E22', fontweight='bold')

    _plot_ablation(axes[0],
                   [r['dim'] for r in dim_results],
                   [str(r['dim']) for r in dim_results],
                   [r['similarity_correlation'] for r in dim_results],
                   [r['analogy_accuracy'] * 100 for r in dim_results],
                   [r['train_time'] for r in dim_results],
                   '向量维度的影响', '向量维度')

    _plot_ablation(axes[1],
                   [r['window'] for r in window_results],
                   [str(r['window']) for r in window_results],
                   [r['similarity_correlation'] for r in window_results],
                   [r['analogy_accuracy'] * 100 for r in window_results],
                   [r['train_time'] for r in window_results],
                   '窗口大小的影响', '窗口大小')

    _plot_ablation(axes[2],
                   [r['negative'] for r in neg_results],
                   [str(r['negative']) for r in neg_results],
                   [r['similarity_correlation'] for r in neg_results],
                   [r['analogy_accuracy'] * 100 for r in neg_results],
                   [r['train_time'] for r in neg_results],
                   '负采样数的影响', '负采样数')

    plt.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(RESULTS_DIR, 'hyperparameter_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('[超参数] 图表已保存')


def run_hyperparameter_analysis():
    """运行超参数分析"""
    print('='*50)
    print('超参数敏感性分析（完整数据）')
    print('='*50)

    dim_results = analyze_embedding_dim()
    window_results = analyze_window_size()
    neg_results = analyze_negative_sampling()

    plot_hyperparameter_results(dim_results, window_results, neg_results)

    # 保存结果
    all_results = {
        'embedding_dim': dim_results,
        'window_size': window_results,
        'negative_sampling': neg_results,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, 'hyperparameter_analysis.json'), 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # 打印汇总表
    print('\n' + '='*60)
    print('超参数分析汇总')
    print('='*60)
    print(f'\n{"向量维度":<10} {"相似度相关":<12} {"类比准确率":<10} {"训练时间":<10}')
    print('-'*42)
    for r in dim_results:
        corr = f'{r["similarity_correlation"]:.4f}' if r['similarity_correlation'] else 'N/A'
        print(f'{r["dim"]:<10} {corr:<12} {r["analogy_accuracy"]:.2%}    {r["train_time"]:.1f}s')

    print(f'\n{"窗口大小":<10} {"相似度相关":<12} {"类比准确率":<10} {"训练时间":<10}')
    print('-'*42)
    for r in window_results:
        corr = f'{r["similarity_correlation"]:.4f}' if r['similarity_correlation'] else 'N/A'
        print(f'{r["window"]:<10} {corr:<12} {r["analogy_accuracy"]:.2%}    {r["train_time"]:.1f}s')

    print(f'\n{"负采样数":<10} {"相似度相关":<12} {"类比准确率":<10} {"训练时间":<10}')
    print('-'*42)
    for r in neg_results:
        corr = f'{r["similarity_correlation"]:.4f}' if r['similarity_correlation'] else 'N/A'
        print(f'{r["negative"]:<10} {corr:<12} {r["analogy_accuracy"]:.2%}    {r["train_time"]:.1f}s')

    print('\n[超参数] 分析完成!')
    return all_results


if __name__ == '__main__':
    run_hyperparameter_analysis()
