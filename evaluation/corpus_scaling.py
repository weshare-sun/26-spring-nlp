"""
语料规模实验
- 不同比例的 text8 语料对词向量质量的影响
评估复用 quality_eval 的完整测试集（35 对相似度 + 77 组类比）
"""
import sys
import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.quality_eval import evaluate_word_similarity, evaluate_word_analogy

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')


def run_scaling_experiment():
    """语料规模实验"""
    print('='*50)
    print('语料规模实验')
    print('='*50)

    from gensim.models import Word2Vec
    from common.download_data import download_text8

    data_path = download_text8('data')
    with open(data_path, 'r', encoding='utf-8') as f:
        words = f.read().strip().split()

    total_words = len(words)
    print(f'总词数: {total_words}')

    proportions = [0.1, 0.3, 0.5, 1.0]
    results = []

    for pct in proportions:
        n_words = int(total_words * pct)
        subset = words[:n_words]

        # 分句
        sentences = []
        chunk_size = 1000
        for i in range(0, len(subset), chunk_size):
            chunk = subset[i:i + chunk_size]
            if chunk:
                sentences.append(chunk)

        print(f'\n--- {int(pct*100)}% 语料 ({n_words} 词, {len(sentences)} 句) ---')

        t0 = time.time()
        model = Word2Vec(
            sentences=sentences,
            vector_size=100,
            window=5,
            min_count=5,
            sg=1,
            negative=5,
            epochs=5,
            workers=os.cpu_count(),
        )
        train_time = time.time() - t0

        word2idx = {w: i for i, w in enumerate(model.wv.index_to_key)}
        embeddings = np.zeros((len(word2idx), 100), dtype=np.float32)
        for w, i in word2idx.items():
            embeddings[i] = model.wv[w]

        _, correlation = evaluate_word_similarity(embeddings, word2idx)
        _, analogy_acc = evaluate_word_analogy(embeddings, word2idx)

        result = {
            'proportion': pct,
            'n_words': n_words,
            'n_sentences': len(sentences),
            'vocab_size': len(word2idx),
            'train_time': round(train_time, 1),
            'similarity_correlation': round(correlation, 4) if correlation else None,
            'analogy_accuracy': round(analogy_acc, 4),
        }
        results.append(result)
        corr_str = f'{correlation:.4f}' if correlation else 'N/A'
        print(f'  词表: {len(word2idx)}, 相似度: {corr_str}, 类比: {analogy_acc:.2%}, 时间: {train_time:.1f}s')

    # 保存
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, 'corpus_scaling.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 绘图
    plot_scaling_results(results)

    print('\n[语料规模] 实验完成!')
    return results


def plot_scaling_results(results):
    """绘制语料规模实验图表"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    pct_labels = [f'{int(r["proportion"]*100)}%' for r in results]
    x = range(len(results))  # 等距 x 坐标
    vocabs = [r['vocab_size'] for r in results]
    corrs = [r['similarity_correlation'] for r in results]
    accs = [r['analogy_accuracy'] * 100 for r in results]
    times = [r['train_time'] for r in results]

    # 词表大小
    axes[0].bar(x, vocabs, color='#4ECDC4', alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(pct_labels)
    axes[0].set_xlabel('语料比例')
    axes[0].set_ylabel('词表大小')
    axes[0].set_title('词表大小 vs 语料规模')
    for i, v in enumerate(vocabs):
        axes[0].text(i, v + 200, str(v), ha='center', fontsize=9)
    axes[0].grid(True, alpha=0.3, axis='y')

    # 质量指标
    ax2 = axes[1]
    ax2.plot(x, corrs, 'o-', color='#FF6B6B', linewidth=2, markersize=8, label='相似度相关')
    ax2.set_xticks(x)
    ax2.set_xticklabels(pct_labels)
    ax2.set_xlabel('语料比例')
    ax2.set_ylabel('Spearman 相关系数', color='#FF6B6B')
    ax2.tick_params(axis='y', labelcolor='#FF6B6B')
    ax2_twin = ax2.twinx()
    ax2_twin.plot(x, accs, 's--', color='#4ECDC4', linewidth=2, markersize=8, label='类比准确率')
    ax2_twin.set_ylabel('类比准确率 (%)', color='#4ECDC4')
    ax2_twin.tick_params(axis='y', labelcolor='#4ECDC4')
    ax2.set_title('词向量质量 vs 语料规模')
    ax2.grid(True, alpha=0.3)

    # 训练时间
    axes[2].bar(x, times, color='#FF6B6B', alpha=0.8)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(pct_labels)
    axes[2].set_xlabel('语料比例')
    axes[2].set_ylabel('训练时间 (s)')
    axes[2].set_title('训练时间 vs 语料规模')
    for i, t in enumerate(times):
        axes[2].text(i, t + 0.5, f'{t:.1f}s', ha='center', fontsize=9)
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'corpus_scaling.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('[语料规模] 图表已保存')


if __name__ == '__main__':
    run_scaling_experiment()
