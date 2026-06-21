"""
中英文对比分析
- 同一算法在不同语言上的表现
- 词向量空间结构对比
"""
import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')


def load_vocab_and_embeddings(out_dir):
    """加载词表和向量"""
    vocab_path = os.path.join(out_dir, 'vocab.json')
    vec_path = os.path.join(out_dir, 'word_vectors.npy')

    if not os.path.exists(vocab_path) or not os.path.exists(vec_path):
        return None, None

    with open(vocab_path, 'r', encoding='utf-8') as f:
        word2idx = json.load(f)

    embeddings = np.load(vec_path)
    return word2idx, embeddings


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)


def find_similar_words(word, embeddings, word2idx, top_k=5):
    """查找相似词"""
    if word not in word2idx:
        return []
    idx = word2idx[word]
    vec = embeddings[idx]

    sims = np.dot(embeddings, vec) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(vec) + 1e-10)
    sorted_idx = np.argsort(sims)[::-1]

    results = []
    for i in sorted_idx:
        if i != idx:
            w = [k for k, v in word2idx.items() if v == i][0]
            results.append((w, float(sims[i])))
            if len(results) >= top_k:
                break
    return results


def analyze_semantic_clusters(embeddings, word2idx, word_groups):
    """分析语义聚类质量"""
    group_similarities = []

    for group_name, words in word_groups.items():
        valid_words = [w for w in words if w in word2idx]
        if len(valid_words) < 2:
            continue

        # 计算组内相似度
        sims = []
        for i, w1 in enumerate(valid_words):
            for w2 in valid_words[i+1:]:
                sim = cosine_similarity(embeddings[word2idx[w1]], embeddings[word2idx[w2]])
                sims.append(sim)

        if sims:
            group_similarities.append({
                'group': group_name,
                'avg_similarity': np.mean(sims),
                'std': np.std(sims),
                'words': valid_words,
            })

    return group_similarities


def compare_languages():
    """对比中英文词向量"""
    print('='*50)
    print('中英文对比分析')
    print('='*50)

    # 加载英文模型
    en_dir = 'gensim_version/output'
    en_word2idx, en_embeddings = load_vocab_and_embeddings(en_dir)

    # 加载中文模型
    zh_dir = 'chinese_wiki_full_output'
    zh_word2idx, zh_embeddings = load_vocab_and_embeddings(zh_dir)

    if en_word2idx is None:
        print('[对比] 英文模型不存在，请先训练')
        return
    if zh_word2idx is None:
        print('[对比] 中文模型不存在，请先训练')
        return

    print(f'\n英文词表: {len(en_word2idx)} 词')
    print(f'中文词表: {len(zh_word2idx)} 词')

    # ===== 英文分析 =====
    print('\n--- 英文词向量分析 ---')

    en_groups = {
        '王室': ['king', 'queen', 'prince'],
        '国家': ['france', 'germany', 'japan'],
        '动物': ['dog', 'cat', 'bird'],
        '技术': ['computer', 'software', 'keyboard'],
    }

    en_clusters = analyze_semantic_clusters(en_embeddings, en_word2idx, en_groups)
    print('语义聚类质量:')
    for c in en_clusters:
        print(f'  {c["group"]}: 相似度 {c["avg_similarity"]:.3f} (±{c["std"]:.3f})')

    # 英文相似词
    print('\n英文相似词测试:')
    en_test_words = ['king', 'computer', 'university']
    for word in en_test_words:
        similar = find_similar_words(word, en_embeddings, en_word2idx, top_k=3)
        if similar:
            words_str = ', '.join([f'{w}({s:.3f})' for w, s in similar])
            print(f'  {word} → {words_str}')

    # ===== 中文分析 =====
    print('\n--- 中文词向量分析 ---')

    zh_groups = {
        '教育': ['学习', '教育', '教学', '培训'],
        '技术': ['计算机', '软件', '互联网', '程序'],
        '国家': ['中国', '美国', '日本', '德国'],
        '动物': ['狗', '猫', '鸟', '马'],
    }

    zh_clusters = analyze_semantic_clusters(zh_embeddings, zh_word2idx, zh_groups)
    print('语义聚类质量:')
    for c in zh_clusters:
        print(f'  {c["group"]}: 相似度 {c["avg_similarity"]:.3f} (±{c["std"]:.3f})')

    # 中文相似词
    print('\n中文相似词测试:')
    zh_test_words = ['学', '计', '数']
    for word in zh_test_words:
        similar = find_similar_words(word, zh_embeddings, zh_word2idx, top_k=3)
        if similar:
            words_str = ', '.join([f'{w}({s:.3f})' for w, s in similar])
            print(f'  {word} → {words_str}')

    # ===== 对比可视化 =====
    plot_cross_lingual_comparison(en_clusters, zh_clusters)

    # 保存结果
    def convert_to_serializable(obj):
        """Convert numpy types to Python native types"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        return obj

    results = {
        'english': {
            'vocab_size': len(en_word2idx),
            'clusters': convert_to_serializable(en_clusters),
        },
        'chinese': {
            'vocab_size': len(zh_word2idx),
            'clusters': convert_to_serializable(zh_clusters),
        },
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, 'cross_lingual.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print('\n[对比] 分析完成!')


def plot_cross_lingual_comparison(en_clusters, zh_clusters):
    """绘制中英文对比图"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 英文聚类
    ax1 = axes[0]
    if en_clusters:
        groups = [c['group'] for c in en_clusters]
        sims = [c['avg_similarity'] for c in en_clusters]
        stds = [c['std'] for c in en_clusters]
        ax1.bar(groups, sims, yerr=stds, capsize=5, color='#4ECDC4', alpha=0.7)
        ax1.set_ylabel('平均相似度')
        ax1.set_title('英文语义聚类质量')
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3, axis='y')

    # 中文聚类
    ax2 = axes[1]
    if zh_clusters:
        groups = [c['group'] for c in zh_clusters]
        sims = [c['avg_similarity'] for c in zh_clusters]
        stds = [c['std'] for c in zh_clusters]
        ax2.bar(groups, sims, yerr=stds, capsize=5, color='#FF6B6B', alpha=0.7)
        ax2.set_ylabel('平均相似度')
        ax2.set_title('中文语义聚类质量')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(RESULTS_DIR, 'cross_lingual_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('[对比] 对比图已保存')


if __name__ == '__main__':
    compare_languages()
