"""
词向量可视化
- t-SNE 降维
- PCA 降维
- 词向量空间分布
"""
import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

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


def get_word_groups(word2idx):
    """定义词组用于可视化"""
    # 判断是中文还是英文词表
    is_chinese = any('一' <= c <= '鿿' for c in list(word2idx.keys())[:10])

    if is_chinese:
        groups = {
            '学习': ['学习', '教育', '教学', '知识'],
            '技术': ['技术', '科学', '研究', '实验'],
            '计算机': ['计算机', '人工智能', '机器学习', '深度学习'],
            '自然': ['自然', '环境', '生态', '气候'],
            '社会': ['社会', '政治', '经济', '文化'],
            '健康': ['健康', '医学', '治疗', '疾病'],
        }
    else:
        groups = {
            '王室': ['king', 'queen', 'prince', 'princess'],
            '国家': ['france', 'germany', 'japan', 'china'],
            '城市': ['paris', 'berlin', 'tokyo', 'beijing'],
            '动物': ['dog', 'cat', 'bird', 'fish'],
            '技术': ['computer', 'software', 'hardware', 'keyboard'],
            '学校': ['university', 'college', 'school', 'student'],
        }

    # 过滤出存在的词
    valid_groups = {}
    for group_name, words in groups.items():
        valid_words = [w for w in words if w in word2idx]
        if len(valid_words) >= 2:
            valid_groups[group_name] = valid_words

    return valid_groups


def visualize_tsne(embeddings, word2idx, title='t-SNE Visualization', save_path=None):
    """t-SNE 可视化"""
    groups = get_word_groups(word2idx)

    # 收集要显示的词
    words_to_show = []
    word_labels = []
    word_colors = []
    color_map = plt.cm.Set2(np.linspace(0, 1, len(groups)))

    for i, (group_name, words) in enumerate(groups.items()):
        for w in words:
            words_to_show.append(word2idx[w])
            word_labels.append(w)
            word_colors.append(color_map[i])

    if len(words_to_show) < 3:
        print('[可视化] 词数不足，跳过')
        return

    # 提取向量
    vectors = embeddings[words_to_show]

    # t-SNE 降维
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(words_to_show)-1))
    coords = tsne.fit_transform(vectors)

    # 绘图
    fig, ax = plt.subplots(figsize=(12, 10))

    for i, (x, y) in enumerate(coords):
        ax.scatter(x, y, c=[word_colors[i]], s=100, alpha=0.7)
        ax.annotate(word_labels[i], (x, y), fontsize=9,
                   ha='center', va='bottom')

    # 添加图例
    legend_elements = [plt.scatter([], [], c=[color_map[i]], s=100, label=group_name)
                      for i, group_name in enumerate(groups.keys())]
    ax.legend(handles=legend_elements, loc='upper right')

    ax.set_title(title)
    ax.set_xlabel('t-SNE 维度 1')
    ax.set_ylabel('t-SNE 维度 2')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'[可视化] t-SNE 图已保存: {save_path}')
    else:
        plt.show()


def visualize_pca(embeddings, word2idx, title='PCA Visualization', save_path=None):
    """PCA 可视化"""
    groups = get_word_groups(word2idx)

    words_to_show = []
    word_labels = []
    word_colors = []
    color_map = plt.cm.Set2(np.linspace(0, 1, len(groups)))

    for i, (group_name, words) in enumerate(groups.items()):
        for w in words:
            words_to_show.append(word2idx[w])
            word_labels.append(w)
            word_colors.append(color_map[i])

    if len(words_to_show) < 2:
        print('[可视化] 词数不足，跳过')
        return

    vectors = embeddings[words_to_show]

    # PCA 降维
    pca = PCA(n_components=2)
    coords = pca.fit_transform(vectors)

    # 绘图
    fig, ax = plt.subplots(figsize=(12, 10))

    for i, (x, y) in enumerate(coords):
        ax.scatter(x, y, c=[word_colors[i]], s=100, alpha=0.7)
        ax.annotate(word_labels[i], (x, y), fontsize=9,
                   ha='center', va='bottom')

    legend_elements = [plt.scatter([], [], c=[color_map[i]], s=100, label=group_name)
                      for i, group_name in enumerate(groups.keys())]
    ax.legend(handles=legend_elements, loc='upper right')

    ax.set_title(title)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} 方差)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} 方差)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'[可视化] PCA 图已保存: {save_path}')
    else:
        plt.show()


def visualize_similarity_heatmap(embeddings, word2idx, words, save_path=None):
    """相似度热力图"""
    # 过滤存在的词
    valid_words = [w for w in words if w in word2idx]
    if len(valid_words) < 2:
        print('[可视化] 词数不足')
        return

    n = len(valid_words)
    sim_matrix = np.zeros((n, n))

    for i, w1 in enumerate(valid_words):
        for j, w2 in enumerate(valid_words):
            v1 = embeddings[word2idx[w1]]
            v2 = embeddings[word2idx[w2]]
            sim_matrix[i, j] = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(sim_matrix, cmap='YlOrRd', vmin=0, vmax=1)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(valid_words, rotation=45, ha='right')
    ax.set_yticklabels(valid_words)

    # 添加数值
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f'{sim_matrix[i, j]:.2f}',
                   ha='center', va='center', fontsize=8)

    plt.colorbar(im)
    ax.set_title('词向量相似度热力图')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'[可视化] 热力图已保存: {save_path}')
    else:
        plt.show()


def run_visualization():
    """运行可视化"""
    print('='*50)
    print('词向量可视化')
    print('='*50)

    versions = {
        'Gensim': 'gensim_version/output',
        'Chinese': 'chinese_wiki_full_output',
    }

    for name, out_dir in versions.items():
        print(f'\n--- {name} 版本 ---')

        word2idx, embeddings = load_vocab_and_embeddings(out_dir)
        if word2idx is None:
            print(f'  跳过（输出文件不存在）')
            continue

        # t-SNE 可视化
        visualize_tsne(
            embeddings, word2idx,
            title=f'{name} - t-SNE 词向量可视化',
            save_path=os.path.join(RESULTS_DIR, f'{name.lower()}_tsne.png')
        )

        # PCA 可视化
        visualize_pca(
            embeddings, word2idx,
            title=f'{name} - PCA 词向量可视化',
            save_path=os.path.join(RESULTS_DIR, f'{name.lower()}_pca.png')
        )

        # 相似度热力图
        test_words = ['king', 'queen', 'man', 'woman', 'computer', 'software',
                      'university', 'college', 'dog', 'cat']
        if name == 'Chinese':
            test_words = ['学', '习', '人', '工', '计', '算', '机', '数', '字']

        visualize_similarity_heatmap(
            embeddings, word2idx, test_words,
            save_path=os.path.join(RESULTS_DIR, f'{name.lower()}_heatmap.png')
        )

    print('\n[可视化] 完成!')


if __name__ == '__main__':
    run_visualization()
