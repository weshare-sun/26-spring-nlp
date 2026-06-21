"""
评估脚本: 加载训练好的词向量，做相似词 / 词类比测试
用法: cd myWord2vec && python -m numpy_version.evaluate
"""
import os
import json
import numpy as np


def load_model(output_dir='numpy_version/output'):
    embeddings = np.load(os.path.join(output_dir, 'word_vectors.npy'))
    with open(os.path.join(output_dir, 'vocab.json'), 'r', encoding='utf-8') as f:
        word2idx = json.load(f)
    idx2word = {v: k for k, v in word2idx.items()}
    return embeddings, word2idx, idx2word


def cosine_similarity_matrix(vec, mat):
    """手动计算余弦相似度，避免依赖 sklearn"""
    dot = mat @ vec
    norm_mat = np.linalg.norm(mat, axis=1)
    norm_vec = np.linalg.norm(vec)
    denom = norm_mat * norm_vec + 1e-10
    return dot / denom


def find_similar(word, embeddings, word2idx, idx2word, top_k=10):
    """找最相似的词"""
    if word not in word2idx:
        print(f"  '{word}' 不在词表中")
        return []

    vec = embeddings[word2idx[word]]
    sims = cosine_similarity_matrix(vec, embeddings)
    top_indices = np.argsort(sims)[::-1][1:top_k + 1]

    return [(idx2word[i], sims[i]) for i in top_indices]


def analogy(a, b, c, embeddings, word2idx, idx2word, top_k=5):
    """
    词类比: a - b + c = ?
    例: king - man + woman = queen
    """
    if any(w not in word2idx for w in [a, b, c]):
        missing = [w for w in [a, b, c] if w not in word2idx]
        print(f"  不在词表中: {missing}")
        return []

    vec = embeddings[word2idx[a]] - embeddings[word2idx[b]] + embeddings[word2idx[c]]
    sims = cosine_similarity_matrix(vec, embeddings)

    exclude = {word2idx[a], word2idx[b], word2idx[c]}
    results = []
    for i in np.argsort(sims)[::-1]:
        if i not in exclude:
            results.append((idx2word[i], sims[i]))
        if len(results) >= top_k:
            break
    return results


def main():
    print('=' * 50)
    print('NumPy 版 Word2Vec 评估')
    print('=' * 50)

    embeddings, word2idx, idx2word = load_model()
    print(f'词表大小: {len(word2idx)}, 向量维度: {embeddings.shape[1]}')

    # ---- 相似词测试 ----
    print('\n--- 相似词测试 ---')
    for word in ['king', 'computer', 'china', 'dog', 'university']:
        results = find_similar(word, embeddings, word2idx, idx2word)
        if results:
            print(f'\n  与 "{word}" 最相似的词:')
            for w, s in results:
                print(f'    {w:15s} {s:.4f}')

    # ---- 词类比测试 ----
    print('\n--- 词类比测试 ---')
    test_cases = [
        ('king', 'man', 'woman'),
        ('paris', 'france', 'germany'),
        ('bigger', 'big', 'small'),
    ]
    for a, b, c in test_cases:
        results = analogy(a, b, c, embeddings, word2idx, idx2word)
        print(f'\n  {a} - {b} + {c} = ?')
        for w, s in results:
            print(f'    {w:15s} {s:.4f}')


if __name__ == '__main__':
    main()
