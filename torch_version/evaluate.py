"""
评估脚本: PyTorch 版
用法: cd myWord2vec && python -m torch_version.evaluate
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from numpy_version.evaluate import load_model, find_similar, analogy, cosine_similarity_matrix


def main():
    print('=' * 50)
    print('PyTorch 版 Word2Vec 评估')
    print('=' * 50)

    embeddings, word2idx, idx2word = load_model('torch_version/output')
    print(f'词表大小: {len(word2idx)}, 向量维度: {embeddings.shape[1]}')

    # ---- 相似词 ----
    print('\n--- 相似词测试 ---')
    for word in ['king', 'computer', 'china', 'dog', 'university']:
        results = find_similar(word, embeddings, word2idx, idx2word)
        if results:
            print(f'\n  与 "{word}" 最相似的词:')
            for w, s in results:
                print(f'    {w:15s} {s:.4f}')

    # ---- 词类比 ----
    print('\n--- 词类比测试 ---')
    for a, b, c in [('king', 'man', 'woman'), ('paris', 'france', 'germany')]:
        results = analogy(a, b, c, embeddings, word2idx, idx2word)
        print(f'\n  {a} - {b} + {c} = ?')
        for w, s in results:
            print(f'    {w:15s} {s:.4f}')


if __name__ == '__main__':
    main()
