"""
评估脚本: Gensim 版
用法: cd myWord2vec && python -m gensim_version.evaluate
"""
import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from numpy_version.evaluate import find_similar, analogy, cosine_similarity_matrix


def main():
    print('=' * 50)
    print('Gensim 版 Word2Vec 评估')
    print('=' * 50)

    # 方式一: 用保存的 numpy 格式 (与 numpy/torch 版统一)
    output_dir = 'gensim_version/output'
    embeddings = np.load(os.path.join(output_dir, 'word_vectors.npy'))
    with open(os.path.join(output_dir, 'vocab.json'), 'r', encoding='utf-8') as f:
        word2idx = json.load(f)
    idx2word = {v: k for k, v in word2idx.items()}
    print(f'词表大小: {len(word2idx)}, 向量维度: {embeddings.shape[1]}')

    # 方式二 (可选): 直接用 gensim 的 API
    # from gensim.models import Word2Vec
    # model = Word2Vec.load(os.path.join(output_dir, 'gensim_model.bin'))
    # print(model.wv.most_similar('king', topn=5))

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

    # ---- 对比 Gensim 原生 API ----
    print('\n--- Gensim 原生 API 结果 ---')
    from gensim.models import Word2Vec
    gensim_model = Word2Vec.load(os.path.join(output_dir, 'gensim_model.bin'))
    print(f'  most_similar("king"): {gensim_model.wv.most_similar("king", topn=5)}')
    print(f'  类比 king-man+woman:  {gensim_model.wv.most_similar(positive=["king","woman"], negative=["man"], topn=3)}')


if __name__ == '__main__':
    main()
