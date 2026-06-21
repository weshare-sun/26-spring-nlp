"""
数据预处理: PyTorch 版
复用 numpy_version 的预处理逻辑，额外转为 Tensor
"""
import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from numpy_version.preprocess import TextPreprocessor, load_text8  # 复用


def prepare_training_data(
    data_path='data/text8',
    max_sentences=5000,
    window_size=5,
    num_negatives=5,
    min_count=5,
):
    """
    完整的数据准备流程
    Returns:
        centers:   LongTensor (N,)
        contexts:  LongTensor (N,)
        negatives: LongTensor (N, K)
        preprocessor: TextPreprocessor (用于评估时查词)
    """
    sentences = load_text8(data_path, max_sentences=max_sentences)

    preprocessor = TextPreprocessor(min_count=min_count)
    preprocessor.build_vocab(sentences)

    pairs = preprocessor.get_training_pairs(sentences, window_size=window_size, mode='skipgram')
    print(f'[torch/preprocess] 训练样本数: {len(pairs)}')

    centers  = np.array([p[0] for p in pairs])
    contexts = np.array([p[1] for p in pairs])

    # 一次性生成所有负样本
    from numpy_version.model import NegativeSampler
    sampler = NegativeSampler(preprocessor.word_freq, preprocessor.word2idx, num_negatives)
    negatives = sampler.sample(len(pairs))

    centers   = torch.from_numpy(centers).long()
    contexts  = torch.from_numpy(contexts).long()
    negatives = torch.from_numpy(negatives).long()

    return centers, contexts, negatives, preprocessor
