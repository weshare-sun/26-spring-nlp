"""
数据预处理：构建词表、生成训练样本对
"""
import numpy as np
from collections import Counter


class TextPreprocessor:
    """文本预处理：构建词表、生成训练样本"""

    def __init__(self, min_count=5):
        self.min_count = min_count
        self.word2idx = {}
        self.idx2word = {}
        self.word_freq = {}
        self.vocab_size = 0

    def build_vocab(self, sentences):
        """
        构建词表
        Args:
            sentences: list of list of str, 如 [['i', 'love', 'nlp'], ...]
        """
        word_counts = Counter()
        for sent in sentences:
            word_counts.update(sent)

        words = [w for w, c in word_counts.items() if c >= self.min_count]

        self.word2idx = {w: i for i, w in enumerate(words)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        self.word_freq = {w: word_counts[w] for w in words}
        self.vocab_size = len(words)

        print(f'[preprocess] 词表大小: {self.vocab_size}')
        return self

    def get_training_pairs(self, sentences, window_size=5, mode='skipgram'):
        """
        生成训练样本对
        Args:
            window_size: 上下文窗口大小
            mode: 'skipgram' 或 'cbow'
        Returns:
            pairs: list of (center_idx, context_idx)
        """
        pairs = []
        for sent in sentences:
            indices = [self.word2idx[w] for w in sent if w in self.word2idx]
            for i, center in enumerate(indices):
                actual_window = np.random.randint(1, window_size + 1)
                start = max(0, i - actual_window)
                end = min(len(indices), i + actual_window + 1)
                for j in range(start, end):
                    if j != i:
                        if mode == 'skipgram':
                            pairs.append((center, indices[j]))
                        else:
                            pairs.append((indices[j], center))
        return pairs


def load_text8(path='data/text8', max_sentences=None, chunk_size=1000):
    """
    加载 text8 数据集
    Returns: list of list of str
    """
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()

    words = text.strip().split()
    print(f'[preprocess] 总词数: {len(words)}')

    sentences = []
    for i in range(0, len(words), chunk_size):
        chunk = words[i:i + chunk_size]
        if chunk:
            sentences.append(chunk)
        if max_sentences and len(sentences) >= max_sentences:
            break

    print(f'[preprocess] 句子数: {len(sentences)}')
    return sentences
