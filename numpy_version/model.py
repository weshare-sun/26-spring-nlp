"""
Word2Vec 纯 NumPy 实现 —— Skip-gram + Negative Sampling
"""
import numpy as np


class Word2Vec:
    """
    Skip-gram + Negative Sampling
    损失函数: L = -log σ(u_o^T v_c) - Σ E[log σ(-u_k^T v_c)]
    """

    def __init__(self, vocab_size, embedding_dim=100):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # Xavier 初始化
        scale = np.sqrt(2.0 / (vocab_size + embedding_dim))
        self.W_in  = np.random.randn(vocab_size, embedding_dim) * scale   # 输入矩阵 V×D
        self.W_out = np.random.randn(vocab_size, embedding_dim) * scale   # 输出矩阵 V×D

    def forward(self, center_idx, context_idx, neg_indices):
        """
        前向传播，计算 loss
        Args:
            center_idx:   (B,)     中心词索引
            context_idx:  (B,)     正样本上下文词索引
            neg_indices:  (B, K)   负样本索引
        Returns:
            loss: 标量
        """
        B = len(center_idx)

        # (B, D) 中心词向量
        v_c = self.W_in[center_idx]

        # ---------- 正样本 ----------
        # (B, D) 正样本上下文词向量
        u_o = self.W_out[context_idx]
        # (B,) 正样本得分
        pos_score = np.sum(v_c * u_o, axis=1)
        # 标量: -mean(log σ(score))
        pos_loss = -np.mean(np.log(self._sigmoid(pos_score) + 1e-10))

        # ---------- 负样本 ----------
        # (B, K, D) 负样本词向量
        u_k = self.W_out[neg_indices]
        # (B, K) 负样本得分: 对每个 (b,k) 做 v_c[b] · u_k[b,k]
        neg_score = np.einsum('bd,bkd->bk', v_c, u_k)
        # 标量: -mean(Σ_k log σ(-score))
        neg_loss = -np.mean(np.sum(np.log(self._sigmoid(-neg_score) + 1e-10), axis=1))

        loss = pos_loss + neg_loss
        return loss

    def backward(self, center_idx, context_idx, neg_indices, lr=0.025):
        """
        反向传播 + SGD 更新
        """
        B = len(center_idx)

        # (B, D)
        v_c = self.W_in[center_idx]

        # ---------- 正样本梯度 ----------
        u_o = self.W_out[context_idx]                          # (B, D)
        pos_score = np.sum(v_c * u_o, axis=1)                  # (B,)
        pos_g = self._sigmoid(pos_score) - 1                    # (B,)  dL/d(score)

        # dL/d(u_o) = pos_g * v_c      → (B, D)
        grad_u_pos = pos_g[:, np.newaxis] * v_c
        # dL/d(v_c) = pos_g * u_o      → (B, D)
        grad_v_pos = pos_g[:, np.newaxis] * u_o

        # ---------- 负样本梯度 ----------
        u_k = self.W_out[neg_indices]                          # (B, K, D)
        neg_score = np.einsum('bd,bkd->bk', v_c, u_k)          # (B, K)
        neg_g = self._sigmoid(neg_score)                        # (B, K)  dL/d(score)

        # dL/d(u_k) = neg_g * v_c      → (B, K, D)
        grad_u_neg = neg_g[:, :, np.newaxis] * v_c[:, np.newaxis, :]
        # dL/d(v_c) += Σ_k neg_g * u_k → (B, D)
        grad_v_neg = np.einsum('bk,bkd->bd', neg_g, u_k)

        # ---------- 更新 W_out ----------
        # 正样本: W_out[context_idx] -= lr * grad_u_pos
        for b in range(B):
            self.W_out[context_idx[b]] -= lr * grad_u_pos[b]

        # 负样本: W_out[neg_indices[b,k]] -= lr * grad_u_neg[b,k]
        for b in range(B):
            for k in range(neg_indices.shape[1]):
                self.W_out[neg_indices[b, k]] -= lr * grad_u_neg[b, k]

        # ---------- 更新 W_in ----------
        grad_v = grad_v_pos + grad_v_neg  # (B, D)
        for b in range(B):
            self.W_in[center_idx[b]] -= lr * grad_v[b]

    def get_embedding(self, word_idx):
        """获取单个词向量"""
        return self.W_in[word_idx]

    def get_all_embeddings(self):
        """获取全部词向量"""
        return self.W_in.copy()

    @staticmethod
    def _sigmoid(x):
        x = np.clip(x, -10, 10)
        return 1.0 / (1.0 + np.exp(-x))


class NegativeSampler:
    """
    负采样器: P(w) ∝ f(w)^0.75
    """

    def __init__(self, word_freq, word2idx, num_negatives=5):
        self.num_negatives = num_negatives
        self.vocab_size = len(word2idx)

        freqs = np.array([word_freq.get(w, 0) for w in word2idx.keys()])
        powered = np.power(freqs.astype(np.float64), 0.75)
        self.distribution = powered / powered.sum()

    def sample(self, batch_size):
        """
        生成负样本索引
        Returns: (batch_size, num_negatives) int array
        """
        return np.random.choice(
            self.vocab_size,
            size=(batch_size, self.num_negatives),
            p=self.distribution
        )
