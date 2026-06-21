"""
Word2Vec PyTorch 实现 —— Skip-gram + Negative Sampling
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipGramNegSampling(nn.Module):
    """
    Skip-gram + Negative Sampling
    L = -log σ(u_o^T v_c) - Σ_k log σ(-u_k^T v_c)
    """

    def __init__(self, vocab_size, embedding_dim=100):
        super().__init__()
        self.center_embeddings  = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)

        nn.init.xavier_uniform_(self.center_embeddings.weight)
        nn.init.xavier_uniform_(self.context_embeddings.weight)

    def forward(self, center, context, negatives):
        """
        Args:
            center:    (B,)     中心词索引
            context:   (B,)     正样本上下文词索引
            negatives: (B, K)   负样本索引
        Returns:
            loss: 标量
        """
        # (B, D) 中心词向量
        v_c = self.center_embeddings(center)

        # 正样本: (B, D)
        u_o = self.context_embeddings(context)
        pos_score = torch.sum(v_c * u_o, dim=1)      # (B,)
        pos_loss  = -F.logsigmoid(pos_score).mean()

        # 负样本: (B, K, D)
        u_k = self.context_embeddings(negatives)
        neg_score = torch.bmm(u_k, v_c.unsqueeze(2)).squeeze(2)  # (B, K)
        neg_loss  = -F.logsigmoid(-neg_score).sum(dim=1).mean()

        return pos_loss + neg_loss
