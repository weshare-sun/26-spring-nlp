"""
训练入口: NumPy 版 Word2Vec
用法: cd myWord2vec && python -m numpy_version.train
"""
import sys
import os
import json
import time
import numpy as np

# 让 import 能找到项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from numpy_version.preprocess import TextPreprocessor, load_text8
from numpy_version.model import Word2Vec, NegativeSampler
from common.download_data import download_text8


def train(
    embedding_dim=100,
    window_size=5,
    num_negatives=5,
    batch_size=512,
    num_epochs=5,
    learning_rate=0.025,
    min_count=5,
    max_sentences=5000,
):
    # ===== 下载数据 =====
    data_path = download_text8('data')

    # ===== 数据准备 =====
    print('[train] 加载数据 ...')
    sentences = load_text8(data_path, max_sentences=max_sentences)

    print('[train] 构建词表 ...')
    preprocessor = TextPreprocessor(min_count=min_count)
    preprocessor.build_vocab(sentences)

    print('[train] 生成训练对 ...')
    pairs = preprocessor.get_training_pairs(sentences, window_size=window_size, mode='skipgram')
    print(f'[train] 训练样本数: {len(pairs)}')

    # ===== 初始化 =====
    model = Word2Vec(preprocessor.vocab_size, embedding_dim)
    sampler = NegativeSampler(preprocessor.word_freq, preprocessor.word2idx, num_negatives)

    # ===== 训练 =====
    print('[train] 开始训练 ...')
    t0 = time.time()

    # JSONL 日志
    log_path = os.path.join('numpy_version', 'output', 'training_log.jsonl')
    log_file = open(log_path, 'w', encoding='utf-8')

    for epoch in range(num_epochs):
        np.random.shuffle(pairs)
        total_loss = 0.0
        num_batches = 0

        # 学习率线性衰减
        progress = epoch / num_epochs
        lr = learning_rate * (1 - progress) + 0.0001 * progress

        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            if len(batch) < batch_size // 2:
                continue

            center    = np.array([p[0] for p in batch])
            context   = np.array([p[1] for p in batch])
            negatives = sampler.sample(len(batch))

            loss = model.forward(center, context, negatives)
            model.backward(center, context, negatives, lr=lr)

            total_loss += loss
            num_batches += 1

            if num_batches % 200 == 0:
                elapsed = time.time() - t0
                msg = (f'  Epoch {epoch+1} | Batch {num_batches:>6d} | '
                       f'Loss {loss:.4f} | LR {lr:.5f} | Time {elapsed:.0f}s')
                print(msg)
                log_file.write(json.dumps({
                    'epoch': epoch + 1, 'batch': num_batches,
                    'loss': float(loss), 'lr': float(lr),
                    'time': round(elapsed, 1),
                }) + '\n')
                log_file.flush()

        avg_loss = total_loss / max(num_batches, 1)
        print(f'[train] Epoch {epoch+1}/{num_epochs} 完成 | Avg Loss: {avg_loss:.4f}')

    total_time = time.time() - t0
    print(f'[train] 训练完成! 总耗时: {total_time:.1f}s')
    log_file.close()

    # ===== 保存 =====
    out_dir = os.path.join('numpy_version', 'output')
    os.makedirs(out_dir, exist_ok=True)

    embeddings = model.get_all_embeddings()
    np.save(os.path.join(out_dir, 'word_vectors.npy'), embeddings)

    with open(os.path.join(out_dir, 'vocab.json'), 'w', encoding='utf-8') as f:
        json.dump(preprocessor.word2idx, f, ensure_ascii=False)

    print(f'[train] 词向量已保存到 {out_dir}/')
    return model, preprocessor


if __name__ == '__main__':
    train()
