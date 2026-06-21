"""
训练入口: Gensim 版 Word2Vec (baseline)
用法: cd myWord2vec && python -m gensim_version.train
"""
import sys
import os
import json
import time
import numpy as np
from multiprocessing import cpu_count

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.download_data import download_text8


def train(
    embedding_dim=100,
    window_size=5,
    num_negatives=5,
    num_epochs=5,
    min_count=5,
    max_sentences=5000,
):
    from gensim.models import Word2Vec

    # ===== 数据 =====
    data_path = download_text8('data')

    print('[gensim/train] 加载数据 ...')
    with open(data_path, 'r', encoding='utf-8') as f:
        words = f.read().strip().split()

    sentences = []
    chunk_size = 1000
    for i in range(0, len(words), chunk_size):
        chunk = words[i:i + chunk_size]
        if chunk:
            sentences.append(chunk)
        if max_sentences and len(sentences) >= max_sentences:
            break

    print(f'[gensim/train] 句子数: {len(sentences)}')

    # ===== 训练 =====
    print('[gensim/train] 开始训练 ...')
    t0 = time.time()

    # JSONL 日志（epoch 级，Gensim 不暴露逐 batch loss）
    log_path = os.path.join('gensim_version', 'output', 'training_log.jsonl')
    log_file = open(log_path, 'w', encoding='utf-8')

    from gensim.models.callbacks import CallbackAny2Vec

    class LossCallback(CallbackAny2Vec):
        def __init__(self, log_f, start_time, num_epochs):
            super().__init__()
            self.log_f = log_f
            self.start_time = start_time
            self.num_epochs = num_epochs
            self.epoch = 0

        def on_epoch_end(self, model):
            self.epoch += 1
            elapsed = time.time() - self.start_time
            loss = model.get_latest_training_loss()
            data = {
                'epoch': self.epoch,
                'loss': round(float(loss), 4) if loss else None,
                'time': round(elapsed, 1),
            }
            self.log_f.write(json.dumps(data) + '\n')
            self.log_f.flush()
            loss_str = f'{loss:.4f}' if loss else 'N/A'
            print(f'  Epoch {self.epoch}/{self.num_epochs} | Loss {loss_str} | Time {elapsed:.0f}s')

    callback = LossCallback(log_file, t0, num_epochs)

    model = Word2Vec(
        sentences=sentences,
        vector_size=embedding_dim,
        window=window_size,
        min_count=min_count,
        sg=1,             # 1=Skip-gram
        negative=num_negatives,
        epochs=num_epochs,
        workers=cpu_count(),
        compute_loss=True,
        callbacks=[callback],
    )

    elapsed = time.time() - t0
    log_file.close()
    print(f'[gensim/train] 训练完成! 耗时: {elapsed:.1f}s')

    # ===== 保存 =====
    out_dir = os.path.join('gensim_version', 'output')
    os.makedirs(out_dir, exist_ok=True)

    model.save(os.path.join(out_dir, 'gensim_model.bin'))

    # 同时保存为 numpy 格式，方便统一评估
    word2idx = {w: i for i, w in enumerate(model.wv.index_to_key)}
    embeddings = np.zeros((len(word2idx), embedding_dim), dtype=np.float32)
    for w, i in word2idx.items():
        embeddings[i] = model.wv[w]

    np.save(os.path.join(out_dir, 'word_vectors.npy'), embeddings)
    with open(os.path.join(out_dir, 'vocab.json'), 'w', encoding='utf-8') as f:
        json.dump(word2idx, f, ensure_ascii=False)

    print(f'[gensim/train] 已保存到 {out_dir}/')
    return model


if __name__ == '__main__':
    train()
