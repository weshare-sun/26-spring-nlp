"""
训练入口: PyTorch 版 Word2Vec
用法: cd myWord2vec && python -m torch_version.train
"""
import sys
import os
import json
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch_version.model import SkipGramNegSampling
from torch_version.preprocess import prepare_training_data
from common.download_data import download_text8


def train(
    embedding_dim=100,
    window_size=5,
    num_negatives=5,
    batch_size=512,
    num_epochs=5,
    learning_rate=0.001,
    min_count=5,
    max_sentences=5000,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[torch/train] 设备: {device}')

    # ===== 数据 =====
    data_path = download_text8('data')
    centers, contexts, negatives, preprocessor = prepare_training_data(
        data_path=data_path,
        max_sentences=max_sentences,
        window_size=window_size,
        num_negatives=num_negatives,
        min_count=min_count,
    )

    dataset = TensorDataset(centers, contexts, negatives)
    # 多线程加载 + MKL 并行，充分利用 CPU
    import os as _os
    torch.set_num_threads(_os.cpu_count())
    num_workers = min(8, _os.cpu_count())
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                         drop_last=True, num_workers=num_workers)
    print(f'[torch/train] DataLoader: {len(loader)} batches | '
          f'workers={num_workers} | threads={torch.get_num_threads()}')

    # ===== 模型 =====
    model = SkipGramNegSampling(preprocessor.vocab_size, embedding_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # ===== 训练 =====
    out_dir = os.path.join('torch_version', 'output')
    os.makedirs(out_dir, exist_ok=True)

    print('[torch/train] 开始训练 ...')
    t0 = time.time()

    # JSONL 日志
    log_path = os.path.join(out_dir, 'training_log.jsonl')
    log_file = open(log_path, 'w', encoding='utf-8')

    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0

        for batch_c, batch_ctx, batch_neg in loader:
            batch_c   = batch_c.to(device)
            batch_ctx = batch_ctx.to(device)
            batch_neg = batch_neg.to(device)

            loss = model(batch_c, batch_ctx, batch_neg)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if num_batches % 200 == 0:
                elapsed = time.time() - t0
                msg = (f'  Epoch {epoch+1} | Batch {num_batches:>6d} | '
                       f'Loss {loss.item():.4f} | Time {elapsed:.0f}s')
                print(msg)
                log_file.write(json.dumps({
                    'epoch': epoch + 1, 'batch': num_batches,
                    'loss': float(loss.item()),
                    'time': round(elapsed, 1),
                }) + '\n')
                log_file.flush()

        avg_loss = total_loss / max(num_batches, 1)
        print(f'[torch/train] Epoch {epoch+1}/{num_epochs} 完成 | Avg Loss: {avg_loss:.4f}')

        # Epoch 级 checkpoint（防止中断丢失进度）
        ckpt = {
            'epoch': epoch + 1,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'total_time': time.time() - t0,
        }
        torch.save(ckpt, os.path.join(out_dir, 'checkpoint.pt'))

    total_time = time.time() - t0
    print(f'[torch/train] 训练完成! 总耗时: {total_time:.1f}s')
    log_file.close()

    # ===== 保存 =====
    embeddings = model.center_embeddings.weight.detach().cpu().numpy()
    np.save(os.path.join(out_dir, 'word_vectors.npy'), embeddings)

    with open(os.path.join(out_dir, 'vocab.json'), 'w', encoding='utf-8') as f:
        json.dump(preprocessor.word2idx, f, ensure_ascii=False)

    print(f'[torch/train] 词向量已保存到 {out_dir}/')
    return model, preprocessor


if __name__ == '__main__':
    train()
