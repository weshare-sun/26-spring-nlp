"""
中文维基百科 Word2Vec 训练脚本（多进程优化版）
多进程分词 → 保存文件 → corpus_file 并行训练
用法: python train_chinese_wiki.py
"""
import sys
import os
import time
import json
import numpy as np
from multiprocessing import Pool, cpu_count

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def segment_line(line):
    """单行分词（用于多进程）"""
    import jieba
    line = line.strip()
    if not line:
        return None
    words = [w for w in jieba.cut(line) if w.strip()]
    return ' '.join(words) if words else None


def preprocess_and_segment(input_path, output_path, workers=None):
    """多进程预分词"""
    if workers is None:
        workers = cpu_count()
    print(f'[wiki] 多进程分词: {input_path} → {output_path} (workers={workers})', flush=True)
    t0 = time.time()

    print(f'[wiki] 读取文件...', flush=True)
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    print(f'[wiki] 共 {len(lines)} 行，开始并行分词...', flush=True)

    with Pool(workers) as pool:
        results = pool.map(segment_line, lines, chunksize=1000)

    count = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for r in results:
            if r:
                f.write(r + '\n')
                count += 1

    elapsed = time.time() - t0
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f'[wiki] 分词完成: {count} 条, {size_mb:.1f}MB, 耗时 {elapsed:.0f}s', flush=True)
    return output_path


def train_word2vec(corpus_file_path, output_dir='chinese_wiki_output'):
    """用 corpus_file 参数训练（真正多线程并行）"""
    from gensim.models import Word2Vec

    print(f'[wiki] 开始训练（corpus_file 模式）...', flush=True)
    t0 = time.time()

    # JSONL 日志回调
    from gensim.models.callbacks import CallbackAny2Vec
    log_path = os.path.join(output_dir, 'training_log.jsonl')
    os.makedirs(output_dir, exist_ok=True)
    log_file = open(log_path, 'w', encoding='utf-8')

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
            print(f'  Epoch {self.epoch}/5 | Loss {loss_str} | Time {elapsed:.0f}s', flush=True)

    callback = LossCallback(log_file, t0, 5)

    model = Word2Vec(
        corpus_file=corpus_file_path,
        vector_size=100,
        window=5,
        min_count=3,
        sg=1,
        negative=5,
        epochs=5,
        workers=cpu_count(),
        compute_loss=True,
        callbacks=[callback],
    )

    elapsed = time.time() - t0
    log_file.close()
    print(f'[wiki] 训练完成! 耗时: {elapsed:.0f}s ({elapsed/60:.1f}分钟)', flush=True)

    # 保存
    os.makedirs(output_dir, exist_ok=True)
    model.save(os.path.join(output_dir, 'wiki_model.bin'))

    word2idx = {w: i for i, w in enumerate(model.wv.index_to_key)}
    embeddings = model.wv.vectors

    np.save(os.path.join(output_dir, 'word_vectors.npy'), embeddings)
    with open(os.path.join(output_dir, 'vocab.json'), 'w', encoding='utf-8') as f:
        json.dump(word2idx, f, ensure_ascii=False)

    print(f'[wiki] 词表大小: {len(word2idx)}', flush=True)
    print(f'[wiki] 已保存到 {output_dir}/', flush=True)

    # 测试
    print('\n[wiki] 测试相似词:', flush=True)
    test_words = ['中国', '科学', '学习', '计算机', '经济', '教育']
    for word in test_words:
        if word in model.wv:
            similar = model.wv.most_similar(word, topn=5)
            words_str = ', '.join([f'{w}({s:.3f})' for w, s in similar])
            print(f'  {word} → {words_str}', flush=True)
        else:
            print(f'  {word} → [不在词表中]', flush=True)

    return model


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true', help='使用全量语料（默认 10%）')
    args = parser.parse_args()

    print('='*50, flush=True)
    mode = '全量' if args.full else '10% 子集'
    print(f'中文维基百科 Word2Vec 训练（{mode}）', flush=True)
    print('='*50, flush=True)

    data_dir = 'data'
    if args.full:
        output_dir = 'chinese_wiki_full_output'
        raw_text_path = os.path.join(data_dir, 'zhwiki_raw.txt')
        seg_text_path = os.path.join(data_dir, 'zhwiki_seg_full.txt')
    else:
        output_dir = 'chinese_wiki_output'
        raw_text_path = os.path.join(data_dir, 'zhwiki_raw_10pct.txt')
        seg_text_path = os.path.join(data_dir, 'zhwiki_seg_10pct.txt')

    if not os.path.exists(raw_text_path):
        print(f'[错误] 未找到: {raw_text_path}', flush=True)
        return

    size_mb = os.path.getsize(raw_text_path) / 1024 / 1024
    print(f'\n[Step 1] 原始文本: {raw_text_path} ({size_mb:.1f}MB)', flush=True)

    # 分词（如果已完成则跳过）
    if os.path.exists(seg_text_path) and os.path.getsize(seg_text_path) > 100000:
        print(f'\n[Step 2] 分词文件已存在: {seg_text_path}', flush=True)
    else:
        print(f'\n[Step 2] 多进程分词...', flush=True)
        preprocess_and_segment(raw_text_path, seg_text_path, workers=cpu_count())

    # 训练
    print(f'\n[Step 3] 开始训练...', flush=True)
    train_word2vec(seg_text_path, output_dir)


if __name__ == '__main__':
    main()
