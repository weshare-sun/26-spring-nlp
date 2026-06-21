"""
训练效率对比分析
- 从 JSONL 日志读取训练过程数据
- Loss 收敛曲线（原始 + EMA 平滑 + epoch 标记）
- 训练时间对比柱状图
"""
import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')

# 版本 → 日志路径
VERSION_LOG_PATHS = {
    'NumPy':   os.path.join('numpy_version', 'output', 'training_log.jsonl'),
    'PyTorch': os.path.join('torch_version', 'output', 'training_log.jsonl'),
    'Gensim':  os.path.join('gensim_version', 'output', 'training_log.jsonl'),
}


def load_jsonl_log(path):
    """加载 JSONL 日志，返回记录列表"""
    records = []
    if not os.path.exists(path):
        print(f'[效率分析] 警告: 日志不存在 {path}')
        return records
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def compute_ema(values, alpha=0.05):
    """指数移动平均"""
    if not values:
        return []
    ema = [values[0]]
    for v in values[1:]:
        ema.append(alpha * v + (1 - alpha) * ema[-1])
    return ema


def plot_loss_curves(all_data):
    """绘制 Loss 收敛曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    colors = {'NumPy': '#FF6B6B', 'PyTorch': '#4ECDC4', 'Gensim': '#FFD93D'}

    # ---- 左上：NumPy + PyTorch Loss vs Batch ----
    ax1 = axes[0, 0]
    for name in ['NumPy', 'PyTorch']:
        records = all_data.get(name, [])
        if not records:
            continue
        batches = [r['batch'] for r in records]
        losses  = [r['loss']  for r in records]
        ema = compute_ema(losses, alpha=0.05)

        ax1.plot(batches, losses, color=colors[name], alpha=0.2, linewidth=0.8)
        ax1.plot(batches, ema, color=colors[name], linewidth=2,
                 label=f'{name} (EMA)')

        # epoch 分界竖线
        prev_epoch = 0
        for r in records:
            if r['epoch'] != prev_epoch:
                ax1.axvline(x=r['batch'], color=colors[name], alpha=0.3,
                            linestyle='--', linewidth=0.8)
                prev_epoch = r['epoch']

    ax1.set_xlabel('Batch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss vs Batch (NumPy / PyTorch)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ---- 右上：NumPy + PyTorch Loss vs Time ----
    ax2 = axes[0, 1]
    for name in ['NumPy', 'PyTorch']:
        records = all_data.get(name, [])
        if not records:
            continue
        times  = [r['time'] for r in records]
        losses = [r['loss'] for r in records]
        ema = compute_ema(losses, alpha=0.05)

        ax2.plot(times, losses, color=colors[name], alpha=0.2, linewidth=0.8)
        ax2.plot(times, ema, color=colors[name], linewidth=2,
                 label=f'{name} (EMA)')

        prev_epoch = 0
        for r in records:
            if r['epoch'] != prev_epoch:
                ax2.axvline(x=r['time'], color=colors[name], alpha=0.3,
                            linestyle='--', linewidth=0.8)
                prev_epoch = r['epoch']

    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss vs Time (收敛速度)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # ---- 左下：Gensim Epoch Loss ----
    ax3 = axes[1, 0]
    gensim_records = all_data.get('Gensim', [])
    if gensim_records:
        epochs  = [r['epoch'] for r in gensim_records]
        losses  = [r['loss'] for r in gensim_records]
        ax3.plot(epochs, losses, 'o-', color=colors['Gensim'], linewidth=2,
                 markersize=8, label='Gensim')
        for ep, lo in zip(epochs, losses):
            if lo is not None:
                ax3.annotate(f'{lo:.3f}', (ep, lo),
                             textcoords="offset points", xytext=(0, 10),
                             fontsize=9, ha='center')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Training Loss')
        ax3.set_title('Gensim Epoch Loss')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xticks(epochs)
    else:
        ax3.text(0.5, 0.5, '无 Gensim 日志数据', ha='center', va='center',
                 transform=ax3.transAxes, fontsize=14, color='gray')
        ax3.set_title('Gensim Epoch Loss')

    # ---- 右下：三版本 Loss 对照（epoch 级） ----
    ax4 = axes[1, 1]
    for name in ['NumPy', 'PyTorch']:
        records = all_data.get(name, [])
        if not records:
            continue
        # 按 epoch 聚合平均 loss
        epoch_losses = {}
        for r in records:
            ep = r['epoch']
            epoch_losses.setdefault(ep, []).append(r['loss'])
        ep_nums = sorted(epoch_losses.keys())
        avg_losses = [np.mean(epoch_losses[e]) for e in ep_nums]
        ax4.plot(ep_nums, avg_losses, 's-', color=colors[name], linewidth=2,
                 markersize=8, label=f'{name} (avg)')

    if gensim_records:
        epochs  = [r['epoch'] for r in gensim_records]
        losses  = [r['loss'] for r in gensim_records]
        ax4.plot(epochs, losses, 'o-', color=colors['Gensim'], linewidth=2,
                 markersize=8, label='Gensim')

    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss')
    ax4.set_title('Epoch 平均 Loss 对照')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(RESULTS_DIR, 'loss_curves.png'), dpi=150,
                bbox_inches='tight')
    plt.close()
    print('[效率分析] Loss 曲线已保存')


def get_training_times(all_data):
    """从 JSONL 日志推算训练时长（秒）"""
    times = {}
    for name, records in all_data.items():
        if not records:
            continue
        # 最后一行 time 作为总时长
        times[name] = records[-1].get('time', 0)
    return times


def plot_time_comparison(training_times):
    """绘制训练时间对比柱状图"""
    if not training_times:
        print('[效率分析] 无训练时间数据，跳过 time_comparison')
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    names = list(training_times.keys())
    times = [training_times[n] for n in names]

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    bars = ax.bar(names, times, color=colors[:len(names)])

    # 数值标签
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(times) * 0.02,
                f'{t:.0f}s', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('训练时间 (秒)')
    ax.set_title('各版本训练时间对比')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(RESULTS_DIR, 'time_comparison.png'), dpi=150,
                bbox_inches='tight')
    plt.close()
    print('[效率分析] 时间对比图已保存')


def compute_stats(all_data):
    """汇总统计"""
    stats = {}
    for name, records in all_data.items():
        if not records:
            continue
        losses = [r['loss'] for r in records if r.get('loss') is not None]
        times  = [r['time'] for r in records]
        stats[name] = {
            'total_time_s': round(times[-1], 1) if times else 0,
            'num_log_entries': len(records),
            'epochs': max(r['epoch'] for r in records) if records else 0,
            'final_loss': round(losses[-1], 4) if losses else None,
            'loss_range': f'{min(losses):.4f} - {max(losses):.4f}' if losses else 'N/A',
        }
    return stats


def run_efficiency_analysis():
    """运行效率分析"""
    print('=' * 50)
    print('训练效率对比分析')
    print('=' * 50)

    # 加载 JSONL 日志
    all_data = {}
    for name, path in VERSION_LOG_PATHS.items():
        records = load_jsonl_log(path)
        if records:
            print(f'[效率分析] {name}: {len(records)} 条日志记录')
            all_data[name] = records
        else:
            print(f'[效率分析] {name}: 无日志数据 (需要先运行训练)')

    if not all_data:
        print('[效率分析] 错误: 没有找到任何训练日志。请先运行训练脚本。')
        return

    # 绘制图表
    plot_loss_curves(all_data)

    training_times = get_training_times(all_data)
    plot_time_comparison(training_times)

    # 统计
    stats = compute_stats(all_data)

    # 打印汇总
    print('\n' + '=' * 60)
    print('训练效率汇总')
    print('=' * 60)
    print(f'\n{"版本":<10} {"训练时间":<12} {"日志条数":<10} {"Epochs":<8} '
          f'{"Final Loss":<12} {"Loss 范围"}')
    print('-' * 65)
    for name, s in stats.items():
        print(f'{name:<10} {s["total_time_s"]:<12.0f} {s["num_log_entries"]:<10} '
              f'{s["epochs"]:<8} {str(s["final_loss"]):<12} {s["loss_range"]}')

    # 保存
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, 'efficiency_stats.json'), 'w',
              encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print('\n[效率分析] 完成!')
    return stats


if __name__ == '__main__':
    run_efficiency_analysis()
