"""
一键运行所有评估
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.efficiency_analysis import run_efficiency_analysis
from evaluation.quality_eval import evaluate_all_versions
from evaluation.visualization import run_visualization
from evaluation.cross_lingual import compare_languages


def run_all():
    """运行所有评估"""
    print('='*60)
    print('Word2Vec 实验评估 - 全部运行')
    print('='*60)
    print(f'开始时间: {time.strftime("%Y-%m-%d %H:%M:%S")}')
    print()

    t0 = time.time()

    # 1. 效率分析
    try:
        print('[1/4] 训练效率分析...')
        run_efficiency_analysis()
    except Exception as e:
        print(f'[1/4] 效率分析失败: {e}')

    print()

    # 2. 质量评估
    try:
        print('[2/4] 词向量质量评估...')
        evaluate_all_versions()
    except Exception as e:
        print(f'[2/4] 质量评估失败: {e}')

    print()

    # 3. 可视化
    try:
        print('[3/4] 词向量可视化...')
        run_visualization()
    except Exception as e:
        print(f'[3/4] 可视化失败: {e}')

    print()

    # 4. 中英文对比
    try:
        print('[4/4] 中英文对比分析...')
        compare_languages()
    except Exception as e:
        print(f'[4/4] 对比分析失败: {e}')

    total_time = time.time() - t0

    print()
    print('='*60)
    print(f'评估完成! 总耗时: {total_time:.1f}s')
    print(f'结果保存在: evaluation/results/')
    print('='*60)


if __name__ == '__main__':
    run_all()
