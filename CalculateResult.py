"""按LOSO被试汇总混淆矩阵；不将未运行的全零行当成结果。"""
import argparse
import json
import os
from pathlib import Path

import numpy as np


def get_args(description='NeuroFDAN跨被试结果统计', argv=None):
    parser = argparse.ArgumentParser(description=description, add_help=False)
    parser.add_argument('-h', '--help', action='help', help='显示中文帮助并退出')
    parser.add_argument('--label', required=True, help='arousal或valence')
    parser.add_argument('--dataset', required=True, help='DEAP或HCI/MAHNOB_HCI')
    parser.add_argument('--root_dir', default=os.environ.get('NEURO_ROOT', '/home/fss'), help='兼容旧结果根目录')
    parser.add_argument('--result_file', default=None, help='第二阶段某个run目录下的混淆矩阵npy文件')
    parser.add_argument('--positive_class', type=int, choices=[0, 1], default=0, help='统计F1的正类；默认0保留旧统计脚本含义')
    parser.add_argument('--allow_smoke', action='store_true', help='允许查看不完整接口检查输出，不能作为论文实验结果')
    return parser.parse_args(argv)


def summarize(confusion_matrices, positive_class=0):
    matrices = np.asarray(confusion_matrices)
    if matrices.ndim != 3 or matrices.shape[1:] != (2, 2) or (matrices < 0).any() or not np.isfinite(matrices).all():
        raise ValueError('结果必须为非负、有限的[N,2,2]混淆矩阵')
    matrices = matrices[matrices.sum((1, 2)) > 0]
    if not len(matrices):
        raise ValueError('没有完成且满足选模规则的被试结果，不能统计全零占位行')
    scores = []
    p = positive_class
    for cm in matrices:
        tp, tn, fp, fn = cm[p, p], cm[1-p, 1-p], cm[1-p, p], cm[p, 1-p]
        accuracy = (tp + tn) / cm.sum()
        sensitivity = tp / (tp + fn) if tp + fn else 0.
        specificity = tn / (tn + fp) if tn + fp else 0.
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.
        scores.append([accuracy, sensitivity, specificity, f1])
    values = np.array(scores)
    return {'completed_subjects': len(values), 'mean': values.mean(0), 'std': values.std(0)}


def main():
    args = get_args()
    dataset = args.dataset.upper().replace('-', '_')
    dataset = 'HCI' if dataset == 'MAHNOB_HCI' else dataset
    if dataset not in ('DEAP', 'HCI') or args.label.lower() not in ('arousal', 'valence'):
        raise ValueError('数据集或任务名称无效')
    path = Path(args.result_file) if args.result_file else Path(args.root_dir) / 'NeuroFDAN' / f'NeuroFDAN-{dataset}_{args.label.lower()}_cm.npy'
    matrices = np.load(path)
    if args.allow_smoke:
        print('已显式允许接口检查输出，以下数值不得作为正式论文实验结果。')
    if not args.allow_smoke:
        for metadata in path.parent.glob('fold*/fold.json'):
            if json.loads(metadata.read_text(encoding='utf-8')).get('smoke'):
                raise ValueError('目录含接口检查结果，拒绝作为正式统计；仅查看时显式加--allow_smoke')
    result = summarize(matrices, args.positive_class)
    print(f'已统计{result["completed_subjects"]}名被试；全零/未运行/未达选模门槛的行不计入；F1正类={args.positive_class}')
    for index, name in enumerate(('准确率', '敏感度', '特异性', 'F1')):
        print(f'{name}均值：{result["mean"][index]:.4f}；被试间标准差：{result["std"][index]:.4f}')
    print('标准差来自LOSO被试，不是多随机种子；held-out测试集参与了checkpoint选择。')


if __name__ == '__main__':
    main()
