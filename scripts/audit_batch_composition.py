"""只读训练标签和原采样器；不加载模型/checkpoint，不更新或改变正式采样策略。"""
import argparse
import json
import pickle
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


def summarize_batches(order, labels, trial_ids, batch_size: int) -> dict:
    """drop_last与原训练一致；报告窗口比例及不同来源trial数量。"""
    import numpy as np
    fractions, unique_trials = [], []
    for start in range(0, len(order) - batch_size + 1, batch_size):
        indices = order[start:start + batch_size]
        fractions.append(float(labels[indices].float().mean()))
        unique_trials.append(len(set(trial_ids[indices].tolist())))
    if not fractions:
        raise ValueError('没有完整batch')
    return dict(batches=len(fractions), positive_fraction=fractions,
                positive_fraction_mean=float(np.mean(fractions)), positive_fraction_std=float(np.std(fractions)),
                single_class_batches=sum(p in (0., 1.) for p in fractions),
                extreme_batches=sum(p <= .1 or p >= .9 for p in fractions),
                unique_trials_mean=float(np.mean(unique_trials)),
                unique_trials_min=min(unique_trials), unique_trials_max=max(unique_trials),
                last_batch_positive_fraction=fractions[-1],
                last_8_batches_positive_fraction_mean=float(np.mean(fractions[-8:])))


def main() -> None:
    parser = argparse.ArgumentParser(description='只读批次组成审计；窗口打乱仅为虚拟统计对照', add_help=False)
    parser.add_argument('-h', '--help', action='help', help='显示中文帮助')
    parser.add_argument('--root_dir', type=Path, default=Path('/home/fss'), help='原数据根目录')
    parser.add_argument('--fold_dir', type=Path, required=True, help='已完成20轮的独立续训fold目录')
    parser.add_argument('--restart_epoch', type=int, default=6, help='原续训重建采样器的轮次；本次为6')
    parser.add_argument('--output_dir', type=Path, required=True, help='独立审计输出目录')
    args = parser.parse_args()
    import numpy as np
    import torch
    from Datasets import DATASET_SPECS, GroupedSampler, PhysiologicalWindows, _numpy
    from run_reporting import source_version
    folder, destination = args.fold_dir.resolve(), args.output_dir.resolve()
    if destination == folder.parent or folder.parent in destination.parents:
        raise ValueError('不能写入原实验目录')
    if (destination / 'probe_summary.json').exists():
        raise FileExistsError('已有审计结果，必须使用新编号')
    cfg = json.loads((folder / 'fold.json').read_text(encoding='utf-8'))
    recorded = [json.loads(line) for line in (folder / 'epoch_metrics.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    original = cfg['arguments']
    if (cfg['dataset'], original['label'], cfg['held_out_subject'], cfg['seed'], original['epochs']) != ('DEAP', 'arousal', 1, 42, 20):
        raise ValueError('本轮限定DEAP arousal fold1 seed42、目标20轮')
    if args.restart_epoch != 6 or original['no_balance'] or original['workers'] != 0 or original['batch_size'] != 40:
        raise ValueError('原重启轮次、平衡/worker设置与本次审计假设不符')
    if [r['epoch'] for r in recorded] != list(range(1, 21)):
        raise ValueError('没有完整且唯一的1–20轮记录，停止声称重放对应轨迹')
    if not all(r['complete'] and not r['smoke'] for r in recorded):
        raise ValueError('存在不完整或smoke记录，不按完整20轮解释')
    subjects = cfg['train_subjects']
    if len(subjects) != 31 or set(subjects) != set(range(2, 33)) or cfg['held_out_subject'] in subjects:
        raise ValueError('训练来源不符合当前fold')
    spec = DATASET_SPECS['DEAP']
    binary_trials = []
    for subject in sorted(subjects):
        with (args.root_dir / 'Dataset/DEAP-python' / f'{subject}.npy').open('rb') as stream:
            record = pickle.load(stream)
        ratings = _numpy(record['labels'])
        if ratings.ndim != 2 or ratings.shape[0] != spec['trials'] or ratings.shape[1] < 2 or not np.isfinite(ratings[:, :2]).all():
            raise ValueError('训练标签格式或数值无效')
        binary_trials.extend((ratings[:, 1] > 5).astype(np.int64).tolist())
        del record
    # 只填充原balanced_indices所需属性，不执行信号预处理或创建替代训练集。
    population = object.__new__(PhysiologicalWindows)
    population.Label = torch.tensor(binary_trials, dtype=torch.long).repeat_interleave(spec['windows'])
    population.windows_per_trial, population.trials_per_subject = spec['windows'], spec['trials']
    selected = population.balanced_indices(42)
    labels, trial_ids = population.Label[selected], selected // spec['windows']
    if len(labels) != cfg['optimization_windows']:
        raise ValueError('本地标签重建的优化窗口数与原记录不同')
    group_sampler = GroupedSampler(labels, spec['windows'], seed=42)
    window_generator = torch.Generator().manual_seed(42)
    epochs = []
    for row in recorded:
        epoch = row['epoch']
        if epoch == args.restart_epoch:
            # 原run_fold在续训前重设seed并重新构建loader；重放该断点而非假设连续20轮。
            group_sampler = GroupedSampler(labels, spec['windows'], seed=42)
            window_generator.manual_seed(42)
        grouped = summarize_batches(torch.tensor(list(group_sampler)), labels, trial_ids, original['batch_size'])
        virtual = summarize_batches(torch.randperm(len(labels), generator=window_generator), labels, trial_ids, original['batch_size'])
        probability = row['test']['prediction_diagnostics']['primary']['probability_mean'][1]
        epochs.append(dict(epoch=epoch, grouped=grouped, virtual_window_shuffle=virtual,
                           recorded_test_p1_mean=probability,
                           recorded_factual_ce=row['optimization']['factual_ce']))
        print(f'epoch={epoch:02d} 原分组: 单类batch={grouped["single_class_batches"]}/{grouped["batches"]} '
              f'正类比例std={grouped["positive_fraction_std"]:.3f} 平均trial={grouped["unique_trials_mean"]:.2f}；'
              f'虚拟窗口打乱: 单类batch={virtual["single_class_batches"]} '
              f'比例std={virtual["positive_fraction_std"]:.3f}', flush=True)
    result = dict(operation='只读训练批次组成审计', dataset='DEAP', task='arousal', fold=1,
        source_version=source_version(PROJECT), seed=42, restart_epoch=args.restart_epoch,
        optimization_windows=len(labels), true_counts=torch.bincount(labels, minlength=2).tolist(),
        batch_size=original['batch_size'], epochs=epochs, training_executed=False,
        source_training_version=cfg.get('source_version'),
        model_loaded=False, sampler_changed=False, original_files_written=False,
        note='以服务器训练数据未改变为前提按已知断点重建；没有历史标签哈希，不能承诺历史逐位重放。窗口打乱仅为虚拟统计对照，批次失衡不独自证明学习失败原因。')
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / 'probe_summary.json').open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(f'审计完成：{destination / "probe_summary.json"}；未运行模型或训练', flush=True)


if __name__ == '__main__':
    main()
