"""固定小批记忆能力诊断：只更新临时内存模型，不保存权重或覆盖原实验。"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT / 'scripts')]
from probe_backbone_checkpoint import activation_summary, file_hash


def source_grid_indices(population_size: int, samples: int) -> list[int]:
    """按来源顺序等间距取固定诊断窗口，不读取标签、不进行类别平衡。"""
    if samples < 2 or population_size < samples:
        raise ValueError('诊断窗口至少2个，且不能超过训练被试窗口总数')
    return [i*(population_size-1)//(samples-1) for i in range(samples)]


class FixedBatchLoader:
    """每次迭代重复所选窗口；次数是诊断步数，不是假装新增样本或epoch。"""
    def __init__(self, batch: dict, repeats: int):
        self.batch, self.repeats = batch, repeats

    def __iter__(self):
        for _ in range(self.repeats):
            yield self.batch, None

    def __len__(self):
        return self.repeats


def measure(model, batch: dict) -> dict:
    """两种模式各4个固定随机流；只读，核对参数/BN buffer，恢复训练随机流。"""
    import torch
    from ModelTrain import confusion_metrics
    result = {}
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    was_training = model.training
    try:
        for train_mode, label in ((True, 'training_mode_without_step'), (False, 'eval_mode')):
            model.train(train_mode)
            if any(m.training for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)):
                raise RuntimeError('没有保持原BN冻结策略')
            probabilities, ce, acc, spreads = [], [], [], []
            with torch.no_grad(), torch.random.fork_rng(devices=[]):
                for seed in (42, 43, 44, 45):
                    torch.manual_seed(seed)
                    out = model(batch['x_c'], batch['x_a'], source_ids=batch['source_ids'], fd_mode='none')
                    p = out['probs_obs'].double()
                    if not torch.isfinite(p).all() or not torch.allclose(p.sum(-1), torch.ones(len(p), dtype=p.dtype), atol=1e-5, rtol=1e-5):
                        raise FloatingPointError('诊断事实概率非有限或未归一化')
                    value = torch.nn.functional.cross_entropy(out['logits_obs'], batch['Labels'])
                    if not torch.isfinite(value):
                        raise FloatingPointError('诊断CE非有限')
                    probabilities.append(p)
                    ce.append(float(value))
                    acc.append(float((p.argmax(-1) == batch['Labels']).double().mean()))
                    spreads.append(float(out['mediator_statistics']['mu_c_spread']))
            average = torch.stack(probabilities).mean(0)
            prediction = average.argmax(-1)
            cm = torch.bincount(batch['Labels'] * 2 + prediction, minlength=4).reshape(2, 2)
            result[label] = dict(single_draw_ce=ce, single_draw_accuracy=acc,
                ce_mean=sum(ce)/4, accuracy_mean=sum(acc)/4,
                averaged_four_draw_confusion=cm.tolist(), averaged_four_draw_metrics=confusion_metrics(cm, 'DEAP'),
                mean_class_probability=average.mean(0).tolist(), mu_c_spread_mean=sum(spreads)/4)
    finally:
        model.train(was_training)
    if not all(torch.equal(before[k], v) for k, v in model.state_dict().items()):
        raise RuntimeError('诊断评估改变参数/buffer')
    result['evaluation_state_unchanged'] = True
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description='固定小批记忆能力诊断；临时模型会执行优化更新，原权重只读', add_help=False)
    parser.add_argument('-h', '--help', action='help', help='显示中文帮助')
    parser.add_argument('--root_dir', type=Path, default=Path('/home/fss'), help='原数据根目录')
    parser.add_argument('--fold_dir', type=Path, required=True, help='DEAP自然训练总体、batch400的epoch20原fold目录')
    parser.add_argument('--output_dir', type=Path, required=True, help='独立诊断目录，不写入原run')
    parser.add_argument('--probe_subject', type=int, default=2, help='原训练被试，仅作固定小批诊断')
    parser.add_argument('--samples', type=int, default=16, help='按来源顺序等间距取固定窗口，不按类别选择；仅诊断')
    parser.add_argument('--steps', type=int, default=300, help='每个临时模型最多更新次数，范围50–300')
    parser.add_argument('--check_every', type=int, default=50, help='检查间隔，需整除steps')
    parser.add_argument('--seed', type=int, choices=[42], default=42, help='固定初始化/训练随机种子')
    parser.add_argument('--device', choices=['cpu'], default='cpu', help='仅CPU小批诊断，不启动GPU/LOSO')
    args = parser.parse_args()
    if not 4 <= args.samples <= 32 or not 50 <= args.steps <= 300 or args.check_every < 1 or args.steps % args.check_every:
        parser.error('samples为4–32，steps为50–300，check_every为steps的正整数因子')
    import torch
    from Datasets import PhysiologicalWindows
    from NeuroFDAN import NeuroFDAN
    from frontdoor_scm import FrontDoorConfig
    from ModelTrain import train_one_epoch, seed_everything
    from run_reporting import source_version
    torch.set_num_threads(1)
    folder, destination = args.fold_dir.resolve(), args.output_dir.resolve()
    if destination == folder.parent or folder.parent in destination.parents:
        raise ValueError('不能写入原实验run')
    if (destination / 'probe_summary.json').exists():
        raise FileExistsError('已有结果；使用新编号，不覆盖')
    path = folder / 'last.pt'
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError('需要服务器原last.pt，GitHub诊断快照不含权重')
    fingerprint = file_hash(path)
    cfg = json.loads((folder / 'fold.json').read_text(encoding='utf-8'))
    payload = torch.load(path, map_location='cpu', weights_only=False)
    old = payload['arguments']
    if payload.get('format_version') != 2 or payload['epoch'] != 20 or payload['frontdoor_config'] != cfg['model']:
        raise ValueError('需要当前DEAP大batch400的epoch20模型')
    if (old['dataset'], old['label'], old['seed'], old['batch_size'], old['sampling_mode']) != ('DEAP', 'arousal', 42, 400, 'window'):
        raise ValueError('原checkpoint身份/配置不匹配')
    if cfg['arguments'] != old or cfg['smoke'] or old['max_train_batches'] or old['max_test_batches']:
        raise ValueError('需要配置一致的完整20轮实验，而非smoke')
    if old.get('no_balance') is not True or cfg['optimization_windows'] != cfg['population_windows']:
        raise ValueError('本次只接受已完成的自然训练总体模型，不复用旧平衡模型')
    report = payload['reference_report']
    if report['held_out_subject'] != 1 or args.probe_subject == 1 or args.probe_subject not in report['train_subjects']:
        raise ValueError('只允许使用原训练被试，不读held-out输入或标签')
    groups = payload['optimizer']['param_groups']
    if len(groups) != 1 or groups[0]['weight_decay'] != 0 or groups[0]['lr'] != .0001 or groups[0].get('decoupled_weight_decay', False):
        raise ValueError('要求原单组Adam、wd=0、lr=1e-4，不静默改变配置')
    obs_weight = old['lambda_obs']
    fire_weight = old['fire_weight'] if old['fire_weight'] is not None else 0.
    if obs_weight != 1. or fire_weight != 0.:
        raise ValueError('原DEAP观察CE/发放率权重与本次诊断预期不符')
    seed_everything(42)
    data = PhysiologicalWindows(args.root_dir, 'DEAP', [args.probe_subject], 'arousal')
    indices = torch.tensor(source_grid_indices(len(data), args.samples), dtype=torch.long)
    batch = dict(x_c=data.x_c[indices], x_a=data.x_a[indices], Labels=data.Label[indices], source_ids=data.source_ids[indices])
    if len(torch.unique(batch['Labels'])) != 2:
        raise ValueError('来源固定的诊断窗口只有一个类别；停止，不自动重采样或更换被试')
    del data
    signatures = {}
    for i, label in enumerate(batch['Labels'].tolist()):
        digest = hashlib.sha256(batch['x_c'][i].contiguous().numpy().tobytes() +
                                batch['x_a'][i].contiguous().numpy().tobytes()).hexdigest()
        signatures.setdefault(digest, set()).add(label)
    if any(len(labels) > 1 for labels in signatures.values()):
        raise ValueError('所选窗口存在完全相同输入但相反标签，先核查数据，不启动记忆训练')
    saved_steps = sorted({int(v['step'].item() if torch.is_tensor(v['step']) else v['step'])
                          for v in payload['optimizer']['state'].values() if 'step' in v})
    result = dict(operation=f'固定{args.samples}窗口记忆能力诊断，不是正式前门训练或分类实验',
        dataset='DEAP', task='arousal', fold=1, seed=42, device='cpu', samples=args.samples,
        source_version=source_version(PROJECT), checkpoint_epoch=20, checkpoint_sha256=fingerprint,
        original_training_source=cfg.get('source_version'), sample_sources=batch['source_ids'].tolist(),
        source_optimizer_step_values=saved_steps,
        true_counts=torch.bincount(batch['Labels'], minlength=2).tolist(),
        selection_policy='固定训练被试的来源索引等间距窗口，不按标签选择或平衡；不是正式训练子集规则',
        duplicate_input_count=args.samples-len(signatures), contradictory_duplicate_inputs=False,
        input_statistics=dict(cns=activation_summary(batch['x_c']), ans=activation_summary(batch['x_a'])),
        loss_epoch=20, steps_per_model=args.steps, models={}, optimizer_moments_reset=True,
        note='临时模型实际更新；原数据、训练器和模型结构不变。仅事实CE+原KL/fire，不算do或oracle监督。诊断窗口均来自本次自然训练总体，不修改正式训练集。')
    for initialization in ('initial_seed42', 'epoch20_weights'):
        seed_everything(42)
        model = NeuroFDAN(ecg_channels=2, frontdoor_config=FrontDoorConfig(**payload['frontdoor_config'])).cpu()
        if initialization == 'epoch20_weights':
            model.load_state_dict(payload['model'], strict=True)
        before = {k: v.detach().clone() for k, v in model.named_parameters()}
        group = groups[0]
        # 两组都用新优化器，测试当前参数能否拟合；不装载或改写原Adam历史moment。
        optimizer = torch.optim.Adam(model.parameters(), lr=group['lr'], betas=tuple(group['betas']),
            eps=group['eps'], weight_decay=group['weight_decay'], amsgrad=group.get('amsgrad', False))
        records = [dict(step=0, diagnostic=measure(model, batch))]
        print(f'开始{initialization}：临时模型最多{args.steps}步，无权重写盘', flush=True)
        for step in range(args.check_every, args.steps+1, args.check_every):
            optimization, _ = train_one_epoch(model, FixedBatchLoader(batch, args.check_every), None,
                optimizer, torch.device('cpu'), 'DEAP', epoch=20, fd_mode='none',
                lambda_obs=obs_weight, fire_weight=fire_weight, diagnostic_level='full')
            diagnostic = measure(model, batch)
            records.append(dict(step=step, optimization=optimization, diagnostic=diagnostic))
            print(f'  step={step} 优化CE={optimization["factual_ce"]:.5f} '
                  f'训练态只读CE={diagnostic["training_mode_without_step"]["ce_mean"]:.5f} '
                  f'eval CE={diagnostic["eval_mode"]["ce_mean"]:.5f} '
                  f'eval单次ACC均值={diagnostic["eval_mode"]["accuracy_mean"]:.4f}', flush=True)
        changed = any(not torch.equal(before[k], p) for k, p in model.named_parameters())
        last = records[-1]['diagnostic']
        # 这是预声明的记忆检查阈值，不是因果识别或正式选模标准；不提前终止或挑最好检查点。
        robust_fit = all(last[key]['ce_mean'] <= .1 and min(last[key]['single_draw_accuracy']) >= .95
                         for key in ('training_mode_without_step', 'eval_mode'))
        result['models'][initialization] = dict(records=records, private_parameters_updated=changed,
            final_memory_threshold_met=robust_fit, steps_executed=args.steps)
        del model, optimizer
    if file_hash(path) != fingerprint:
        raise RuntimeError('原checkpoint读取期间发生变化')
    result.update(completed=True, temporary_training_executed=True, formal_loso_training=False,
                  original_checkpoint_unchanged=True, weights_saved=False)
    content = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / 'probe_summary.json').open('x', encoding='utf-8') as stream:
        stream.write(content + '\n')
    print(f'完成：{destination / "probe_summary.json"}；不据记忆阈值声明正式分类或因果效果', flush=True)


if __name__ == '__main__':
    main()
