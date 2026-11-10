"""E2：只读分项梯度与现有Adam耦合L2量级，不创建优化器或更新权重。"""
import argparse
import json
import math
import random
import re
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT / 'scripts')]
from probe_backbone_checkpoint import activation_summary, file_hash, tensor_summary


def module_group(name: str) -> str:
    for prefix, group in [('backbone.cns_enc.', 'cns_backbone'), ('backbone.ans_enc.', 'ans_backbone'),
                          ('backbone.dag_adj', 'graph'), ('frontdoor.cns.', 'cns_mediator'),
                          ('frontdoor.ans.cns_current.', 'cns2ans'), ('frontdoor.ans.', 'ans_mediator'),
                          ('frontdoor.outcome.', 'outcome'), ('frontdoor.reference_encoder.', 'reference')]:
        if name.startswith(prefix):
            return group
    raise ValueError(f'未明确归组的参数：{name}')


def square_norm(value) -> float:
    if value is None:
        return 0.
    import torch
    data = value.detach().double()
    if not torch.isfinite(data).all():
        raise FloatingPointError('分项梯度或衰减方向含NaN/Inf，停止')
    return data.square().sum().item()


def measure(model, payload: dict, xc, xa, y, source_ids, mode: str) -> dict:
    """一次固定批次的原事实目标；autograd.grad不写parameter.grad。"""
    import torch
    from frontdoor_scm import frontdoor_loss
    if mode == 'train_frozen_bn':
        model.train()
    else:
        model.eval()
    if any(m.training for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)):
        raise RuntimeError('BN没有遵守原冻结策略，停止')
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    for name, value in before.items():
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise FloatingPointError(f'非有限模型参数或buffer：{name}')
    named = list(model.named_parameters())
    if not all(p.requires_grad for _, p in named):
        raise ValueError('存在被冻结的Parameter，与本阶段原训练器假设不同；不自动解冻')
    saved = payload['optimizer']
    groups = saved['param_groups']
    if len(groups) != 1 or len(groups[0]['params']) != len(named) or len(set(groups[0]['params'])) != len(named):
        raise ValueError('保存优化器不是原单组model.parameters顺序，无法可靠映射Adam状态')
    group = groups[0]
    if not {'betas', 'eps', 'weight_decay', 'lr'}.issubset(group) or group.get('decoupled_weight_decay', False):
        raise ValueError('无法按原Adam耦合L2解释此优化器配置')
    decay = float(group['weight_decay'])
    params = [p for _, p in named]
    records, grouped = {}, {}
    for (name, p), identifier in zip(named, group['params']):
        mgroup = module_group(name)
        moment = saved['state'].get(identifier, {})
        if any(moment[key].shape != p.shape for key in ('exp_avg', 'exp_avg_sq') if key in moment):
            raise ValueError(f'Adam状态形状无法对应参数：{name}')
        step = moment.get('step')
        records[name] = dict(group=mgroup, parameter=tensor_summary(p),
            adam_step=int(step.item() if torch.is_tensor(step) else step) if step is not None else None,
            adam_state_present=bool(moment),
            adam_exp_avg_l2=math.sqrt(square_norm(moment['exp_avg'])) if 'exp_avg' in moment else None,
            adam_exp_avg_sq_l2=math.sqrt(square_norm(moment['exp_avg_sq'])) if 'exp_avg_sq' in moment else None,
            mathematical_l2_direction_l2=decay * math.sqrt(square_norm(p)), gradients={})
        grouped.setdefault(mgroup, dict(parameter_tensors=0, gradients={}))['parameter_tensors'] += 1
    trace, handles = {}, []
    for name, module in model.backbone.named_modules():
        if re.fullmatch(r'(cns_enc|ans_enc)\.\d+', name):
            def record(module, inputs, output, layer=name):
                trace[layer] = activation_summary(output)
            handles.append(module.register_forward_hook(record))
    # 本地单批固定随机流；不尝试重现epoch1/37当时的训练批次或全部历史RNG。
    random.seed(42)
    import numpy as np
    np.random.seed(42)
    torch.manual_seed(42)
    try:
        with torch.enable_grad():
            out = model(x_c=xc, x_a=xa, source_ids=source_ids, fd_mode='none',
                        S=model.frontdoor_config.S_train)
            epoch, args, cfg = payload['epoch'], payload['arguments'], model.frontdoor_config
            terms = frontdoor_loss(out, y, cfg, epoch=epoch)
            obs_weight = float(args.get('lambda_obs', 1.))
            fire_weight = args.get('fire_weight')
            fire_weight = float(fire_weight if fire_weight is not None else (0. if args['dataset'] == 'DEAP' else 1.))
            beta = cfg.beta * min(max(epoch, 0) / cfg.beta_warmup_epochs, 1.)
            total = terms['loss'] + (obs_weight - 1.) * terms['loss_obs'] + fire_weight * out['fire_loss']
            components = [('weighted_kl_c', beta * out['kl_c'].mean()),
                          ('weighted_kl_a', beta * out['kl_a'].mean()),
                          ('weighted_firing', fire_weight * out['fire_loss']),
                          ('weighted_factual_ce', obs_weight * terms['loss_obs'])]
            losses = {name: float(value.detach()) for name, value in [('total', total), *components]}
            if not all(math.isfinite(value) for value in losses.values()):
                raise FloatingPointError('局部损失有非有限值，停止')
            total_grads = torch.autograd.grad(total, params, retain_graph=True, allow_unused=True)
            total_norm = math.sqrt(sum(square_norm(g) for g in total_grads))
            # 只计算原max_norm=1规则的缩放系数，不改任何grad；float64估计不承诺与foreach逐位相同。
            coefficient = min(1., 1. / (total_norm + 1e-6))
            active_decay_squares = {}
            for (name, p), gradient in zip(named, total_grads):
                mgroup = records[name]['group']
                records[name]['total_grad_present'] = gradient is not None
                records[name]['estimated_postclip_total_l2'] = coefficient * math.sqrt(square_norm(gradient))
                square = decay ** 2 * square_norm(p) if gradient is not None else 0.
                active_decay_squares[mgroup] = active_decay_squares.get(mgroup, 0.) + square
                records[name]['adam_eligible_l2_direction_l2'] = math.sqrt(square)
            def summarize_gradients(label, gradients):
                accum = {}
                for (name, p), gradient in zip(named, gradients):
                    mgroup = records[name]['group']
                    item = accum.setdefault(mgroup, dict(squared=0., missing=0, zero=0))
                    sq = square_norm(gradient)
                    item['squared'] += sq
                    item['missing'] += int(gradient is None)
                    item['zero'] += int(gradient is not None and sq == 0.)
                    norm = math.sqrt(sq)
                    diagnostic = dict(present=gradient is not None, preclip_l2=norm,
                                      estimated_postclip_l2=coefficient * norm)
                    if label == 'weighted_factual_ce':
                        l2 = records[name]['adam_eligible_l2_direction_l2']
                        diagnostic['l2_to_postclip_ce_ratio'] = l2 / (coefficient * norm) if norm > 0 else None
                        diagnostic['zero_ce_with_nonzero_decay'] = gradient is not None and norm == 0 and l2 > 0
                        diagnostic['missing_ce_with_nonzero_decay'] = gradient is None and l2 > 0
                        if norm > 0 and l2 > 0 and records[name]['total_grad_present']:
                            dot = (gradient.detach().double() * p.detach().double() * decay).sum().item()
                            diagnostic['ce_decay_cosine'] = max(-1., min(1., dot / (norm * l2)))
                    records[name]['gradients'][label] = diagnostic
                for mgroup, values in accum.items():
                    norm = math.sqrt(values.pop('squared'))
                    data = dict(preclip_l2=norm, estimated_postclip_l2=coefficient * norm, **values)
                    if label == 'weighted_factual_ce':
                        l2 = math.sqrt(active_decay_squares[mgroup])
                        data['l2_direction_l2'] = l2
                        data['l2_to_postclip_ce_ratio'] = l2 / (coefficient * norm) if norm > 0 else None
                        data['zero_ce_with_nonzero_decay'] = norm == 0 and l2 > 0
                    grouped[mgroup]['gradients'][label] = data
            summarize_gradients('total', total_grads)
            for index, (label, loss) in enumerate(components):
                gradients = torch.autograd.grad(loss, params, retain_graph=index < len(components)-1,
                                                allow_unused=True) if loss.requires_grad else (None,) * len(params)
                summarize_gradients(label, gradients)
                del gradients
            probability = out['probs_obs'].detach().double()
            probability_statistics = dict(mean=probability.mean(0).tolist(), std=probability.std(0, unbiased=False).tolist())
    finally:
        for handle in handles:
            handle.remove()
    if len(trace) != 18:
        raise RuntimeError('未覆盖18个骨干模块，停止')
    if not all(torch.equal(before[name], value) for name, value in model.state_dict().items()):
        raise RuntimeError('模型参数或BN等buffer发生变化，停止')
    if any(p.grad is not None for p in params):
        raise RuntimeError('不应写入parameter.grad，停止')
    return dict(epoch=epoch, mode=mode, S=cfg.S_train, losses=losses,
                weights=dict(lambda_obs=obs_weight, effective_beta=beta, fire_weight=fire_weight, weight_decay=decay),
                global_total_gradient_l2_float64=total_norm, estimated_clip_coefficient=coefficient,
                module_gradients=grouped, parameter_statistics=records, backbone_layers=trace,
                factual_probability_statistics=probability_statistics,
                model_state_unchanged=True, parameter_grad_fields_unchanged=True,
                optimizer_configuration={k: group.get(k) for k in ('lr', 'betas', 'eps', 'weight_decay', 'amsgrad')},
                interpretation='Adam的L2在现有梯度裁剪之后加入；这是局部批次量级，不是历史完整训练梯度或实际更新量。')


def main() -> None:
    parser = argparse.ArgumentParser(description='E2只读分项梯度审计，不进行optimizer.step', add_help=False)
    parser.add_argument('-h', '--help', action='help', help='显示中文说明')
    parser.add_argument('--root_dir', type=Path, default=Path('/home/fss'), help='原数据根目录')
    parser.add_argument('--dataset', choices=['DEAP', 'HCI'], required=True, help='原训练数据集')
    parser.add_argument('--task', choices=['arousal', 'valence'], required=True, help='原训练任务')
    parser.add_argument('--held_out_subject', type=int, required=True, help='原held-out编号，仅用于来源校验')
    parser.add_argument('--probe_subject', type=int, required=True, help='从原训练被试选一人，不使用held-out标签计算梯度')
    parser.add_argument('--fold_dir', type=Path, required=True, help='原fold目录，只读')
    parser.add_argument('--e1b_summary', type=Path, required=True, help='已完成E1b结果，校验权重哈希')
    parser.add_argument('--expected_epoch', type=int, required=True, help='last.pt预期轮次')
    parser.add_argument('--output_dir', type=Path, required=True, help='独立新结果目录')
    parser.add_argument('--samples', type=int, default=16, help='训练被试局部窗口数，单批')
    parser.add_argument('--sampling', choices=['balanced', 'uniform'], default='balanced', help='仅探针批次按真实标签均衡或均匀取窗；不改变训练数据选择')
    parser.add_argument('--seed', type=int, choices=[42], default=42, help='固定局部随机流')
    parser.add_argument('--device', choices=['cpu'], default='cpu', help='仅CPU')
    parser.add_argument('--mode', choices=['train_frozen_bn', 'eval'], default='train_frozen_bn', help='原训练模式且BN冻结，或显式eval敏感度；两者都不更新参数')
    parser.add_argument('--checkpoints', nargs='+', choices=['best', 'last'], default=['best', 'last'], help='同一批窗口对比早期best和last')
    args = parser.parse_args()
    if not 2 <= args.samples <= 32 or len(set(args.checkpoints)) != len(args.checkpoints):
        parser.error('samples范围2–32；checkpoint不能重复')
    if args.sampling == 'balanced' and args.samples % 2:
        parser.error('均衡探针批次需要偶数samples')
    import numpy as np
    import torch
    from Datasets import PhysiologicalWindows
    from NeuroFDAN import NeuroFDAN
    from frontdoor_scm import FrontDoorConfig
    from run_reporting import source_version
    torch.set_num_threads(1)
    folder, destination = args.fold_dir.resolve(), args.output_dir.resolve()
    if destination == folder or folder in destination.parents or destination == args.e1b_summary.resolve().parent:
        raise ValueError('不能写入原fold或E1b目录')
    if (destination / 'probe_summary.json').exists():
        raise FileExistsError('已有结果，请使用新目录')
    destination.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((folder / 'fold.json').read_text(encoding='utf-8'))
    e1b = json.loads(args.e1b_summary.read_text(encoding='utf-8'))
    expected = (args.dataset, args.task, args.held_out_subject, args.seed)
    if (e1b['dataset'], e1b['task'], e1b['fold'], e1b['seed']) != expected or (
            cfg['dataset'], cfg['arguments']['label'], cfg['held_out_subject'], cfg['seed']) != expected:
        raise ValueError('任务、fold或seed不匹配')
    if args.probe_subject == args.held_out_subject or args.probe_subject not in cfg['train_subjects']:
        raise ValueError('梯度探针必须选原训练被试，不能使用held-out')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    data = PhysiologicalWindows(args.root_dir, args.dataset, [args.probe_subject], args.task)
    if args.sampling == 'balanced':
        selected = []
        for label in (0, 1):
            candidates = (data.Label == label).nonzero(as_tuple=True)[0]
            if len(candidates) < args.samples // 2:
                raise ValueError('该训练被试不足两类探针窗口；先核查标签或明确选择其他训练被试，不自动改采样')
            selected.append(candidates[torch.linspace(0, len(candidates)-1, args.samples//2).long()])
        indices = torch.stack(selected, dim=1).reshape(-1)
    else:
        indices = torch.linspace(0, len(data)-1, args.samples).long()
    xc, xa, y, ids = data.x_c[indices], data.x_a[indices], data.Label[indices], data.source_ids[indices]
    result = dict(operation='E2只读分项梯度与Adam耦合L2量级审计', dataset=args.dataset, task=args.task,
                  fold=args.held_out_subject, probe_subject=args.probe_subject, samples=args.samples, seed=args.seed,
                  probe_sampling=args.sampling,
                  device='cpu', source_version=source_version(PROJECT), sample_sources=ids.tolist(),
                  probe_label_counts=torch.bincount(y, minlength=2).tolist(), checkpoint_results={},
                  input_statistics=dict(cns=activation_summary(xc), ans=activation_summary(xa)),
                  optimizer_created=False, optimizer_step_executed=False,
                  note='训练总体中一个被试的局部窗口，未保证全部属于当时平衡优化子集；不代表历史全训练梯度；不使用oracle或do分类监督。')
    hashes = {}
    def load(label):
        path = folder / (label + '.pt')
        fingerprint = file_hash(path)
        if fingerprint != e1b['checkpoint_sha256'].get(path.name):
            raise ValueError(f'{path.name}不是E1b的同一权重')
        payload = torch.load(path, map_location='cpu', weights_only=False)
        if not isinstance(payload, dict) or payload.get('format_version') != 2 or payload['frontdoor_config'] != cfg['model']:
            raise ValueError(f'{path.name}格式或模型配置不匹配')
        p = payload['arguments']
        if (p['dataset'], p['label'], payload['reference_report']['held_out_subject'], p['seed']) != expected:
            raise ValueError(f'{path.name}来源身份不匹配')
        if args.probe_subject not in payload['reference_report']['train_subjects']:
            raise ValueError('checkpoint未将该探针被试记录为训练来源')
        if label == 'last' and payload['epoch'] != args.expected_epoch:
            raise ValueError('last.pt轮次不匹配')
        hashes[path.name] = fingerprint
        return payload
    # 真实last必须先校验，不能以其他模型顶替失败的checkpoint。
    last = load('last')
    for label in args.checkpoints:
        payload = last if label == 'last' else load(label)
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        model = NeuroFDAN(ecg_channels=2 if args.dataset == 'DEAP' else 3,
                          frontdoor_config=FrontDoorConfig(**payload['frontdoor_config'])).cpu()
        model.load_state_dict(payload['model'], strict=True)
        print(f'开始{label} epoch={payload["epoch"]}局部梯度审计；没有optimizer.step', flush=True)
        diagnosis = measure(model, payload, xc, xa, y, ids, args.mode)
        result['checkpoint_results'][label] = diagnosis
        print(f'  损失={diagnosis["losses"]}；估计裁剪系数={diagnosis["estimated_clip_coefficient"]:.6g}', flush=True)
        for group, values in diagnosis['module_gradients'].items():
            ce = values['gradients']['weighted_factual_ce']
            print(f'  {group}: CE范数={ce["preclip_l2"]:.6e} L2方向={ce["l2_direction_l2"]:.6e} '
                  f'L2/裁剪后CE={ce["l2_to_postclip_ce_ratio"]}', flush=True)
        del model
    for name, fingerprint in hashes.items():
        if file_hash(folder / name) != fingerprint:
            raise RuntimeError(f'{name}读取期间发生变化，停止')
    result.update(checkpoint_sha256=hashes, checkpoint_files_unchanged=True)
    content = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    with (destination / 'probe_summary.json').open('x', encoding='utf-8') as stream:
        stream.write(content + '\n')
    print(f'E2完成：{destination / "probe_summary.json"}', flush=True)


if __name__ == '__main__':
    main()
