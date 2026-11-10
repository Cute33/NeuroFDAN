"""W1之后的E3/E4只读探针：固定随机数分离输入、中介噪声和参考边缘化。"""
import argparse
import json
import random
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT / 'scripts')]
from probe_backbone_checkpoint import activation_summary, file_hash, tensor_summary


def difference(original, changed) -> dict:
    """相同shape配对相减；不把数值敏感度解释为生理因果效应。"""
    import torch
    if original.shape != changed.shape:
        raise ValueError('配对比较shape不同')
    a, b = original.detach().double(), changed.detach().double()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise FloatingPointError('配对张量含NaN/Inf')
    delta = b - a
    rms, base = delta.square().mean().sqrt().item(), a.square().mean().sqrt().item()
    return dict(delta_rms=rms, delta_abs_max=delta.abs().max().item(), baseline_rms=base,
                relative_delta_rms=rms / base if base > 0 else None)


def probability_summary(probability, labels, dataset: str) -> dict:
    """仅所选窗口的描述性矩阵，不是完整训练/测试成绩。"""
    import torch
    from ModelTrain import confusion_metrics
    p = probability.detach().double()
    if p.shape != (len(labels), 2) or not torch.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
        raise FloatingPointError('输出必须为有限[B,2]概率')
    if not torch.allclose(p.sum(-1), torch.ones(len(labels), dtype=p.dtype), atol=1e-5, rtol=1e-5):
        raise FloatingPointError('输出概率未归一化')
    prediction = p.argmax(-1)
    cm = torch.bincount(labels.long() * 2 + prediction, minlength=4).reshape(2, 2)
    return dict(mean=p.mean(0).tolist(), std=p.std(0, unbiased=False).tolist(),
                confusion_matrix=cm.tolist(), metrics=confusion_metrics(cm, dataset),
                probabilities=p.tolist(), note='小批训练窗口，只描述探针，不作正式分类结果')


def read_reference_windows(root: Path, dataset: str, task: str, selected_ids):
    """每个所需被试只加载一次；复制所需窗口后释放其完整数据，CNS/ANS始终同源。"""
    import torch
    from Datasets import PhysiologicalWindows
    requested = [tuple(row) for row in selected_ids.tolist()]
    unique = set(requested)
    found = {}
    for subject in sorted({row[0] for row in unique}):
        data = PhysiologicalWindows(root, dataset, [subject], task)
        lookup = {tuple(row): i for i, row in enumerate(data.source_ids.tolist())}
        for source in sorted(row for row in unique if row[0] == subject):
            if source not in lookup:
                raise ValueError('参考来源在实际数据中不存在')
            i = lookup[source]
            found[source] = (data.x_c[i].clone(), data.x_a[i].clone())
        del data
    return torch.stack([found[row][0] for row in requested]), torch.stack([found[row][1] for row in requested])


def inspect(model, xc, xa, labels, reference_c, reference_a, eps_c, other_eps_c, eps_a) -> dict:
    """同一模型eval/no_grad，全部比较复用同一输入、参考和显式外生标准正态样本。"""
    import torch
    model.cpu().eval()
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    for name, value in before.items():
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise FloatingPointError(f'模型有非有限参数/buffer：{name}')
    if any(m.training for m in model.modules()):
        raise RuntimeError('探针必须为完整eval模式')
    fd = model.frontdoor
    b, s, d = len(xc), eps_c.shape[1], model.frontdoor_config.mediator_dim
    trace, handles = {}, []
    active = ['未设置']

    def capture_drive(module, inputs):
        drive, extra = inputs
        if drive.ndim != 3 or extra is None or extra.shape != (drive.shape[0], drive.shape[2]):
            raise ValueError('ANS驱动/附加电流shape与当前SCM不一致')
        trace[active[0]] = dict(drive=tensor_summary(drive), extra_current=tensor_summary(extra))

    def capture_projection(module, inputs, output):
        # 读取真实input_proj输出，不重复实现或替代SNN前向。
        intrinsic = output + fd.ans.snn.cell.bias
        row = trace[active[0]]
        row['intrinsic_current'] = tensor_summary(intrinsic)
        base = row['intrinsic_current']['rms']
        row['extra_to_intrinsic_rms_ratio'] = row['extra_current']['rms'] / base if base > 0 else None

    handles.append(fd.ans.snn.register_forward_pre_hook(capture_drive))
    handles.append(fd.ans.snn.cell.input_proj.register_forward_hook(capture_projection))
    try:
        with torch.no_grad(), torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            hc, ha = model.backbone(xc, xa)
            features_c, features_a = [], []
            for start in range(0, len(reference_c), 16):
                c, a = model.backbone(reference_c[start:start+16], reference_a[start:start+16])
                features_c.append(c); features_a.append(a)
            own = fd.reference_encoder(hc, ha)
            ref = fd.reference_encoder(torch.cat(features_c), torch.cat(features_a))
            ref_grid = ref[None].expand(b, -1, -1)  # [B,K,d_ref]，各当前窗口共享同一K个经验参考。
            cns = fd.cns(hc, s)
            mu_c, logvar_c = cns['mu_c'], cns['logvar_c']
            # 同一epsilon在B轴共享，S轴独立；隔离输入差异，保持各点正确高斯边缘分布。
            shared_c = eps_c.expand(b, s, d)
            mc = mu_c[:, None] + (.5 * logvar_c[:, None]).exp() * shared_c
            mc_changed_input = mu_c.roll(1, 0)[:, None] + (.5 * logvar_c.roll(1, 0)[:, None]).exp() * shared_c
            mc_changed_noise = mu_c[:, None] + (.5 * logvar_c[:, None]).exp() * other_eps_c.expand(b, s, d)
            cases = [('baseline', ha, mc), ('change_cns_input', ha, mc_changed_input),
                     ('change_ans_input', ha.roll(1, 0), mc), ('change_cns_noise', ha, mc_changed_noise)]
            ans, samples, do_probabilities, own_probabilities = {}, {}, {}, {}
            base_logits = None
            for name, features, conditional_mc in cases:
                active[0] = name
                result = fd.ans(features, conditional_mc)
                ans[name] = result
                ma = result['mu_a'] + (.5 * result['logvar_a']).exp() * eps_a.expand(b, s, d)
                samples[name] = ma  # [B,S,D]，所有比较复用同一ANS标准正态噪声。
                probability, logits = fd.marginalize(ma, ref_grid)
                do_probabilities[name] = probability
                own_probability, own_logits = fd.marginalize(ma, own[:, None, :])
                own_probabilities[name] = own_probability
                if name == 'baseline':
                    base_logits = logits
                    factual_single = own_logits[:, 0, 0].softmax(-1)
            # 固定ma，仅将事实条件中的g(x)替换为另一个真实窗口的g(x)。
            changed_reference, _ = fd.marginalize(samples['baseline'], own.roll(1, 0)[:, None, :])
            conditional_probs = base_logits.softmax(-1)
            e3 = dict(cns_input_to_mu=difference(mu_c, mu_c.roll(1, 0)),
                      cns_input_to_logvar=difference(logvar_c, logvar_c.roll(1, 0)),
                      cns_input_to_mc=difference(mc, mc_changed_input),
                      cns_noise_to_mc=difference(mc, mc_changed_noise), ans_paths={}, currents=trace)
            for name in ('change_cns_input', 'change_ans_input', 'change_cns_noise'):
                e3['ans_paths'][name] = dict(
                    mu_a=difference(ans['baseline']['mu_a'], ans[name]['mu_a']),
                    logvar_a=difference(ans['baseline']['logvar_a'], ans[name]['logvar_a']),
                    m_a=difference(samples['baseline'], samples[name]),
                    probs_do=difference(do_probabilities['baseline'], do_probabilities[name]))
            obs_single = probability_summary(factual_single, labels, 'DEAP')
            obs_average = probability_summary(own_probabilities['baseline'], labels, 'DEAP')
            intervention = probability_summary(do_probabilities['baseline'], labels, 'DEAP')
            e4 = dict(factual_single_draw=obs_single, factual_average_S=obs_average,
                      frontdoor_average_S_K=intervention,
                      single_to_average_S=difference(factual_single, own_probabilities['baseline']),
                      same_S_reference_marginalization=difference(own_probabilities['baseline'], do_probabilities['baseline']),
                      fixed_ma_changed_factual_reference=difference(own_probabilities['baseline'], changed_reference),
                      conditional_reference_pair_difference=difference(conditional_probs, conditional_probs.roll(1, 2)),
                      conditional_mediator_draw_difference=difference(conditional_probs, conditional_probs.roll(1, 1)),
                      note='条件组合的K轴配对差用于看Head依赖；K置换不改变积分均值，不是条件独立性检验。')
            summaries = dict(backbone_c=activation_summary(hc), backbone_a=activation_summary(ha),
                mu_c=activation_summary(mu_c), logvar_c=activation_summary(logvar_c),
                posterior_std_c=tensor_summary((.5 * logvar_c).exp()),
                mu_a=activation_summary(ans['baseline']['mu_a'].mean(1)),
                posterior_std_a=tensor_summary((.5 * ans['baseline']['logvar_a']).exp()))
    finally:
        for handle in handles:
            handle.remove()
    if not all(torch.equal(before[name], value) for name, value in model.state_dict().items()):
        raise RuntimeError('模型参数或buffer发生变化，停止')
    if any(p.grad is not None for p in model.parameters()):
        raise RuntimeError('只读探针不应创建参数梯度')
    return dict(E3=e3, E4=e4, distributions=summaries, model_state_unchanged=True, gradients_created=False)


def main() -> None:
    parser = argparse.ArgumentParser(description='E3/E4配对机制探针，不训练、不更新权重', add_help=False)
    parser.add_argument('-h', '--help', action='help', help='显示中文说明')
    parser.add_argument('--root_dir', type=Path, default=Path('/home/fss'), help='既有数据根目录')
    parser.add_argument('--dataset', choices=['DEAP'], required=True, help='本次仅审计W1的DEAP')
    parser.add_argument('--task', choices=['arousal'], required=True, help='本次仅审计W1的arousal')
    parser.add_argument('--held_out_subject', type=int, required=True, help='原LOSO测试被试编号，仅作来源校验')
    parser.add_argument('--probe_subject', type=int, required=True, help='读取一个原训练被试，不用测试数据')
    parser.add_argument('--fold_dir', type=Path, required=True, help='W1 B组fold01，读取last.pt')
    parser.add_argument('--control_fold_dir', type=Path, required=True, help='W1 A组fold01，读取last.pt')
    parser.add_argument('--expected_epoch', type=int, required=True, help='两组last.pt预期轮次')
    parser.add_argument('--samples', type=int, default=16, help='两类各半的小批窗口数，必须为偶数')
    parser.add_argument('--S', type=int, default=16, help='配对中介采样数，不改训练配置')
    parser.add_argument('--K', type=int, default=64, help='固定经验参考抽样数，不改原2048成员参考库')
    parser.add_argument('--seed', type=int, choices=[42], default=42, help='固定噪声与参考抽样种子')
    parser.add_argument('--device', choices=['cpu'], default='cpu', help='仅CPU只读运行')
    parser.add_argument('--output_dir', type=Path, required=True, help='独立新探针目录，禁止写入原run')
    args = parser.parse_args()
    if not 4 <= args.samples <= 32 or args.samples % 2 or not 2 <= args.S <= 32 or not 2 <= args.K <= 64:
        parser.error('samples需为4–32偶数；S为2–32，K为2–64')
    import numpy as np
    import torch
    from Datasets import PhysiologicalWindows
    from NeuroFDAN import NeuroFDAN
    from frontdoor_scm import FrontDoorConfig
    from run_reporting import source_version
    torch.set_num_threads(1)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    destination = args.output_dir.resolve()
    folders = {'A_wd001': args.control_fold_dir.resolve(), 'B_wd0': args.fold_dir.resolve()}
    if len(set(folders.values())) != 2:
        raise ValueError('两组fold不能是同一个目录')
    if any(destination == folder.parent or folder.parent in destination.parents for folder in folders.values()):
        raise ValueError('不能写入任一原实验run或其子目录')
    if (destination / 'probe_summary.json').exists():
        raise FileExistsError('已有探针结果；使用独立新编号')
    payloads, configs, hashes = {}, {}, {}
    identity = (args.dataset, args.task, args.held_out_subject, args.seed)
    for label, folder in folders.items():
        path = folder / 'last.pt'
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError('需要服务器本地真实last.pt；GitHub诊断快照不含权重')
        fingerprint = file_hash(path)
        cfg = json.loads((folder / 'fold.json').read_text(encoding='utf-8'))
        payload = torch.load(path, map_location='cpu', weights_only=False)
        if payload.get('format_version') != 2 or payload['epoch'] != args.expected_epoch:
            raise ValueError('checkpoint格式或轮次不匹配')
        saved_args, report = payload['arguments'], payload['reference_report']
        if (saved_args['dataset'], saved_args['label'], report['held_out_subject'], saved_args['seed']) != identity:
            raise ValueError('checkpoint身份不匹配')
        if (cfg['dataset'], cfg['arguments']['label'], cfg['held_out_subject'], cfg['seed']) != identity:
            raise ValueError('fold.json身份不匹配')
        if cfg['model'] != payload['frontdoor_config'] or cfg['model']['no_cns2ans'] or cfg['model']['mediator_family'] != 'normal':
            raise ValueError('要求W1原高斯中介、CNS→ANS开启及相同模型配置')
        expected_decay = .01 if label == 'A_wd001' else 0.
        groups = payload['optimizer']['param_groups']
        if not groups or any(g['weight_decay'] != expected_decay for g in groups):
            raise ValueError('两组实际优化器衰减系数不符合W1定义')
        if saved_args['weight_decay'] != expected_decay or cfg['arguments'] != saved_args:
            raise ValueError('checkpoint与fold参数不一致')
        ids = payload['reference_source_ids'].cpu()
        if ids.dtype != torch.long or ids.ndim != 2 or ids.shape[1] != 3:
            raise ValueError('参考来源应为long类型[N,3]')
        observed = set(ids[:, 0].tolist())
        if (len(ids) < 1 or torch.unique(ids, dim=0).shape[0] != len(ids) or
                args.held_out_subject in observed or args.held_out_subject in report['train_subjects'] or
                not observed.issubset(set(report['train_subjects'])) or args.probe_subject not in report['train_subjects']):
            raise ValueError('参考库或当前探针来源不符合原训练总体')
        configs[label], payloads[label], hashes[label] = cfg, payload, fingerprint
    a, b = payloads['A_wd001'], payloads['B_wd0']
    if a['frontdoor_config'] != b['frontdoor_config'] or not torch.equal(a['reference_source_ids'], b['reference_source_ids']):
        raise ValueError('两组模型配置或完整参考库来源不一致')
    shared_arguments = {k: v for k, v in a['arguments'].items() if k not in ('run_name', 'weight_decay')}
    if shared_arguments != {k: v for k, v in b['arguments'].items() if k not in ('run_name', 'weight_decay')}:
        raise ValueError('两组训练参数存在衰减/名称以外差异')
    reference_ids = a['reference_source_ids'].cpu()
    reference_set = {tuple(row) for row in reference_ids.tolist()}
    data = PhysiologicalWindows(args.root_dir, args.dataset, [args.probe_subject], args.task)
    # 只约束探针窗口选择，不改变参考总体：选原参考库外的训练窗口，便于两组共用同一K个参考且无自身。
    eligible = torch.tensor([tuple(row) not in reference_set for row in data.source_ids.tolist()])
    chosen = []
    for cls in (0, 1):
        candidates = ((data.Label == cls) & eligible).nonzero(as_tuple=True)[0]
        if len(candidates) < args.samples // 2:
            raise ValueError('该训练被试参考库外两类窗口不足；不静默放宽约束或换被试')
        chosen.append(candidates[torch.linspace(0, len(candidates)-1, args.samples//2).long()])
    indices = torch.stack(chosen, 1).flatten()
    xc, xa, labels, current_ids = data.x_c[indices], data.x_a[indices], data.Label[indices], data.source_ids[indices]
    del data
    generator = torch.Generator().manual_seed(args.seed)
    selected = torch.randint(len(reference_ids), (args.K,), generator=generator)
    selected_ids = reference_ids[selected]
    rc, ra = read_reference_windows(args.root_dir, args.dataset, args.task, selected_ids)
    d = a['frontdoor_config']['mediator_dim']
    eps_c, other_eps_c, eps_a = [torch.randn(1, args.S, d, generator=generator) for _ in range(3)]
    result = dict(operation='W1后的E3/E4配对只读机制探针', dataset=args.dataset, task=args.task,
                  fold=args.held_out_subject, probe_subject=args.probe_subject, samples=args.samples,
                  seed=args.seed, S=args.S, K=args.K, device='cpu', source_version=source_version(PROJECT),
                  current_sources=current_ids.tolist(), current_label_counts=torch.bincount(labels, minlength=2).tolist(),
                  reference=dict(bank_count=len(reference_ids), training_population_count=a['reference_report']['population_windows'],
                    selected_sources=selected_ids.tolist(),
                    unique_selected_count=len(torch.unique(selected_ids, dim=0)), source_check_passed=True,
                    sampling='在原训练库中有放回均匀抽样；当前探针均非库成员，合法排除自身；不改变参考总体'),
                  input_statistics=dict(cns=activation_summary(xc), ans=activation_summary(xa)), models={},
                  checkpoint_sha256=hashes, optimizer_created=False, optimizer_step_executed=False,
                  checkpoint_training_source={key: cfg.get('source_version') for key, cfg in configs.items()},
                  note='跨窗口共享外生噪声、参考；每点边缘分布不变。局部替换可能超出联合支持，不是物理干预/条件独立检验/正式成绩。')
    for name in folders:
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        model = NeuroFDAN(ecg_channels=2, frontdoor_config=FrontDoorConfig(**payloads[name]['frontdoor_config']))
        model.load_state_dict(payloads[name]['model'], strict=True)
        print(f'开始{name} epoch={args.expected_epoch}：eval/no_grad，无optimizer.step', flush=True)
        diagnosis = inspect(model, xc, xa, labels, rc, ra, eps_c, other_eps_c, eps_a)
        result['models'][name] = dict(epoch=args.expected_epoch, weight_decay=payloads[name]['arguments']['weight_decay'], **diagnosis)
        for case, values in diagnosis['E3']['ans_paths'].items():
            print(f'  {case}：mu_a变化RMS={values["mu_a"]["delta_rms"]:.6e}；do概率变化RMS={values["probs_do"]["delta_rms"]:.6e}', flush=True)
        print('  E4单次事实、同S事实、同S前门P1均值：',
              [diagnosis['E4'][key]['mean'][1] for key in ('factual_single_draw', 'factual_average_S', 'frontdoor_average_S_K')], flush=True)
        del model
    for name, folder in folders.items():
        if file_hash(folder / 'last.pt') != hashes[name]:
            raise RuntimeError('原checkpoint文件读取期间发生变化，停止')
    result['checkpoint_files_unchanged'] = True
    content = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / 'probe_summary.json').open('x', encoding='utf-8') as stream:
        stream.write(content + '\n')
    print(f'完成：{destination / "probe_summary.json"}', flush=True)


if __name__ == '__main__':
    main()
