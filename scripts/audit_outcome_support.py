"""固定checkpoint的联合支持/后验/局部敏感度审计；不训练、不修改研究假设。"""
import argparse
import json
import math
import pickle
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT / 'scripts')]
from probe_backbone_checkpoint import file_hash, activation_summary
from probe_mechanism_paths import read_reference_windows


def describe(value) -> dict:
    """有限数值摘要；不把低密度阈值解释为识别条件检验。"""
    import torch
    x = value.detach().double().flatten()
    if not x.numel() or not torch.isfinite(x).all():
        raise FloatingPointError('诊断统计含空张量或NaN/Inf')
    return dict(mean=x.mean().item(), median=x.median().item(), min=x.min().item(),
                max=x.max().item(), p05=x.quantile(.05).item(), p95=x.quantile(.95).item())


def posterior_statistics(mu, logvar) -> dict:
    """第0轴为输入，末轴为潜变量；可有固定条件Mc网格轴，不混淆样本轴和采样轴。"""
    sigma = (.5 * logvar.double()).exp()
    mean = mu.double()
    vm, vs = mean.var(0, unbiased=False), sigma.var(0, unbiased=False)
    # expm1减少近先验时相消；仅诊断，原训练KL定义不改。
    kl = .5 * (mean.square() + logvar.double().expm1() - logvar.double()).sum(-1)
    per_dim_vm = vm if vm.ndim == 1 else vm.mean(0)
    per_dim_vs = vs if vs.ndim == 1 else vs.mean(0)
    return dict(var_X_mu=describe(vm), var_X_sigma=describe(vs),
                var_X_logvar=describe(logvar.double().var(0,unbiased=False)),
                var_X_mu_per_dimension=per_dim_vm.tolist(), var_X_sigma_per_dimension=per_dim_vs.tolist(),
                posterior_std=describe(sigma), gaussian_conditional_kl_float64=describe(kl))


def marginal_ans_moments(mu, logvar) -> dict:
    """Ma|X是对Mc的混合；按全方差公式估计，不能当作单高斯或直接用高斯KL。"""
    m = mu.double()
    conditional_var = logvar.double().exp()
    average = m.mean(1)
    total_var = (conditional_var.mean(1) + m.var(1, unbiased=False)).clamp_min(0.)
    std = total_var.sqrt()
    return dict(var_X_E_Ma=describe(average.var(0, unbiased=False)),
                var_X_total_std=describe(std.var(0, unbiased=False)),
                E_conditional_variance=describe(conditional_var.mean(1)),
                Var_Mc_conditional_mean=describe(m.var(1, unbiased=False)),
                note='共同随机数MC矩估计；没有把混合分布KL冒充高斯KL。')


def mixture_logp(values, mu, logvar):
    """values[Q,D]、条件分量[K,L,D] -> [Q,K]；分块并在log域平均Mc条件密度。"""
    import torch
    if values.ndim != 2 or mu.ndim != 3 or logvar.shape != mu.shape or values.shape[-1] != mu.shape[-1]:
        raise ValueError('条件混合密度shape不匹配')
    m, lv = mu.double(), logvar.double()
    pieces = []
    for start in range(0, len(values), 16):
        delta = values[start:start+16].double()[:, None, None, :] - m[None]
        log_component = -.5 * (delta.square() * (-lv).exp()[None] + lv[None] + math.log(2*math.pi)).sum(-1)
        pieces.append(torch.logsumexp(log_component, -1) - math.log(mu.shape[1]))
    return torch.cat(pieces)


def scaled_distance(a, b, scale):
    """块尺度归一、维数归一的平方距离，不作模型投影或修改ReferenceEncoder。"""
    x, y = a.double() / scale, b.double() / scale
    return ((x.square().mean(-1)[:, None] + y.square().mean(-1)[None]) -
            2 * (x @ y.T) / x.shape[-1]).clamp_min(0.)


def density_audit(query, components, own_components, calibration, own_calibration) -> dict:
    """参考专属独立MC校准；全部基于学习到的q，不是真实生理支持域证明。"""
    import torch
    b, s, d = query.shape
    cross = mixture_logp(query.reshape(-1, d), components['mu_a'], components['logvar_a'])
    own_all = mixture_logp(query.reshape(-1, d), own_components['mu_a'], own_components['logvar_a'])
    own = own_all[torch.arange(b*s), torch.arange(b).repeat_interleave(s)]
    ref_thresholds, own_thresholds = [], []
    for k in range(len(components['mu_a'])):
        values = calibration['m_a'][k]
        scores = mixture_logp(values, components['mu_a'][k:k+1], components['logvar_a'][k:k+1])[:, 0]
        ref_thresholds.append(scores.quantile(.05))
    for i in range(b):
        scores = mixture_logp(own_calibration['m_a'][i], own_components['mu_a'][i:i+1], own_components['logvar_a'][i:i+1])[:, 0]
        own_thresholds.append(scores.quantile(.05))
    return dict(own_log_density_per_dimension=describe(own / d),
                cross_log_density_per_dimension=describe(cross / d),
                cross_minus_own_per_dimension=describe((cross-own[:, None])/d),
                cross_below_reference_5pct_fraction=float((cross < torch.stack(ref_thresholds)[None]).double().mean()),
                own_below_own_5pct_fraction=float((own < torch.stack(own_thresholds).repeat_interleave(s)).double().mean()),
                note='完整高斯支持不等于有限样本overlap；q塌缩可使此检查看似良好，不能据此证明真实positivity。')


def capture_forward(model, c, a, training: bool, S: int, seed: int):
    """hook真实OutcomeHead的事实输入；无更新，不把重建分布说成历史全训练记录。"""
    import torch
    model.train(training)
    if any(m.training for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)):
        raise RuntimeError('BN未保持原冻结策略')
    captured = {}
    def hook(module, inputs):
        captured['ma'], captured['reference'] = (x.detach().clone() for x in inputs)
    handle = model.frontdoor.outcome.register_forward_pre_hook(hook)
    try:
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            torch.manual_seed(seed)
            out = model(c, a, fd_mode='none', S=S)
    finally:
        handle.remove()
    return out, captured


def local_sensitivity(head, ma, reference, scales, cns_dim: int) -> dict:
    """只对输入求局部梯度，不写parameter.grad；同量纲比较仍不等于因果贡献。"""
    import torch
    m, r = ma.detach().clone().requires_grad_(True), reference.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        probability = head(m, r).softmax(-1)[:, 1]
        gm, gr = torch.autograd.grad(probability.sum(), (m, r))
    result = {}
    for key, gradient, scale in [('ma', gm, scales[0]), ('reference_cns', gr[:, :cns_dim], scales[1]),
                                  ('reference_ans', gr[:, cns_dim:], scales[2])]:
        result[key] = dict(raw_gradient_rms=describe(gradient.double().square().mean(-1).sqrt()),
                           standardized_gradient_l2=describe((gradient.double()*scale).square().sum(-1).sqrt()))
    generator = torch.Generator().manual_seed(42)
    with torch.no_grad():
        baseline = head(ma, reference).softmax(-1)[:, 1]
        for key, start, end, scale in [('ma', 0, ma.shape[-1], scales[0]),
                ('reference_cns', 0, cns_dim, scales[1]), ('reference_ans', cns_dim, reference.shape[-1], scales[2])]:
            direction = torch.randn(len(ma), end-start, generator=generator)
            direction /= direction.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-12)
            mm, rr = ma.clone(), reference.clone()
            delta = direction * (.01 * scale)
            if key == 'ma':
                mm += delta
            else:
                rr[:, start:end] += delta
            result[key]['one_percent_block_scale_probability_delta'] = describe((head(mm, rr).softmax(-1)[:, 1]-baseline).abs())
    return result


def choose_sources(args, cfg, payload):
    """重建原优化窗口，和未平衡参考库分开记录；不改变任何正式总体。"""
    import torch
    import numpy as np
    from Datasets import DATASET_SPECS, PhysiologicalWindows, _numpy
    spec, dataset = DATASET_SPECS[cfg['dataset']], cfg['dataset']
    subjects = sorted(cfg['train_subjects'])
    labels = []
    for subject in subjects:
        with (args.root_dir/'Dataset'/f'{dataset}-python'/f'{subject}.npy').open('rb') as stream:
            record = pickle.load(stream)
        ratings = _numpy(record['labels'])
        column = 1 if cfg['arguments']['label'] == 'arousal' else 0
        if ratings.ndim != 2 or len(ratings) != spec['trials'] or ratings.shape[1] < 2 or not np.isfinite(ratings[:,:2]).all():
            raise ValueError('本地trial标签数不匹配')
        labels.extend((ratings[:, column] > 5).astype('int64').tolist())
        del record
    pop = object.__new__(PhysiologicalWindows)
    pop.Label = torch.tensor(labels, dtype=torch.long).repeat_interleave(spec['windows'])
    pop.windows_per_trial, pop.trials_per_subject = spec['windows'], spec['trials']
    indices = torch.arange(len(pop.Label)) if cfg['arguments']['no_balance'] else pop.balanced_indices(42)
    if len(indices) != cfg['optimization_windows']:
        raise ValueError('本地优化窗口数量与原记录不同')
    all_ids = torch.tensor([(subject, trial, window) for subject in subjects for trial in range(spec['trials']) for window in range(spec['windows'])])
    optimized = {tuple(x) for x in all_ids[indices].tolist()}
    reference_ids = payload['reference_source_ids'].cpu()
    if reference_ids.dtype != torch.long or reference_ids.ndim != 2 or reference_ids.shape[1] != 3:
        raise ValueError('参考来源需为long[N,3]')
    if not len(reference_ids) or len(reference_ids)!=cfg['actual_N_ref'] or len(torch.unique(reference_ids, dim=0)) != len(reference_ids) or not set(reference_ids[:, 0].tolist()) <= set(subjects):
        raise ValueError('原参考来源不唯一或含非训练被试')
    reference_set = {tuple(x) for x in reference_ids.tolist()}
    generator = torch.Generator().manual_seed(42)
    selected_ref = reference_ids[torch.randint(len(reference_ids), (args.K,), generator=generator)]
    current_parts = []
    for label in (0, 1):
        eligible = torch.tensor([i for i in indices.tolist() if pop.Label[i] == label and tuple(all_ids[i].tolist()) not in reference_set])
        if len(eligible) < args.samples//2:
            raise ValueError('参考库外的优化窗口不足两类诊断配额，不放宽来源约束')
        current_parts.append(eligible[torch.randperm(len(eligible), generator=generator)[:args.samples//2]])
    current = torch.stack(current_parts, 1).flatten()
    forbidden = {tuple(x) for x in selected_ref.tolist()} | {tuple(x) for x in all_ids[current].tolist()}
    remaining = torch.tensor([i for i in indices.tolist() if tuple(all_ids[i].tolist()) not in forbidden])
    if len(remaining) < args.anchors+args.calibration:
        raise ValueError('独立训练参照/距离校准窗口不足')
    picked = remaining[torch.randperm(len(remaining), generator=generator)[:args.anchors+args.calibration]]
    raw_counts = torch.bincount(pop.Label, minlength=2)
    opt_counts = torch.bincount(pop.Label[indices], minlength=2)
    selection = dict(raw_window_counts=raw_counts.tolist(), optimization_window_counts=opt_counts.tolist(),
        class_retention_probability=(opt_counts.double()/raw_counts.clamp_min(1)).tolist(),
        reference_sample_optimization_membership_fraction=sum(tuple(x) in optimized for x in selected_ref.tolist())/args.K,
        note='标签选择可能改变结果条件分布；本次只报告，不新增class weight或改变下采样。')
    return all_ids[picked[:args.anchors]], all_ids[picked[args.anchors:]], all_ids[current], selected_ref, selection


def main() -> None:
    parser = argparse.ArgumentParser(description='固定权重的结果条件模型支持域与CNS/ANS后验诊断', add_help=False)
    parser.add_argument('-h', '--help', action='help', help='显示中文帮助')
    parser.add_argument('--root_dir', type=Path, default=Path('/home/fss'), help='既有数据根目录')
    parser.add_argument('--fold_dir', type=Path, required=True, help='DEAP400或HCI100的完整epoch20 fold')
    parser.add_argument('--output_dir', type=Path, required=True, help='独立新审计目录')
    parser.add_argument('--samples', type=int, default=16, help='参考库外的优化查询窗口，两类各半')
    parser.add_argument('--anchors', type=int, default=128, help='优化集事实参照窗口数，不训练模型')
    parser.add_argument('--calibration', type=int, default=64, help='距离阈值诊断校准窗口，不是validation集')
    parser.add_argument('--K', type=int, default=64, help='原参考库内有放回抽样数')
    parser.add_argument('--L', type=int, default=16, help='条件Ma密度的CNS混合分量MC数')
    parser.add_argument('--C', type=int, default=64, help='独立条件密度校准及共同随机数矩估计次数')
    parser.add_argument('--seed', type=int, choices=[42], default=42, help='固定诊断随机流')
    parser.add_argument('--device', choices=['cpu'], default='cpu', help='仅CPU；无优化器，不训练')
    args = parser.parse_args()
    if args.samples % 2 or not 4 <= args.samples <= 32 or not 32 <= args.anchors <= 256 or not 32 <= args.calibration <= 128 or not 4 <= args.K <= 64 or not 4 <= args.L <= 32 or not 32 <= args.C <= 128:
        parser.error('诊断规模超出已限定范围')
    import torch
    from NeuroFDAN import NeuroFDAN
    from frontdoor_scm import FrontDoorConfig
    from run_reporting import source_version
    torch.set_num_threads(1)
    folder, dest = args.fold_dir.resolve(), args.output_dir.resolve()
    if dest == folder.parent or folder.parent in dest.parents or (dest/'probe_summary.json').exists():
        raise ValueError('不能覆盖原run或已有诊断')
    path = folder/'last.pt'
    fingerprint = file_hash(path)
    cfg = json.loads((folder/'fold.json').read_text(encoding='utf-8'))
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if cfg['dataset'] not in ('DEAP', 'HCI') or cfg['smoke'] or payload['epoch'] != 20 or payload.get('format_version') != 2:
        raise ValueError('需要真实完整20轮的原checkpoint')
    if cfg['arguments'] != payload['arguments'] or cfg['model'] != payload['frontdoor_config'] or cfg['held_out_subject'] != 1 or 1 in cfg['train_subjects']:
        raise ValueError('任务/来源/配置不一致')
    if cfg['model']['mediator_family'] != 'normal' or cfg['model']['ref_dim'] != 0 or cfg['model']['no_cns2ans']:
        raise ValueError('本诊断限定默认完整参考高斯分层中介，CNS→ANS开启')
    if payload['reference_report']['held_out_subject'] != 1 or payload['reference_report']['train_subjects'] != cfg['train_subjects']:
        raise ValueError('checkpoint参考库来源不符合fold')
    if cfg['seed'] != 42 or cfg['arguments']['seed'] != 42 or cfg['arguments']['max_train_batches'] or cfg['arguments']['max_test_batches']:
        raise ValueError('需要seed42及完整训练/评估记录')
    train_ids, cal_ids, current_ids, ref_ids, selection = choose_sources(args, cfg, payload)
    all_ids = torch.cat((train_ids, cal_ids, current_ids, ref_ids))
    print('来源检查完成，读取所需配对训练窗口；未读测试被试', flush=True)
    c, a = read_reference_windows(args.root_dir, cfg['dataset'], cfg['arguments']['label'], all_ids)
    n, nc, b = args.anchors, args.calibration, args.samples
    tc, ta, cc, ca = c[:n], a[:n], c[n:n+nc], a[n:n+nc]
    xc, xa, rc, ra = c[n+nc:n+nc+b], a[n+nc:n+nc+b], c[n+nc+b:], a[n+nc+b:]
    torch.manual_seed(42)
    model = NeuroFDAN(ecg_channels=2 if cfg['dataset']=='DEAP' else 3, frontdoor_config=FrontDoorConfig(**payload['frontdoor_config'])).cpu()
    model.load_state_dict(payload['model'], strict=True)
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    fd, dc = model.frontdoor, model.backbone.cns_dim
    with torch.no_grad():
        # 参照事实样本分小块收集多个训练态图样本，重建监督头输入分布，不声称历史回放。
        factual = {}
        for training, mode in ((True, 'train_mode'), (False, 'eval_mode')):
            chunks = [capture_forward(model, tc[i:i+32], ta[i:i+32], training, cfg['model']['S_train'] if training else cfg['model']['S_eval'], 100+i)[1] for i in range(0,n,32)]
            factual[mode] = {key: torch.cat([row[key] for row in chunks]) for key in ('ma','reference')}
        _, cal_train = capture_forward(model, cc, ca, True, cfg['model']['S_train'], 201)
        _, cal_eval = capture_forward(model, cc, ca, False, cfg['model']['S_eval'], 202)
        current, own = capture_forward(model, xc, xa, False, cfg['model']['S_eval'], 203)
        own_components, _ = capture_forward(model, xc, xa, False, args.L, 204)
        own_calib, _ = capture_forward(model, xc, xa, False, args.C, 205)
        ref_components, reference = capture_forward(model, rc, ra, False, args.L, 206)
        ref_calib, _ = capture_forward(model, rc, ra, False, args.C, 207)
        training_query, _ = capture_forward(model, xc, xa, True, cfg['model']['S_train'], 208)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(209)
            trc, tra = model.backbone(rc,ra)
            training_reference = fd.reference_encoder(trc,tra)
        model.eval()
        hc, ha = model.backbone(xc, xa)
        gen = torch.Generator().manual_seed(42)
        eps = torch.randn(1,args.C,cfg['model']['mediator_dim'],generator=gen)
        common_mc = current['mu_c'][:,None] + (.5*current['logvar_c'][:,None]).exp()*eps
        shared = fd.ans(ha, common_mc)
        fixed_mc = fd.ans(ha, common_mc[:1].expand(b,-1,-1))
        sampling = dict(cns=posterior_statistics(current['mu_c'],current['logvar_c']),
            ans_with_common_cns_noise=posterior_statistics(shared['mu_a'],shared['logvar_a']),
            ans_vary_xa_hold_mc=posterior_statistics(fixed_mc['mu_a'],fixed_mc['logvar_a']),
            ans_marginal_over_mc=marginal_ans_moments(shared['mu_a'],shared['logvar_a']),
            ans_conditional_mean_variation_over_mc=describe(shared['mu_a'].double().var(1,unbiased=False)),
            cns_sample_variance_given_input=describe(current['logvar_c'].double().exp()),
            cns_empirical_sample_std=describe(current['m_c'].double().std(1,unbiased=False)),
            ans_empirical_marginal_sample_std=describe(current['m_a'].double().std(1,unbiased=False)),
            note='ANS条件高斯KL是对Mc条件项的诊断，不是Ma|X混合分布的KL；共同噪声检查不是因果效应。')
        print('CNS/ANS均值、sigma和全方差统计完成；开始条件混合密度', flush=True)
        density = density_audit(current['m_a'], ref_components, own_components, ref_calib, own_calib)
        anchor = factual['train_mode']
        anchor_blocks = [anchor['ma'], anchor['reference'][:,:dc],anchor['reference'][:,dc:]]
        scales_raw = [v.double().var(0,unbiased=False).mean().sqrt().item() for v in anchor_blocks]
        scales = [max(v,1e-6) for v in scales_raw]
        def pair_distance(ma, ref):
            return (scaled_distance(ma,anchor_blocks[0],scales[0]) +
                    scaled_distance(ref[:,:dc],anchor_blocks[1],scales[1]) +
                    scaled_distance(ref[:,dc:],anchor_blocks[2],scales[2])).min(-1).values.sqrt()
        cal_train_dist, cal_eval_dist = pair_distance(cal_train['ma'],cal_train['reference']),pair_distance(cal_eval['ma'],cal_eval['reference'])
        dm = scaled_distance(current['m_a'].reshape(-1,current['m_a'].shape[-1]),anchor_blocks[0],scales[0])
        dr = scaled_distance(reference['reference'][:,:dc],anchor_blocks[1],scales[1])+scaled_distance(reference['reference'][:,dc:],anchor_blocks[2],scales[2])
        cross_dist = (dm[:,None,:]+dr[None,:,:]).min(-1).values.sqrt()
        own_dr = scaled_distance(own['reference'][:,:dc],anchor_blocks[1],scales[1])+scaled_distance(own['reference'][:,dc:],anchor_blocks[2],scales[2])
        own_dist = (dm.reshape(b,-1,args.anchors)+own_dr[:,None,:]).min(-1).values.sqrt()
        nearest = dict(block_scales=scales_raw, scale_floor=1e-6,
            training_factual_calibration=describe(cal_train_dist),eval_factual_calibration=describe(cal_eval_dist),
            query_factual_eval=describe(own_dist),query_cross_eval=describe(cross_dist),
            matched_cross_minus_factual=describe(cross_dist-own_dist.reshape(-1)[:,None]),
            cross_above_train_factual_p95_fraction=float((cross_dist>cal_train_dist.quantile(.95)).double().mean()),
            cross_above_eval_factual_p95_fraction=float((cross_dist>cal_eval_dist.quantile(.95)).double().mean()),
            note='全表示分块归一化最近邻距离，仅启发式，不认证高维支持/positivity。校准窗口不是新validation。')
        probs_do,_ = fd.marginalize(current['m_a'],reference['reference'][None].expand(b,-1,-1))
        probs_own,_ = fd.marginalize(current['m_a'],own['reference'][:,None])
        factual_input = {mode:dict(ma=activation_summary(value['ma']),reference_cns=activation_summary(value['reference'][:,:dc]),reference_ans=activation_summary(value['reference'][:,dc:])) for mode,value in factual.items()}
        reconstructed_cross = dict(
            train_mode=dict(ma=activation_summary(training_query['m_a'].flatten(0,1)),
                reference_cns=activation_summary(training_reference[:,:dc]),reference_ans=activation_summary(training_reference[:,dc:])),
            eval_mode=dict(ma=activation_summary(current['m_a'].flatten(0,1)),
                reference_cns=activation_summary(reference['reference'][:,:dc]),reference_ans=activation_summary(reference['reference'][:,dc:])),
            note='边缘摘要不检验联合支持；train cross为固定权重重建，不是假称保存了历史输入。只有事实头具有分类标签监督，这是loss静态审阅结论。')
    model.eval()
    print('支持诊断完成；开始只对OutcomeHead输入求局部梯度，无参数更新',flush=True)
    sens = dict(factual=local_sensitivity(fd.outcome,own['ma'],own['reference'],scales,dc),
        cross=local_sensitivity(fd.outcome,current['m_a'][:,0],reference['reference'][torch.arange(b)%args.K],scales,dc))
    if any(p.grad is not None for p in model.parameters()) or not all(torch.equal(before[k],v) for k,v in model.state_dict().items()):
        raise RuntimeError('参数梯度字段或模型state发生变化，停止')
    if file_hash(path)!=fingerprint:
        raise RuntimeError('原checkpoint文件发生变化')
    result = dict(operation='联合支持、CNS/ANS后验和OutcomeHead局部依赖审计', dataset=cfg['dataset'],task=cfg['arguments']['label'],fold=1,
        source_version=source_version(PROJECT), original_training_source=cfg.get('source_version'),checkpoint_epoch=20,checkpoint_sha256=fingerprint,
        settings=vars(args)|{'root_dir':str(args.root_dir),'fold_dir':str(args.fold_dir),'output_dir':str(args.output_dir)},
        sources=dict(anchors=train_ids.tolist(),distance_calibration=cal_ids.tolist(),query=current_ids.tolist(),reference=ref_ids.tolist()),
        selection=selection,factual_head_inputs=factual_input,reconstructed_cross_head_inputs=reconstructed_cross,
        posterior=sampling,density_support=density,joint_nearest_neighbor=nearest,
        local_head_sensitivity=sens,paired_probability=dict(
            factual_average_S=dict(mean=probs_own.double().mean(0).tolist(),std=probs_own.double().std(0,unbiased=False).tolist()),
            frontdoor_average_S_K=dict(mean=probs_do.double().mean(0).tolist(),std=probs_do.double().std(0,unbiased=False).tolist())),
        dimensions=dict(mediator=current['m_a'].shape[-1],reference_cns=dc,reference_ans=model.backbone.ans_dim),
        model_state_unchanged=True,checkpoint_file_unchanged=True,parameter_grad_fields_unchanged=True,optimizer_created=False,training_executed=False,
        note='重建的是固定权重下的输入分布，不是历史完整训练记录；有限q密度/NN通过不证明真实联合支持、条件结果拟合或因果识别充分。')
    content=json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)
    dest.mkdir(parents=True,exist_ok=True)
    with (dest/'probe_summary.json').open('x',encoding='utf-8') as stream:stream.write(content+'\n')
    print(f'完成：{dest/"probe_summary.json"}；没有训练或修改原权重',flush=True)


if __name__=='__main__':
    main()
