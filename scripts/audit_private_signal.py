"""固定私有checkpoint的只读尺度/MC/Head依赖诊断；不创建优化器。"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT / 'scripts')]


def stats(x):
    x = x.detach().double().flatten(1)
    if not x.isfinite().all() or not x.numel():
        raise ValueError('统计量输入为空或非有限')
    return dict(rms=float(x.square().mean().sqrt()),
                between_sample_std_mean=float(x.std(0, unbiased=False).mean()),
                within_sample_dimension_std_mean=float(x.std(1, unbiased=False).mean()))


def variance_parts(values):
    """[R,B,C]，有限重复的总体方差恒等分解，不声称无偏输入信号估计。"""
    x = values.double()
    if x.ndim != 3 or min(x.shape[:2]) < 2 or not x.isfinite().all():
        raise ValueError('需要有限的[R>=2,B>=2,C]')
    between = x.mean(0).var(0, unbiased=False)
    within = x.var(0, unbiased=False).mean(0)
    return dict(finite_R_between_sample_mean_variance=between.tolist(),
                mean_within_sample_MC_variance=within.tolist(),
                total_variance=x.flatten(0, 1).var(0, unbiased=False).tolist(),
                note='between项仍有有限R误差；不能直接解释为纯输入信息')


def validate(payload):
    a = payload.get('arguments', {})
    if (payload.get('format_version'), payload.get('actual_steps'), a.get('dataset'),
        a.get('label'), a.get('held_out_subjects'), a.get('seed'), a.get('beta')) != (
        'neurofdan_private_fit_v1', 300, 'DEAP', 'arousal', [2], 42, 0.):
        raise ValueError('需要本次DEAP arousal fold2 beta0 step300私有权重')
    if payload.get('source_version', {}).get('commit') != 'bfe108e703b5d089fc53d59d254b61d6fcbbb885':
        raise ValueError('私有checkpoint训练源码不符')
    report = payload['reference_report']
    allowed = set(report['train_subjects'])
    if 2 in allowed or report['held_out_subject'] != 2:
        raise ValueError('训练总体身份错误')
    for key in ('selected_source_ids', 'reference_source_ids'):
        ids = [tuple(row) for row in payload[key]]
        if len(ids) != len(set(ids)) or any(row[0] not in allowed or row[0] == 2 for row in ids):
            raise ValueError('来源重复或包含held-out')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--root_dir', type=Path, required=True)
    p.add_argument('--output_dir', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    if args.output_dir.exists():
        raise FileExistsError('只写新目录，拒绝覆盖')
    args.output_dir.mkdir(parents=True)
    result = dict(operation='私有权重只读信号审查', completed=False, optimizer_steps=0,
                  dataset='DEAP', label='arousal', held_out_subject=2, seed=42,
                  samples=32, repeats=4, selection='已拟合400窗来源的前32个，不按标签选择',
                  note='训练子集局部诊断，不是LOSO、物理干预或因果识别证明')
    target = args.output_dir / 'probe_summary.json'
    (args.output_dir / 'config.json').write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2), encoding='utf-8')
    def save():
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    try:
        import torch
        from NeuroFDAN import NeuroFDAN
        from frontdoor_scm import FrontDoorConfig
        from ModelTrain import cache_reference_features, resolve_device, seed_everything
        from phase6_gradient_audit import labeled_windows
        from capacity_path_diagnostics import original_reference_bank, probability_summary
        from training_diagnostics import isolated_evaluation
        from run_reporting import source_version
        fingerprint = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
        payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        validate(payload)
        result['training_arguments'] = payload['arguments']
        seed_everything(42)
        device = resolve_device(args.device)
        model = NeuroFDAN(ecg_channels=2, frontdoor_config=FrontDoorConfig(**payload['frontdoor_config'])).to(device)
        model.load_state_dict(payload['model'], strict=True)
        before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        ids = torch.tensor(payload['selected_source_ids'][:32], dtype=torch.long)
        xc, xa, y = labeled_windows(args.root_dir, 'DEAP', 'arousal', ids)
        raw_payload = dict(payload, reference_source_ids=torch.tensor(payload['reference_source_ids'], dtype=torch.long))
        raw_bank = original_reference_bank(args.root_dir, raw_payload)
        result.update(source_version=source_version(PROJECT), checkpoint_sha256=fingerprint,
                      true_counts=torch.bincount(y, minlength=2).tolist(), query_ids=ids.tolist(),
                      S=model.frontdoor_config.S_eval, K=model.frontdoor_config.K_eval,
                      reference_members=raw_bank.used_count)
        xc, xa, y, ids = (x.to(device) for x in (xc, xa, y, ids))
        with isolated_evaluation(model):
            bank = cache_reference_features(model, raw_bank, device, batch_size=32)
            hc, ha = model.backbone(xc, xa)
            scales = dict(raw_cns=stats(xc), raw_ans=stats(xa), backbone_cns=stats(hc), backbone_ans=stats(ha))
            handles = []
            def capture(name):
                def hook(module, inputs, output):
                    if isinstance(output, tuple):
                        scales[name + '_rates'], scales[name + '_mean_current'] = (stats(t) for t in output)
                    else:
                        scales[name] = stats(output)
                return hook
            handles.append(model.frontdoor.cns.encoder.register_forward_hook(capture('cns_drive')))
            handles.append(model.frontdoor.cns.snn.register_forward_hook(capture('cns_lif')))
            probabilities, rows = [], []
            try:
                for repeat in range(4):
                    torch.manual_seed(42 + repeat)
                    out = model(xc, xa, reference_bank=bank, source_ids=ids, fd_mode='full')
                    probabilities.append(out['probs_do'].cpu())
                    row = probability_summary(out['probs_do'], y)
                    row.pop('probability')
                    rows.append(row)
                    if repeat == 0:
                        for h in handles: h.remove()
                        handles = []
                        result['scales'] = scales
                        lookup = {tuple(s): i for i, s in enumerate(bank.source_ids.cpu().tolist())}
                        positions = torch.tensor([[lookup[tuple(s)] for s in r] for r in out['reference_indices'].cpu().tolist()], device=device)
                        b, k = positions.shape
                        refs = model.frontdoor.reference_encoder(bank.x_c[positions].reshape(b*k, -1), bank.x_a[positions].reshape(b*k, -1)).reshape(b, k, -1)
                        original = model.frontdoor.marginalize(out['m_a'], refs)[0]
                        if not torch.allclose(original, out['probs_do'], atol=1e-6, rtol=1e-6):
                            raise RuntimeError('未重现原前门参考组合')
                        result['head_sensitivity'] = {
                            'rotate_ma_fixed_ref_l1': float((model.frontdoor.marginalize(out['m_a'].roll(1, 0), refs)[0]-original).abs().sum(-1).mean()),
                            'rotate_ref_fixed_ma_l1': float((model.frontdoor.marginalize(out['m_a'], refs.roll(1, 0))[0]-original).abs().sum(-1).mean()),
                            'note': '固定抽样的函数敏感度，含供体噪声；置换配对不作为合法因果干预或正式预测'}
                        mu, sigma = out['mu_c'], (.5*out['logvar_c']).exp()
                        eps = ((out['m_c'][0]-mu[0])/sigma[0])[None]
                        common_mc = mu[:, None] + sigma[:, None]*eps
                        s = common_mc.shape[1]
                        ans = model.frontdoor.ans
                        def conditional(mc):
                            extra = ans.cns_current(mc).reshape(b*s, -1)
                            am, av = ans.distribution(ha[:, None].expand(b, s, *ha.shape[1:]).reshape(b*s, -1), extra)
                            am, ast = am.reshape(b, s, -1), (.5*av).exp().reshape(b, s, -1)
                            return dict(mean_of_conditional_mu=stats(am.mean(1)), mean_of_conditional_sigma=stats(ast.mean(1)),
                                        mean_within_input_mu_std_across_Mc_draws=float(am.double().std(1, unbiased=False).mean()))
                        result['ans_dependence'] = dict(common_cns_standard_noise=conditional(common_mc),
                            fixed_zero_mc=conditional(torch.zeros_like(common_mc)),
                            note='共同噪声减少输入间独立噪声差异；零Mc是函数输入检查，可能偏离支持域，非生理干预')
                        result['cns_posterior'] = dict(mu=stats(mu), sigma=stats(sigma))
                    print(f'只读MC={repeat+1}/4 CM={row["confusion_matrix"]} NLL={row["observed_label_nll"]:.6f}', flush=True)
            finally:
                for h in handles: h.remove()
            result.update(monte_carlo=rows, variance_decomposition=variance_parts(torch.stack(probabilities)))
        if any(not torch.equal(before[k], v.detach().cpu()) for k, v in model.state_dict().items()):
            raise RuntimeError('参数或BN缓冲区改变')
        if any(p.grad is not None for p in model.parameters()):
            raise RuntimeError('只读推理产生了参数梯度')
        if hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() != fingerprint:
            raise RuntimeError('原checkpoint文件发生变化')
        result.update(completed=True, model_state_unchanged=True, checkpoint_unchanged=True)
        save()
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    except BaseException as error:
        result.update(error_type=type(error).__name__, error=str(error))
        save()
        raise


if __name__ == '__main__':
    main()
