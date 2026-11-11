"""固定小批已拟合模型的末端路径诊断；不提供新训练目标或SCM边。"""
import hashlib


def original_reference_bank(root, payload):
    """只读取checkpoint原参考成员，完整保留训练总体经验库，不读取测试被试。"""
    import torch
    from frontdoor_scm import ReferenceBank
    from scripts.phase6_gradient_audit import labeled_windows
    report = payload['reference_report']
    ids = payload['reference_source_ids'].cpu()
    if ids.shape != (report['actual_N_ref'], 3) or len(torch.unique(ids, dim=0)) != len(ids):
        raise ValueError('原参考成员数量或唯一性错误')
    subjects = set(ids[:, 0].tolist())
    if report['held_out_subject'] in subjects or not subjects <= set(report['train_subjects']):
        raise ValueError('原参考库含测试或非训练来源')
    xc, xa, _ = labeled_windows(root, 'DEAP', 'arousal', ids)
    bank = ReferenceBank(xc, xa, ids, forbidden_subjects=(report['held_out_subject'],),
                         seed=42, dataset='DEAP')
    if {tuple(x) for x in bank.source_ids.tolist()} != {tuple(x) for x in ids.tolist()}:
        raise RuntimeError('参考经验库成员被改变')
    bank.raw_windows = True
    bank.population_count = report['population_windows']
    bank.report = dict(report)
    return bank


def probability_summary(probs, y):
    """同批诊断分数，不是独立测试或泛化性能。"""
    import torch
    from ModelTrain import confusion_metrics
    if probs.shape != (len(y), 2) or not torch.isfinite(probs).all() or not ((probs >= 0) & (probs <= 1)).all():
        raise FloatingPointError('路径诊断概率形状或范围无效')
    if not torch.allclose(probs.sum(-1), torch.ones_like(probs[:, 0]), atol=1e-5, rtol=1e-5):
        raise FloatingPointError('路径诊断概率未归一化')
    cm = torch.bincount(2*y+probs.argmax(-1), minlength=4).reshape(2, 2)
    return dict(metrics=confusion_metrics(cm, 'DEAP'), confusion_matrix=cm.tolist(),
        probability=probs.tolist(), mean_probability=probs.double().mean(0).tolist(),
        std_probability=probs.double().std(0, unbiased=False).tolist(),
        observed_label_nll=float(-probs.gather(1, y[:, None]).clamp_min(1e-7).log().mean()),
        note='对观察标签的局部诊断评分，不是干预真值误差。')


def outcome_path_sensitivity(estimator, ma, own_ref, sampled_ref, y):
    """固定中介噪声与参考，循环移位输入测函数依赖；不是条件独立性检验。"""
    import torch
    if ma.ndim != 3 or own_ref.ndim != 2 or sampled_ref.ndim != 3:
        raise ValueError('需要[B,S,D]、[B,r]与[B,K,r]')
    b = len(ma)
    if len(own_ref) != b or len(sampled_ref) != b or len(y) != b or b < 2:
        raise ValueError('样本轴不一致或不足两个')
    own = estimator.marginalize(ma, own_ref[:, None])[0]
    cross = estimator.marginalize(ma, sampled_ref)[0]
    summaries = {}
    for label in ('rotate_ma_own_ref', 'rotate_ref_fixed_ma', 'rotate_ma_cross_ref'):
        total_l1 = cross_l1 = donor_correct = own_correct = count = opposite = 0.
        # K次积分较贵，只对固定的一次跨样本移位检查；自身参考两项遍历所有移位。
        shifts = list(range(1, b)) if label != 'rotate_ma_cross_ref' else [1]
        for shift in shifts:
            donor = torch.roll(torch.arange(b, device=y.device), shift)
            mask = y != y[donor]
            if label == 'rotate_ma_own_ref':
                p = estimator.marginalize(ma[donor], own_ref[:, None])[0]
                baseline = own
            elif label == 'rotate_ref_fixed_ma':
                p = estimator.marginalize(ma, own_ref[donor, None])[0]
                baseline = own
            else:
                p = estimator.marginalize(ma[donor], sampled_ref)[0]
                baseline = cross
            delta = (p-baseline).abs().sum(-1)
            total_l1 += float(delta.sum())
            cross_l1 += float(delta[mask].sum())
            donor_correct += float((p.argmax(-1)[mask] == y[donor][mask]).sum())
            own_correct += float((p.argmax(-1)[mask] == y[mask]).sum())
            count += b
            opposite += float(mask.sum())
        summaries[label] = dict(mean_probability_l1=total_l1/count,
            shifts_evaluated=shifts,
            opposite_label_pairs=int(opposite),
            opposite_label_probability_l1=cross_l1/opposite if opposite else None,
            opposite_label_donor_agreement=donor_correct/opposite if opposite else None,
            opposite_label_own_accuracy=own_correct/opposite if opposite else None)
    return dict(own_reference_same_S=probability_summary(own, y),
                empirical_reference_do=probability_summary(cross, y), sensitivity=summaries)


def fitted_path_check(model, batch, raw_bank, repeats):
    """只在私有模型拟合结束后执行；保留参数、BN缓冲区、已有梯度与随机流。"""
    import torch
    from ModelTrain import cache_reference_features
    was_training = model.training
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    gradients = {k: None if p.grad is None else p.grad.detach().clone() for k, p in model.named_parameters()}
    rows = []
    try:
        with torch.no_grad(), torch.random.fork_rng(devices=[]):
            model.eval()
            bank = cache_reference_features(model, raw_bank, torch.device('cpu'), batch_size=40)
            hc, ha = model.backbone(batch['x_c'], batch['x_a'])
            own_ref = model.frontdoor.reference_encoder(hc, ha)
            for repeat in range(repeats):
                torch.manual_seed(92042+repeat)
                out = model(batch['x_c'], batch['x_a'], reference_bank=bank,
                            source_ids=batch['source_ids'], fd_mode='full')
                ids = out['reference_indices']
                if (ids == batch['source_ids'][:, None]).all(-1).any():
                    raise RuntimeError('积分参考没有排除查询自身')
                lookup = {tuple(source): i for i, source in enumerate(bank.source_ids.tolist())}
                positions = torch.tensor([[lookup[tuple(source)] for source in row] for row in ids.tolist()])
                b, k = positions.shape
                refs = model.frontdoor.reference_encoder(bank.x_c[positions].reshape(b*k, *bank.x_c.shape[1:]),
                         bank.x_a[positions].reshape(b*k, *bank.x_a.shape[1:])).reshape(b, k, -1)
                row = outcome_path_sensitivity(model.frontdoor, out['m_a'], own_ref, refs, batch['Labels'])
                if not torch.allclose(torch.tensor(row['empirical_reference_do']['probability']), out['probs_do'], atol=1e-6, rtol=1e-6):
                    raise RuntimeError('诊断未复用原前门组合和概率积分')
                # 保持Xa、结果Head、参考和同一ANS标准噪声，只替换Mc输入。
                donor = torch.roll(torch.arange(b), 1)
                eps_a = (out['m_a']-out['mu_a'])/(.5*out['logvar_a']).exp()
                changed = model.frontdoor.ans(ha, out['m_c'][donor])
                ma = changed['mu_a']+(.5*changed['logvar_a']).exp()*eps_a
                altered = model.frontdoor.marginalize(ma, refs)[0]
                row.update(repeat=repeat, S=model.frontdoor_config.S_eval, K=model.frontdoor_config.K_eval,
                    cns2ans_fixed_noise=dict(
                        ans_mean_delta_rms=float((changed['mu_a']-out['mu_a']).double().square().mean().sqrt()),
                        ans_sigma_delta_rms=float(((.5*changed['logvar_a']).exp()-(.5*out['logvar_a']).exp()).double().square().mean().sqrt()),
                        do_probability_l1=float((altered-out['probs_do']).abs().sum(-1).mean())),
                    reference_indices_sha256=hashlib.sha256(ids.numpy().tobytes()).hexdigest())
                rows.append(row)
                print(f'末端路径MC={repeat+1}/{repeats} '
                      f'自身参考同S的ACC={row["own_reference_same_S"]["metrics"]["acc"]:.4f} '
                      f'经验参考do的ACC={row["empirical_reference_do"]["metrics"]["acc"]:.4f} '
                      f'固定Ma换自身参考L1={row["sensitivity"]["rotate_ref_fixed_ma"]["mean_probability_l1"]:.6f}', flush=True)
    finally:
        model.train(was_training)
    if any(not torch.equal(before[k], v) for k, v in model.state_dict().items()):
        raise RuntimeError('末端路径诊断改变了参数或缓冲区')
    for name, p in model.named_parameters():
        old = gradients[name]
        if (old is None) != (p.grad is None) or (old is not None and not torch.equal(old, p.grad)):
            raise RuntimeError('末端路径诊断改变了已有梯度')
    return dict(records=rows, model_state_unchanged=True, existing_gradients_unchanged=True,
        reference_members=raw_bank.used_count, reference_subjects=sorted(set(raw_bank.source_ids[:, 0].tolist())),
        note='同批函数依赖诊断，不是物理干预或SCM条件独立性检验；置换组合不解释为新的合法干预估计。')
