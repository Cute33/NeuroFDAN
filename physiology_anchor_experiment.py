"""生理统计锚点的独立配对合成验证；真实训练器保持原样。"""
import argparse
import csv
import hashlib
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from anchor_synthetic_scm import AnchorSyntheticSCM, coefficient_anchors
from frontdoor_scm import FrontDoorConfig, FrontDoorEstimator
from physiological_anchors import PhysiologicalAnchorReadouts
from phase6_learning import LearningWeights, ModalityReadouts, gradient_components, learning_terms
from phase6_nonidentifiability import counterexample
from phase6_synthetic_experiment import save_json, training_view, data_digest, evaluate
from run_reporting import source_version
from synthetic_scm import probability_metrics, SCMConfig

CONDITIONS = ('oracle_mediator', 'observation_only', 'observation_physiological_anchors')


def anchor_view(raw: dict) -> dict:
    """从本样本X计算观测target，训练视图中不持有真实M/U/干预概率。"""
    data = training_view(raw, 'observation_only')
    targets = coefficient_anchors(raw['x_c'], raw['x_a'])
    data.update({key:targets[key] for key in ('c','a','mask_c','mask_a')})
    return data


def validate_view(data: dict, condition: str):
    """严格白名单；候选组只多出由输入派生的统计量及有效掩码。"""
    expected = {'x_c','x_a','y'}
    if condition == 'oracle_mediator':
        expected |= {'m_c','m_a'}
    elif condition == 'observation_physiological_anchors':
        expected |= {'c','a','mask_c','mask_a'}
    elif condition != 'observation_only':
        raise ValueError('未知训练条件')
    if set(data) != expected or len(data['y']) < 2 or any(len(v) != len(data['y']) for v in data.values()):
        raise ValueError('训练字段权限或样本配对不符')


def anchor_learning_terms(model, readouts, legacy_readouts, data, condition, epoch, alpha_c, alpha_a):
    """既有CE/KL/fire不变；两条训练读出只增加明确命名的锚点工作损失。"""
    validate_view(data, condition)
    keys = ('x_c','x_a','y','m_c','m_a') if condition == 'oracle_mediator' else ('x_c','x_a','y')
    base_condition = 'oracle_mediator' if condition == 'oracle_mediator' else 'observation_only'
    terms, out = learning_terms(model, legacy_readouts, {k:data[k] for k in keys}, base_condition, epoch,
                                LearningWeights(fidelity=0., fire=0.))
    zero = out['logits_obs'].new_zeros(())
    lc, la = readouts.losses(out, data) if condition == 'observation_physiological_anchors' else (zero, zero)
    terms['anchor_c'], terms['anchor_a'] = alpha_c*lc, alpha_a*la
    terms['total'] = terms['total']+terms['anchor_c']+terms['anchor_a']
    out['phase6_unweighted_losses'].update(anchor_c=lc, anchor_a=la)
    return terms, out


def fit(data, targets, condition, args, directory, mechanism, gamma):
    """只接收训练视图；固定最后轮评价，不按任何测试值选epoch/alpha。"""
    validate_view(data, condition)
    torch.manual_seed(args.seed)
    cfg = FrontDoorConfig(mediator_family=args.mediator_family, reference_chunk_size=4,
                          no_cns2ans=args.no_cns2ans)
    model = FrontDoorEstimator(16, 8, cfg).cpu()
    # 各组都以相同次序初始化读出，避免初始化消耗随机流产生不同baseline。
    readouts = PhysiologicalAnchorReadouts(64, targets)
    legacy = ModalityReadouts(64, data['x_c'], data['x_a'])
    named = list(model.named_parameters())+[('readouts.'+k,v) for k,v in readouts.named_parameters()]
    optimizer = torch.optim.Adam([p for _,p in named], lr=args.lr, weight_decay=0.)
    initial = hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest()
    order = torch.Generator().manual_seed(args.seed)
    started = time.perf_counter()
    for epoch in range(1, args.epochs+1):
        model.train(); readouts.train()
        weighted, unweighted, n, gradients = {}, {}, 0, None
        for j, indices in enumerate(torch.randperm(len(data['y']), generator=order).split(args.batch_size)):
            if len(indices) < 2:
                raise ValueError('合成规模应使每批至少2个样本，不静默丢弃尾样本')
            torch.manual_seed(args.seed+epoch*10000+j)
            batch = {k:v[indices] for k,v in data.items()}
            optimizer.zero_grad(set_to_none=True)
            terms, out = anchor_learning_terms(model, readouts, legacy, batch, condition, epoch, args.alpha_c, args.alpha_a)
            if not all(torch.isfinite(v) for v in terms.values()):
                raise FloatingPointError('锚点合成损失非有限')
            if j == 0:
                gradients = gradient_components(terms, named)
            terms['total'].backward()
            torch.nn.utils.clip_grad_norm_([p for _,p in named], 1.)
            optimizer.step()
            for key, value in terms.items():
                weighted[key] = weighted.get(key, 0.)+float(value.detach())*len(indices)
            for key, value in out['phase6_unweighted_losses'].items():
                unweighted[key] = unweighted.get(key, 0.)+float(value.detach())*len(indices)
            n += len(indices)
        row = dict(mechanism=mechanism, gamma=gamma, condition=condition, epoch=epoch, count=n,
                   weighted_losses={k:v/n for k,v in weighted.items()},
                   unweighted_losses={k:v/n for k,v in unweighted.items()}, first_batch_gradients=gradients,
                   alpha_c=args.alpha_c, alpha_a=args.alpha_a,
                   effective_beta=cfg.beta*min(epoch/cfg.beta_warmup_epochs, 1.))
        with (directory/'history.jsonl').open('a', encoding='utf-8') as stream:
            import json
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
        print(f'{mechanism} gamma={gamma} {condition} epoch={epoch}/{args.epochs} '
              f'CE={row["unweighted_losses"]["factual_ce"]:.6f} '
              f'锚点={row["unweighted_losses"]["anchor_c"]:.6f}/{row["unweighted_losses"]["anchor_a"]:.6f}', flush=True)
    model.eval(); readouts.eval()
    model_dir = directory/'models'; model_dir.mkdir(exist_ok=True)
    torch.save(dict(model=model.state_dict(), readouts=readouts.state_dict(), config=asdict(cfg),
                    condition=condition, epoch=args.epochs, alpha_c=args.alpha_c, alpha_a=args.alpha_a),
               model_dir/f'{mechanism}_gamma{gamma}_{condition}_last.pt')
    return model, dict(condition=condition, initial_model_sha256=initial, training_data_sha256=data_digest(data),
                       training_keys=sorted(data), fit_seconds=time.perf_counter()-started, config=asdict(cfg))


@torch.no_grad()
def posterior_features(model, data, S=64):
    """共同噪声估计E[M|X]用于同等线性探针，非真实中介测量。"""
    model.eval()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(90042)
        ec, ea = torch.randn(1,S,64), torch.randn(1,S,64)
        c, a = [], []
        for start in range(0, len(data['x_c']), 64):
            xc, xa = data['x_c'][start:start+64], data['x_a'][start:start+64]
            mu, lv = model.cns.distribution(xc)
            mc = mu[:,None]+(.5*lv[:,None]).exp()*ec
            if model.config.mediator_family == 'tanh_normal': mc = mc.tanh()
            ans = model.ans(xa, mc)
            ma = ans['mu_a']+(.5*ans['logvar_a']).exp()*ea
            if model.config.mediator_family == 'tanh_normal': ma = ma.tanh()
            c.append(mc.mean(1)); a.append(ma.mean(1))
        return dict(c=torch.cat(c), a=torch.cat(a))


def anchor_probe(model, train, observed, training_targets):
    """所有组用相同训练X上的ridge探针，测试不拟合、不挑超参，lambda=1。

    避免用候选组训练过的读出对比baseline随机读出，造成不公平可解释性结论。
    """
    test = training_view(observed, 'observation_only')
    test_targets = coefficient_anchors(observed['x_c'], observed['x_a'])
    z_train, z_test = posterior_features(model, train), posterior_features(model, test)
    result = {}
    from physiological_anchors import training_statistics
    for side in ('c','a'):
        stats = training_statistics(training_targets[side], training_targets['mask_'+side])
        train_y = ((training_targets[side]-stats['mean'])/stats['scale']).double()
        test_y = ((test_targets[side]-stats['mean'])/stats['scale']).double()
        z = z_train[side].double(); z0 = z.mean(0); scale = z.std(0, unbiased=False).clamp_min(1e-6)
        z, zz = (z-z0)/scale, (z_test[side].double()-z0)/scale
        rows = []
        for d in range(train_y.shape[1]):
            use = training_targets['mask_'+side][:,d]
            valid = test_targets['mask_'+side][:,d]
            if not stats['active'][d] or valid.sum() < 2:
                rows.append(dict(active=False)); continue
            center = train_y[use,d].mean()
            weights = torch.linalg.solve(z[use].T@z[use]/use.sum()+torch.eye(64, dtype=torch.float64),
                                         z[use].T@(train_y[use,d]-center)/use.sum())
            pred = zz[valid]@weights+center
            target = test_y[valid,d]
            mse = (pred-target).square().mean()
            variance = (target-target.mean()).square().mean()
            rows.append(dict(active=True, normalized_mse=float(mse),
                             r2=float(1-mse/variance) if variance > 1e-12 else None))
        active = [r['normalized_mse'] for r in rows if r['active']]
        result[side] = dict(normalized_mse=float(np.mean(active)) if active else None, per_feature=rows)
    return result


def export_tables_and_plots(directory, results, plots):
    """图表从同一数值摘要生成；阴影/误差不冒充多训练种子置信区间。"""
    rows = []
    for setting in results['settings']:
        for c in setting['conditions']:
            e = c['evaluation']; row = dict(mechanism=setting['mechanism'], gamma=setting['gamma'], condition=c['condition'],
                do_mse=e['interventional_mean_probability']['probability_mse'],
                do_mae=e['interventional_mean_probability']['mae'],
                constant_mse=e['null_training_prevalence_error']['probability_mse'],
                factual_acc=e['observed_factual']['acc'], factual_macro_f1=e['observed_factual']['macro_f1'],
                anchor_c_mse=c['anchor_probe']['c']['normalized_mse'], anchor_a_mse=c['anchor_probe']['a']['normalized_mse'])
            rows.append(row)
    with (directory/'metrics.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    if not plots: return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # 使用固定英文缩写保证Linux服务器无中文字体时仍可导出；说明/标题释义见文档。
    labels = {'oracle_mediator':'Oracle-M', 'observation_only':'Obs', 'observation_physiological_anchors':'Obs+Anchor'}
    for s in results['settings']:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        cs = s['conditions']; names = [labels[c['condition']] for c in cs]
        axes[0].bar(names, [c['evaluation']['interventional_mean_probability']['probability_mse'] for c in cs])
        axes[0].axhline(cs[0]['evaluation']['null_training_prevalence_error']['probability_mse'], color='black', ls='--', label='Prevalence')
        axes[0].set_ylabel('Interventional probability MSE'); axes[0].legend()
        ix = np.arange(len(cs))
        axes[1].bar(ix-.2, [c['anchor_probe']['c']['normalized_mse'] for c in cs], .4, label='CNS')
        axes[1].bar(ix+.2, [c['anchor_probe']['a']['normalized_mse'] for c in cs], .4, label='ANS')
        axes[1].set_xticks(ix, names); axes[1].set_ylabel('Held-out anchor probe MSE'); axes[1].legend()
        for c in cs:
            p = np.mean(c['evaluation']['independent_probability_per_repeat'], axis=0)
            axes[2].scatter(s['direct_probability'], p, label=labels[c['condition']])
        axes[2].plot([0,1], [0,1], color='black', ls='--'); axes[2].set_xlabel('True do probability'); axes[2].set_ylabel('Estimated'); axes[2].legend()
        prefix = 'SMOKE: interface only | ' if results['smoke'] else ''
        fig.suptitle(f'{prefix}{s["mechanism"]}, gamma={s["gamma"]}'); fig.tight_layout()
        stem = f'anchors_{s["mechanism"]}_gamma{s["gamma"]}'
        fig.savefig(directory/(stem+'.pdf')); fig.savefig(directory/(stem+'.png'), dpi=200); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description='生理锚点三组配对合成；不接入真实LOSO训练')
    p.add_argument('--output_dir', type=Path, required=True, help='新的phase6结果根目录')
    p.add_argument('--run_name', required=True, help='唯一运行名，不覆盖')
    p.add_argument('--seed', type=int, choices=[42], default=42, help='固定训练种子')
    p.add_argument('--device', choices=['cpu'], default='cpu', help='独立CPU合成试验')
    p.add_argument('--workers', type=int, choices=[0], default=0, help='顺序张量读取')
    p.add_argument('--diagnostic_level', choices=['full'], default='full', help='分项损失梯度与独立评价')
    p.add_argument('--epochs', type=int, default=20, help='固定最后轮，不按测试真值选模')
    p.add_argument('--batch_size', type=int, default=128, help='三组全局打乱，不做平衡')
    p.add_argument('--train_size', type=int, default=2048, help='自然训练样本数')
    p.add_argument('--test_size', type=int, default=512, help='独立测试样本数')
    p.add_argument('--points', type=int, default=16, help='相同联合干预点数')
    p.add_argument('--gammas', type=float, nargs='+', default=[0.,1.], help='混杂强度')
    p.add_argument('--mechanisms', choices=['aligned','misaligned'], nargs='+', default=['aligned','misaligned'], help='有利及遗漏信息的机制，必须如实报告')
    p.add_argument('--lr', type=float, default=.001, help='独立合成Adam学习率，不改真实默认')
    p.add_argument('--alpha_c', type=float, default=.1, help='预设CNS锚点权重，零为工作损失消融')
    p.add_argument('--alpha_a', type=float, default=.1, help='预设ANS锚点权重')
    p.add_argument('--mediator_family', choices=['normal','tanh_normal'], default='normal', help='高斯实际模型/有界分布匹配控制')
    p.add_argument('--no_cns2ans', action='store_true', help='仅估计器路径消融，不改生成机制')
    p.add_argument('--mc_repeats', type=int, default=3, help='同一模型积分重复，非训练种子重复')
    p.add_argument('--direct_n', type=int, default=32768, help='直接干预MC规模')
    p.add_argument('--oracle_s', type=int, default=1024, help='独立前门真值S')
    p.add_argument('--oracle_k', type=int, default=512, help='独立前门真值K')
    p.add_argument('--smoke', action='store_true', help='标注仅接口检查，不能解释成方法效果')
    p.add_argument('--plots', action='store_true', help='导出本次全部条件的CSV、PNG及PDF')
    args = p.parse_args()
    if min(args.epochs,args.points,args.mc_repeats) < 1 or min(args.train_size,args.test_size,args.batch_size,args.direct_n,args.oracle_s,args.oracle_k) < 2 or args.train_size % args.batch_size == 1:
        p.error('规模无效，或训练尾批仅1个样本')
    if not all(math.isfinite(x) for x in [args.lr,args.alpha_c,args.alpha_a,*args.gammas]) or args.lr <= 0 or min(args.alpha_c,args.alpha_a,*args.gammas) < 0:
        p.error('权重、学习率和混杂强度必须有限且合法')
    if len(set(args.mechanisms)) != len(args.mechanisms) or len(set(args.gammas)) != len(args.gammas):
        p.error('机制/混杂强度不得重复')
    if args.smoke and (args.epochs > 2 or args.train_size > 512 or args.test_size > 128 or args.points > 4):
        p.error('smoke限2轮/512训练/128测试/4个干预点')
    directory = args.output_dir/args.run_name
    if any((directory/f).exists() for f in ('config.json','probe_summary.json','history.jsonl')):
        raise FileExistsError('结果目录已有产物，不覆盖；允许日志工具预先创建train.log')
    directory.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1); torch.manual_seed(args.seed)
    config = dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                  source_version=source_version(Path(__file__).resolve().parent), conditions=list(CONDITIONS),
                  supervision='候选只用X/Y及从X计算的锚点；真实M/U/do概率不传入fit',
                  interpretation='合成系数单射渲染到波形；模型读取完整系数而非真实骨干，无临床信号真实性保证',
                  defaults=asdict(FrontDoorConfig()), scm=asdict(SCMConfig()))
    save_json(directory/'config.json', config)
    results = dict(operation='生理锚点配对合成验证', source_version=config['source_version'], completed=False,
                   smoke=args.smoke, real_data_training=False, training_executed=False,
                   counterexample=counterexample(), settings=[], config=config)
    save_json(directory/'probe_summary.json', results)
    for mechanism in args.mechanisms:
        scm = AnchorSyntheticSCM(mechanism)
        scm.save(directory/f'scm_{mechanism}.npz')
        for gamma in args.gammas:
            raw = scm.sample(args.train_size, gamma, args.seed+1000)
            observed = scm.sample(args.test_size, gamma, args.seed+2000)
            points = scm.sample(args.points, 0., 78901)
            started = time.perf_counter()
            truth, se = scm.direct_intervention(points['x_c'], points['x_a'], gamma, args.direct_n, args.seed+3000)
            direct_seconds = time.perf_counter()-started; started = time.perf_counter()
            oracle, ose = scm.frontdoor_integral(points['x_c'], points['x_a'], gamma, args.oracle_s, args.oracle_k, args.seed+4000)
            setting = dict(mechanism=mechanism, gamma=gamma, direct_probability=truth.tolist(), direct_se=se.tolist(),
                           oracle_probability=oracle.tolist(), oracle_approx_se=ose.tolist(),
                           direct_seconds=direct_seconds, oracle_seconds=time.perf_counter()-started,
                           truth_comparison=probability_metrics(oracle,truth), conditions=[])
            results['settings'].append(setting); save_json(directory/'probe_summary.json', results)
            if np.any(abs(truth-oracle) > 4*np.sqrt(se**2+ose**2)+.01):
                raise RuntimeError('直接/前门真值超出工程容差，已保存结果；停止拟合')
            av = anchor_view(raw); targets = {k:av[k] for k in ('c','a','mask_c','mask_a')}
            for condition in CONDITIONS:
                data = av if condition == CONDITIONS[2] else training_view(raw, condition)
                results['training_executed'] = None
                results['current_fit'] = dict(mechanism=mechanism, gamma=gamma, condition=condition)
                save_json(directory/'probe_summary.json', results)
                model, record = fit(data, targets, condition, args, directory, mechanism, gamma)
                record['evaluation'] = evaluate(model, training_view(raw,'observation_only'), observed, points, truth, oracle, scm, gamma, args)
                record['anchor_probe'] = anchor_probe(model, training_view(raw,'observation_only'), observed, targets)
                setting['conditions'].append(record)
                results['training_executed'] = True; save_json(directory/'probe_summary.json', results)
                del model
            if len({r['initial_model_sha256'] for r in setting['conditions']}) != 1 or len({r['training_data_sha256'] for r in setting['conditions']}) != 1:
                raise RuntimeError('配对初始化或X/Y不一致，不能解释组间结果')
    export_tables_and_plots(directory, results, args.plots)
    results['completed'] = True; results.pop('current_fit', None)
    save_json(directory/'probe_summary.json', results)
    print(f'完成：{directory}/probe_summary.json；smoke={args.smoke}，不等于真实数据实验完成', flush=True)


if __name__ == '__main__':
    main()
