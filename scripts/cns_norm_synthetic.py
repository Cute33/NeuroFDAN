"""CNS入口归一化N0：已知Gaussian SCM配对训练，仅供服务器CPU执行。"""
import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def view(raw, oracle=False):
    import torch
    keys = ('x_c', 'x_a', 'y', 'm_c', 'm_a') if oracle else ('x_c', 'x_a', 'y')
    return {k: torch.as_tensor(raw[k].copy(), dtype=torch.long if k == 'y' else torch.float32) for k in keys}


def evaluation_view(raw):
    """独立测试可访问真实中介以评分条件恢复；不传给观察监督fit。"""
    return view(raw, oracle=True)


def fit(data, norm, oracle, steps, seed):
    import torch
    from torch.nn import functional as F
    from frontdoor_scm import FrontDoorConfig, FrontDoorEstimator, ReferenceBank
    from classification_objectives import classification_terms
    if set(data) != ({'x_c','x_a','y','m_c','m_a'} if oracle else {'x_c','x_a','y'}):
        raise ValueError('训练字段权限错误')
    torch.manual_seed(seed)
    cfg = FrontDoorConfig(mediator_dim=4, hidden_dim=16, lif_neurons=16, dropout=0.,
        S_train=4, K_train=8, S_eval=16, K_eval=32, N_ref=len(data['y']), cns_input_norm=norm)
    model = FrontDoorEstimator(16, 8, cfg)
    initial = hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest()
    opt = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=0.)
    ids = torch.tensor([[1, i, 0] for i in range(len(data['y']))])
    bank = ReferenceBank(data['x_c'], data['x_a'], ids, seed=seed)
    history = []
    for step in range(steps):
        torch.manual_seed(seed+step+1)
        idx = torch.randperm(len(ids))[:64]
        b = {k: v[idx] for k,v in data.items()}
        model.train(); opt.zero_grad(set_to_none=True)
        out = model(b['x_c'], b['x_a'], bank, ids[idx])
        terms = classification_terms(out, b['y'], cfg, epoch=1+step//4,
                                     objective='integrated_observational', lambda_obs=0.)
        fire = .5*((model.cns.snn.last_rates.mean(0)-.05).square().mean()+
                    (model.ans.snn.last_rates.mean(0)-.05).square().mean())
        if oracle:
            # 只有此分支访问真实中介；没有U或do概率进入训练函数。
            conditional = model.ans(b['x_a'], b['m_c'][:, None])
            nll = (model.cns.nll(b['m_c'], out['mu_c'], out['logvar_c'])+
                   model.ans.nll(b['m_a'], conditional['mu_a'][:,0], conditional['logvar_a'][:,0]))/4
            ce = F.cross_entropy(model.outcome(b['m_a'], model.reference_encoder(b['x_c'],b['x_a'])), b['y'])
            loss = ce+nll+terms['weighted_kl_loss']+.028*fire
        else:
            loss = terms['loss']+.028*fire
        if not torch.isfinite(loss): raise FloatingPointError('合成损失非有限')
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); opt.step()
        if step == 0 or (step+1)%50 == 0:
            history.append(dict(step=step+1, loss=float(loss.detach())))
    return model, bank, dict(initial_parameter_sha256=initial, config=asdict(cfg), history=history,
        training_keys=sorted(data), steps=steps, classification_supervision='oracle factual CE+mediator NLL' if oracle else 'integrated observation NLL')


def main():
    import numpy as np
    import torch
    from scipy.special import expit
    from run_reporting import source_version
    from ModelTrain import confusion_metrics
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output_dir', type=Path, required=True)
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--seed', type=int, choices=[42], default=42)
    args=p.parse_args()
    if args.steps < 1 or args.steps > 1000: p.error('steps须在1–1000')
    if args.output_dir.exists(): raise FileExistsError('拒绝覆盖')
    args.output_dir.mkdir(parents=True)
    torch.set_num_threads(1)
    rng=np.random.default_rng(2026)
    Ac=rng.normal(size=(2,16))*.3; Aa=rng.normal(size=(2,8))*.3
    Bc=rng.normal(size=(16,4))*.2; Ba=rng.normal(size=(8,4))*.2
    C=rng.normal(size=(4,4))*.3; w=rng.normal(size=4)*.5
    def sample(n, seed):
        r=np.random.default_rng(seed); u=r.normal(size=(n,2))
        xc=u@Ac+r.normal(size=(n,16)); xa=u@Aa+r.normal(size=(n,8))
        mc=xc@Bc+r.normal(size=(n,4)); ma=xa@Ba+mc@C+r.normal(size=(n,4))
        return dict(x_c=xc,x_a=xa,m_c=mc,m_a=ma,u=u,
                    y=(r.random(n)<expit(ma@w+.8*u[:,0])).astype('int64'))
    raw=sample(256,42); test=sample(128,1042)
    xc,xa=test['x_c'][:16],test['x_a'][:16]
    # 独立评价真值；此数组不传给fit。
    r=np.random.default_rng(2042); truth=[]; se=[]
    for c,a in zip(xc,xa):
        u=r.normal(size=(8192,2)); mc=c@Bc+r.normal(size=(8192,4))
        ma=a@Ba+mc@C+r.normal(size=(8192,4)); values=expit(ma@w+.8*u[:,0])
        truth.append(float(values.mean()));se.append(float(values.std(ddof=1)/np.sqrt(len(values))))
    result=dict(operation='CNS归一化N0小规模合成对照',completed=False,seed=42,
        source_version=source_version(Path(__file__).resolve().parents[1]), rows=[],
        scm=dict(Ac=Ac.tolist(),Aa=Aa.tolist(),Bc=Bc.tolist(),Ba=Ba.tolist(),C=C.tolist(),w=w.tolist(),
                 external_noises='independent standard Gaussian',outcome='Bernoulli(sigmoid(Ma@w+0.8*U0))'),
        train_size=256,test_size=128,batch_size=64,device='cpu',steps=args.steps,
        observation_data_sha256=hashlib.sha256(b''.join(raw[k].tobytes() for k in ('x_c','x_a','y'))).hexdigest(),
        direct_do_probability=truth,direct_do_mc_se=se,
        note='固定最后模型、同数据同初始化；oracle是额外信息参考，不保证有限训练下构成数值上界。')
    path=args.output_dir/'probe_summary.json'
    def save(): path.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    save()
    for name,norm,oracle in [('observation_only','none',False),('observation_norm','layernorm',False),('oracle_mediator','none',True)]:
        model,bank,row=fit(view(raw,oracle),norm,oracle,args.steps,42)
        model.eval(); t=evaluation_view(test); predictions=[]; do=[]
        with torch.no_grad():
            for repeat in range(3):
                torch.manual_seed(3042+repeat)
                predictions.append(model(t['x_c'],t['x_a'],bank)['probs_do'])
                do.append(model(t['x_c'][:16],t['x_a'][:16],bank)['probs_do'][:,1])
            probs=torch.stack(predictions).mean(0)
            cm=torch.bincount(2*t['y']+probs.argmax(-1),minlength=4).reshape(2,2)
            out=model(t['x_c'],t['x_a'],bank)
            am=model.ans(t['x_a'],t['m_c'][:,None])
            mean_c=t['x_c'].numpy()@Bc
            mean_a=t['x_a'].numpy()@Ba+t['m_c'].numpy()@C
            estimated=torch.stack(do).mean(0).numpy()
            row.update(name=name,metrics=confusion_metrics(cm,'DEAP'),confusion_matrix=cm.tolist(),
                observed_nll=float(-probs.gather(1,t['y'][:,None]).log().mean()),
                brier=float((probs[:,1]-t['y']).square().mean()),
                do_mae=float(np.abs(estimated-truth).mean()), do_mse=float(np.square(estimated-truth).mean()),
                do_mc_std_mean=float(torch.stack(do).std(0,unbiased=False).mean()),
                posterior_mu_c_std=float(out['mu_c'].std(0,unbiased=False).mean()),
                cns_conditional_mean_mse=float(np.square(out['mu_c'].numpy()-mean_c).mean()),
                ans_given_true_mc_conditional_mean_mse=float(np.square(am['mu_a'][:,0].numpy()-mean_a).mean()),
                conditional_nll_c=float(model.cns.nll(t['m_c'],out['mu_c'],out['logvar_c']))/4,
                conditional_nll_a=float(model.ans.nll(t['m_a'],am['mu_a'][:,0],am['logvar_a'][:,0]))/4)
        result['rows'].append(row);save()
        print(f'{name}: ACC={row["metrics"]["acc"]:.4f} F1={row["metrics"]["selection_f1"]:.4f} do_MAE={row["do_mae"]:.6f}',flush=True)
    if len({row['initial_parameter_sha256'] for row in result['rows']}) != 1:
        raise RuntimeError('初始化不一致')
    # 已知失效反例：沿全1方向改变CNS幅度，LayerNorm精确丢掉该自由度。
    pattern=np.arange(16,dtype=float)-7.5
    z=torch.tensor(np.stack([pattern,pattern+4.]),dtype=torch.float64)
    normed=torch.nn.functional.layer_norm(z,(16,),eps=1e-5)
    noise=np.random.default_rng(4042).normal(size=(8192,3))
    ps=[float(expit(.7*(a+noise[:,0])+noise[:,1]+.8*noise[:,2]).mean()) for a in (0.,4.)]
    result.update(completed=True,amplitude_counterexample=dict(
        normalized_input_max_difference=float((normed[0]-normed[1]).abs().max()),
        true_do_probability_mc_estimate=ps,estimated_max_error_lower_bound=abs(ps[0]-ps[1])/2,
        direct_n=8192,
        definition='观测时a=U+ex、Xa=U+exa、Xc=pattern+a*ones；Mc=mean(Xc)+ec，Ma=0.7Mc+0.2Xa+ea，Y~Bernoulli(sigmoid(Ma+0.8U))；do评价固定Xa=0、a=0或4；独立标准Gaussian外生量',
        note='固定Xa及参考分布后，归一化CNS入口不能区分这两个do点；这是幅度信息丢失反例，不是原paired SCM的观测等价反例'),
        automatically_start_real_training=False)
    from phase6_nonidentifiability import counterexample
    result['observational_nonidentifiability_counterexample'] = counterexample()
    save();print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__': main()
