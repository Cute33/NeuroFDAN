"""三种监督权限的配对合成验证；只供服务器执行，不接入DEAP/HCI训练。"""
import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from frontdoor_scm import FrontDoorConfig, FrontDoorEstimator, ReferenceBank
from phase6_learning import CONDITIONS, LearningWeights, ModalityReadouts, check_training_view, gradient_components, learning_terms
from phase6_nonidentifiability import counterexample
from synthetic_scm import SyntheticSCM, probability_metrics
from run_reporting import source_version


def save_json(path: Path, value: dict) -> None:
    """拒绝NaN/Inf，临时文件替换仅限本次独立输出目录。"""
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(path)


def training_view(raw: dict, condition: str) -> dict:
    """复制白名单字段；fit函数不接收生成器、U、测试真值或原始raw对象。"""
    keys = ['x_c','x_a','y']+(['m_c','m_a'] if condition=='oracle_mediator' else [])
    data = {key:torch.tensor(raw[key].copy(),dtype=torch.long if key=='y' else torch.float32) for key in keys}
    for key in ('m_c','m_a'):
        if key in data:
            data[key] = data[key].clamp(-1+1e-6,1-1e-6)
    check_training_view(data,condition)
    return data


def data_digest(data: dict) -> str:
    """相同X/Y字节指纹用于三组配对核验，不将真实数据上传。"""
    digest = hashlib.sha256()
    for key in ('x_c','x_a','y'):
        digest.update(key.encode());digest.update(data[key].contiguous().numpy().tobytes())
    return digest.hexdigest()


def fit(data: dict, condition: str, args, directory: Path, gamma: float):
    """固定轮数最后模型；从不接收评估数据或按真值/测试成绩选择模型。"""
    check_training_view(data,condition)
    torch.manual_seed(args.seed)
    cfg = FrontDoorConfig(mediator_family=args.mediator_family,reference_chunk_size=4,no_cns2ans=args.no_cns2ans)
    model = FrontDoorEstimator(data['x_c'].shape[1],data['x_a'].shape[1],cfg).cpu()
    readouts = ModalityReadouts(cfg.mediator_dim,data['x_c'],data['x_a'])
    weights = LearningWeights(fidelity=args.fidelity_weight,fire=args.fire_weight)
    named = list(model.named_parameters())+ [('readouts.'+k,v) for k,v in readouts.named_parameters()]
    optimizer = torch.optim.Adam([p for _,p in named],lr=args.lr,weight_decay=args.weight_decay)
    initial = hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest()
    started = time.perf_counter()
    generator = torch.Generator().manual_seed(args.seed)
    for epoch in range(1,args.epochs+1):
        model.train();readouts.train()
        sums, raw_sums, count, first_gradients = {},{},0,None
        for batch_number, indices in enumerate(torch.randperm(len(data['y']),generator=generator).split(args.batch_size)):
            if len(indices)<2:
                continue
            # 三组相同窗口顺序及每批初始随机流；额外oracle计算不推进后续批次随机流。
            torch.manual_seed(args.seed+epoch*10000+batch_number)
            batch = {key:value[indices] for key,value in data.items()}
            optimizer.zero_grad(set_to_none=True)
            terms,out = learning_terms(model,readouts,batch,condition,epoch,weights)
            if not all(torch.isfinite(value) for value in terms.values()):
                raise FloatingPointError('合成训练出现非有限损失')
            if batch_number==0:
                first_gradients = gradient_components(terms,named)
            terms['total'].backward()
            torch.nn.utils.clip_grad_norm_([p for _,p in named],1.)
            optimizer.step()
            for key,value in terms.items():
                sums[key]=sums.get(key,0.)+float(value.detach())*len(indices)
            for key,value in out['phase6_unweighted_losses'].items():
                raw_sums[key]=raw_sums.get(key,0.)+float(value.detach())*len(indices)
            count+=len(indices)
        if not count:
            raise ValueError('训练没有有效批次')
        row = dict(gamma=gamma,condition=condition,epoch=epoch,count=count,
            weighted_losses={k:v/count for k,v in sums.items()},first_batch_gradients=first_gradients,
            unweighted_losses={k:v/count for k,v in raw_sums.items()},
            weights=asdict(weights)|{'effective_beta':cfg.beta*min(epoch/cfg.beta_warmup_epochs,1.)},
            note='梯度为每轮第一批，非全数据平均；无测试/干预真值选模。')
        with (directory/'history.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        print(f'gamma={gamma} 条件={condition} epoch={epoch}/{args.epochs} CE={row["weighted_losses"]["factual_ce"]:.6f} 总损失={row["weighted_losses"]["total"]:.6f}',flush=True)
    model.eval();readouts.eval()
    duration=time.perf_counter()-started
    checkpoint=directory/'models'/f'gamma{gamma}_{condition}_last.pt'
    checkpoint.parent.mkdir(exist_ok=True)
    torch.save(dict(model=model.state_dict(),readouts=readouts.state_dict(),config=asdict(cfg),
        weights=asdict(weights),condition=condition,epoch=args.epochs,seed=args.seed),checkpoint)
    return model,dict(condition=condition,initial_model_sha256=initial,training_data_sha256=data_digest(data),
        fit_seconds=duration,training_keys=sorted(data),training_label_counts=torch.bincount(data['y'],minlength=2).tolist(),
        config=asdict(cfg),weights=asdict(weights))


def class_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    """合成正类1；矩阵行真值列预测；零分母记零，主真实数据协议不受影响。"""
    predicted=p.argmax(1)
    cm=np.bincount(2*y+predicted,minlength=4).reshape(2,2)
    tn,fp,fn,tp=cm.ravel()
    ratio=lambda a,b:float(a/b) if b else 0.
    f0,f1=ratio(2*tn,2*tn+fp+fn),ratio(2*tp,2*tp+fp+fn)
    q=np.clip(p[:,1],1e-7,1-1e-7)
    return dict(confusion_matrix=cm.tolist(),acc=ratio(tp+tn,cm.sum()),macro_f1=(f0+f1)/2,
        balanced_accuracy=(ratio(tp,tp+fn)+ratio(tn,tn+fp))/2,precision=ratio(tp,tp+fp),
        recall=ratio(tp,tp+fn),specificity=ratio(tn,tn+fp),f1_positive1=f1,
        observed_nll=float(np.mean(-y*np.log(q)-(1-y)*np.log1p(-q))),
        observed_brier=float(np.mean((q-y)**2)),mean_probability=p.mean(0).tolist(),
        std_probability=p.std(0).tolist())


def common_noise_probability(model, xc, xa, bank, S: int, K: int, seed: int):
    """只作MC归因：所有输入共用标准噪声与参考，积分仍调用唯一marginalize。"""
    with torch.random.fork_rng(devices=[]),torch.no_grad():
        torch.manual_seed(seed)
        b=len(xc);mu,lv=model.cns.distribution(xc)
        eps=torch.randn(1,S,model.config.mediator_dim)
        mc=mu[:,None]+(.5*lv[:,None]).exp()*eps
        if model.config.mediator_family=='tanh_normal':mc=mc.tanh()
        ans=model.ans(xa,mc)
        ma=ans['mu_a']+(.5*ans['logvar_a']).exp()*torch.randn(1,S,model.config.mediator_dim)
        if model.config.mediator_family=='tanh_normal':ma=ma.tanh()
        rc,ra,_=bank.sample(1,K)
        ref=model.reference_encoder(rc[0],ra[0])[None].expand(b,-1,-1)
        return model.marginalize(ma,ref)[0][:,1].numpy()


def evaluate(model, train_xy: dict, observed: dict, points: dict, truth: np.ndarray,
             oracle: np.ndarray, scm: SyntheticSCM, gamma: float, args) -> dict:
    """唯一可读真实中介/U机制的位置是独立评价；结果不回流fit或选模。"""
    model.eval()
    n=len(train_xy['y'])
    ids=torch.stack((torch.zeros(n,dtype=torch.long),torch.arange(n),torch.zeros(n,dtype=torch.long)),1)
    bank=ReferenceBank(train_xy['x_c'],train_xy['x_a'],ids,max_size=model.config.N_ref,seed=args.seed,forbidden_subjects=(1,2))
    test=training_view(observed,'oracle_mediator')
    pc=torch.tensor(points['x_c'],dtype=torch.float32);pa=torch.tensor(points['x_a'],dtype=torch.float32)
    observed_factual,observed_do=[],[]
    with torch.no_grad(),torch.random.fork_rng(devices=[]):
        torch.manual_seed(70000+args.seed)
        for start in range(0,len(test['y']),32):
            out=model(test['x_c'][start:start+32],test['x_a'][start:start+32],reference_bank=bank)
            observed_factual.append(out['probs_obs']);observed_do.append(out['probs_do'])
        own_ref=model.reference_encoder(test['x_c'],test['x_a'])
        conditional=model.outcome(test['m_a'],own_ref).softmax(-1)[:,1].numpy()
        true_conditional=scm.conditional_outcome(observed['m_a'],observed['x_c'],observed['x_a'],gamma)
        mu_c,lv_c=model.cns.distribution(test['x_c'])
        teacher_ans=model.ans(test['x_a'],test['m_c'][:,None])
        nll_c=model.cns.nll(test['m_c'],mu_c,lv_c).item()/model.config.mediator_dim
        nll_a=model.ans.nll(test['m_a'],teacher_ans['mu_a'][:,0],teacher_ans['logvar_a'][:,0]).item()/model.config.mediator_dim
        own=model(test['x_c'],test['x_a'],fd_mode='none')
        # 用同一标准CNS噪声网格跨输入比较，分开后验参数差异与噪声变化。
        eps=torch.randn(1,64,model.config.mediator_dim)
        common_mc=own['mu_c'][:,None]+(.5*own['logvar_c'][:,None]).exp()*eps
        if model.config.mediator_family=='tanh_normal':common_mc=common_mc.tanh()
        own_ans=model.ans(test['x_a'],common_mc)
        fixed_ans=model.ans(test['x_a'],common_mc[:1].expand(len(test['y']),-1,-1))
        from scripts.audit_outcome_support import posterior_statistics,marginal_ans_moments
        posterior=dict(cns=posterior_statistics(mu_c,lv_c),
            ans_fixed_mc=posterior_statistics(fixed_ans['mu_a'],fixed_ans['logvar_a']),
            ans_marginal=marginal_ans_moments(own_ans['mu_a'],own_ans['logvar_a']),
            parameter_space='Gaussian中介' if model.config.mediator_family=='normal' else 'tanh前的Gaussian参数与混合矩，不是tanh后Ma矩')
        posterior['cns_input_replacement_at_fixed_xa'] = dict(
            ans_mu_delta_rms=(own_ans['mu_a'].double()-fixed_ans['mu_a'].double()).square().mean().sqrt().item(),
            ans_sigma_delta_rms=((.5*own_ans['logvar_a'].double()).exp()-(.5*fixed_ans['logvar_a'].double()).exp()).square().mean().sqrt().item(),
            note='相同Xa与标准噪声，CNS后验参数替换为第一个输入；数值依赖诊断，不是物理干预。')
        posterior['cns_current_weight_rms'] = (model.ans.cns_current.weight.double().square().mean().sqrt().item()
            if model.ans.cns_current is not None else None)
        posterior['class_conditional_mean_distance'] = {}
        for name,value in [('cns',mu_c),('ans_pre_transform',own_ans['mu_a'].mean(1))]:
            zero,one=value[test['y']==0],value[test['y']==1]
            posterior['class_conditional_mean_distance'][name] = (zero.double().mean(0)-one.double().mean(0)).norm().item() if len(zero) and len(one) else None
        estimates,common,factual_points=[],[],[]
        for repeat in range(args.mc_repeats):
            torch.manual_seed(80000+args.seed+repeat)
            point_out=model(pc,pa,reference_bank=bank)
            estimates.append(point_out['probs_do'][:,1].numpy())
            point_reference=model.reference_encoder(pc,pa)
            factual_points.append(model.marginalize(point_out['m_a'],point_reference[:,None])[0][:,1].numpy())
            common.append(common_noise_probability(model,pc,pa,bank,model.config.S_eval,model.config.K_eval,80000+args.seed+repeat))
    values,crn=np.stack(estimates),np.stack(common)
    prevalence=float(train_xy['y'].float().mean())
    return dict(observed_factual=class_metrics(observed['y'],torch.cat(observed_factual).numpy()),
        do_against_observed_labels=class_metrics(observed['y'],torch.cat(observed_do).numpy()),
        interventional_per_repeat=[dict(repeat=i,against_direct=probability_metrics(v,truth),
            against_oracle=probability_metrics(v,oracle)) for i,v in enumerate(values)],
        interventional_mean_probability=probability_metrics(values.mean(0),truth),
        common_noise_interventional_mean_probability=probability_metrics(crn.mean(0),truth),
        factual_same_S_against_do_target=probability_metrics(np.stack(factual_points).mean(0),truth),
        factual_same_S_probability_per_repeat=np.stack(factual_points).tolist(),
        independent_mc_std_per_point=values.std(0).tolist(),common_noise_mc_std_per_point=crn.std(0).tolist(),
        independent_probability_per_repeat=values.tolist(),common_noise_probability_per_repeat=crn.tolist(),
        null_training_prevalence_error=probability_metrics(np.full(len(truth),prevalence),truth),
        half_probability_error=probability_metrics(np.full(len(truth),.5),truth),
        conditional_outcome_on_true_ma_mae=float(abs(conditional-true_conditional).mean()),
        coordinate_conditional_nll_per_dim=dict(cns=nll_c,ans_teacher_mc=nll_a),posterior=posterior,
        reference=dict(population=n,used=bank.used_count,shortfall=bank.shortfall),
        note='中介真值只供此独立评价。固定坐标密度与真值Ma上的Head误差都不具有可逆重参数化不变性；不能作为唯一恢复判据。观察标签上的do分类成绩不是干预概率真值误差。重复仅为MC流，不是多个训练种子。')


def main() -> None:
    p=argparse.ArgumentParser(description='Phase6同一SCM三种监督条件的独立合成验证',add_help=False)
    p.add_argument('-h','--help',action='help',help='显示中文帮助')
    p.add_argument('--output_dir',type=Path,required=True,help='新的phase6输出根目录')
    p.add_argument('--run_name',required=True,help='独立运行名，禁止覆盖')
    p.add_argument('--seed',type=int,choices=[42],default=42,help='固定训练种子，MC重复另编号')
    p.add_argument('--device',choices=['cpu'],default='cpu',help='独立CPU合成实验')
    p.add_argument('--epochs',type=int,default=20,help='三组均固定训练轮数，不用测试真值选模')
    p.add_argument('--batch_size',type=int,default=128,help='合成训练全局窗口打乱batch')
    p.add_argument('--workers',type=int,choices=[0],default=0,help='本实验顺序张量读取')
    p.add_argument('--diagnostic_level',choices=['full'],default='full',help='分项损失/梯度及概率误差')
    p.add_argument('--train_size',type=int,default=2048,help='三组同一自然分布训练样本数，不作标签下采样')
    p.add_argument('--test_size',type=int,default=512,help='独立观察测试样本数')
    p.add_argument('--points',type=int,default=16,help='共用干预评估点数')
    p.add_argument('--gammas',type=float,nargs='+',default=[0.,1.],help='固定机制下混杂强度')
    p.add_argument('--lr',type=float,default=.001,help='三组合成Adam学习率，沿用Phase1；不改真实训练学习率')
    p.add_argument('--weight_decay',type=float,default=0.,help='三组相同耦合L2，零沿用当前大batch对照')
    p.add_argument('--fidelity_weight',type=float,default=1.,help='候选组本模态统计重建权重')
    p.add_argument('--fire_weight',type=float,default=0.,help='合成发放率项权重，不等同HCI默认')
    p.add_argument('--no_cns2ans',action='store_true',help='仅估计器路径消融，生成SCM的CNS→ANS仍存在')
    p.add_argument('--mediator_family',choices=['normal','tanh_normal'],default='normal',help='默认匹配真实模型高斯；tanh_normal为分布匹配对照，先验同步推前')
    p.add_argument('--mc_repeats',type=int,default=3,help='同一已训模型的独立积分随机流次数')
    p.add_argument('--direct_n',type=int,default=32768,help='每点评价的直接干预MC样本数')
    p.add_argument('--oracle_s',type=int,default=1024,help='独立前门真值中介积分次数')
    p.add_argument('--oracle_k',type=int,default=512,help='独立前门真值参考积分次数')
    p.add_argument('--smoke',action='store_true',help='显式标记仅验证接口，不得解释为方法实验完成')
    args=p.parse_args()
    numeric=[args.lr,args.weight_decay,args.fidelity_weight,args.fire_weight,*args.gammas]
    if not all(math.isfinite(v) for v in numeric) or args.lr<=0 or min(numeric[1:])<0 or min(args.train_size,args.test_size,args.batch_size)<2 or min(args.epochs,args.points,args.mc_repeats)<1 or min(args.direct_n,args.oracle_s,args.oracle_k)<2 or len(set(args.gammas))!=len(args.gammas):
        p.error('规模须为正且概率积分至少2次，权重/混杂强度非负有限')
    if args.smoke and (args.epochs>2 or args.train_size>512 or args.test_size>128 or args.points>4):
        p.error('smoke限2轮/512训练/128测试/4个干预点')
    directory=args.output_dir/args.run_name
    if (directory/'config.json').exists():raise FileExistsError('已有运行，不能覆盖或隐式续训')
    directory.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.manual_seed(args.seed)
    scm=SyntheticSCM();scm.save(directory/'scm_parameters.npz')
    points=scm.sample(args.points,0.,78901)
    config=dict(source_version=source_version(Path(__file__).resolve().parent),
        arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        scm=asdict(scm.config),conditions=list(CONDITIONS),selection='自然训练总体，无标签选择，三组相同',
        supervision='只有oracle_mediator读取训练中介；其余fit严格Xc/Xa/Y白名单',
        model_family_note='生成器仍为tanh-Gaussian；normal为显式学习模型错设，不能称真实条件高斯。tanh_normal的KL对应tanh推前先验。',
        identification_note='输入保真是统计工作约束，不能识别潜在中介；无真实生理测量锚点。',
        evaluation='最后一轮评价，无test/oracle调参或选模；不修改真实LOSO选模规则。')
    save_json(directory/'config.json',config)
    results=dict(operation='Phase6三种监督条件合成验证',source_version=config['source_version'],smoke=args.smoke,
        completed=False,counterexample=counterexample(),gammas=[],training_executed=False,
        training_status='尚未进入学习阶段',real_data_training=False)
    save_json(directory/'probe_summary.json',results)
    for gamma in args.gammas:
        raw=scm.sample(args.train_size,gamma,args.seed+1000)
        observed=scm.sample(args.test_size,gamma,args.seed+2000)
        started=time.perf_counter()
        truth,se=scm.direct_intervention(points['x_c'],points['x_a'],gamma,args.direct_n,args.seed+3000)
        direct_time=time.perf_counter()-started
        started=time.perf_counter()
        oracle,ose=scm.frontdoor_integral(points['x_c'],points['x_a'],gamma,args.oracle_s,args.oracle_k,args.seed+4000)
        row=dict(gamma=gamma,direct_probability=truth.tolist(),direct_standard_error=se.tolist(),
            oracle_probability=oracle.tolist(),oracle_approx_standard_error=ose.tolist(),
            direct_seconds=direct_time,oracle_seconds=time.perf_counter()-started,
            truth_comparison=probability_metrics(oracle,truth),true_probability_std=float(truth.std()),conditions=[])
        if np.any(abs(oracle-truth)>4*np.sqrt(se**2+ose**2)+.01):
            row['truth_agreement_warning']=True
            results['gammas'].append(row)
            save_json(directory/'probe_summary.json',results)
            raise RuntimeError('直接与oracle差异超出预设诊断容差，已保存部分结果；停止学习实验，先检查数值真值。')
        else:row['truth_agreement_warning']=False
        for condition in CONDITIONS:
            data=training_view(raw,condition)
            results['training_status']=f'拟合已启动：gamma={gamma}，条件={condition}；若中断，实际步数查history/日志'
            if results['training_executed'] is not True:results['training_executed']=None
            results['gammas']=[x for x in results['gammas'] if x['gamma']!=gamma]+[row]
            save_json(directory/'probe_summary.json',results)
            model,record=fit(data,condition,args,directory,gamma)
            row['conditions'].append(record)
            results['training_executed']=True
            results['training_status']=f'本条件拟合完成，准备独立评价：{condition}'
            save_json(directory/'probe_summary.json',results)
            started=time.perf_counter()
            record['evaluation']=evaluate(model,training_view(raw,'observation_only'),observed,points,truth,oracle,scm,gamma,args)
            record['evaluation_seconds']=time.perf_counter()-started
            results['gammas']=[x for x in results['gammas'] if x['gamma']!=gamma]+[row]
            save_json(directory/'probe_summary.json',results)
            print(f'gamma={gamma} 条件={condition} 干预MAE={record["evaluation"]["interventional_mean_probability"]["mae"]:.6f}',flush=True)
            del model,data
        if len({x['training_data_sha256'] for x in row['conditions']})!=1 or len({x['initial_model_sha256'] for x in row['conditions']})!=1:
            raise RuntimeError('三组训练X/Y或初始模型没有保持配对')
    results['completed']=True
    results['training_status']='所有预定义拟合与独立评价已完成'
    save_json(directory/'probe_summary.json',results)
    (directory/'run_report.txt').write_text('三组配对合成验证完成。请检查真值差异、MC误差、干预误差与输入保真是否同步改善。\n'+
        'smoke仅接口测试；重建不能证明潜在中介可识别；不得将合成结果写成真实分类成绩。\n',encoding='utf-8')
    print(f'合成验证已结束：{directory/"probe_summary.json"}',flush=True)


if __name__=='__main__':
    main()
