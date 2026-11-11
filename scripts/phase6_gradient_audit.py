"""现有epoch20权重的分项梯度复核：只读CPU，不创建优化器，不修改真实训练。"""
import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT/'scripts')]
from probe_backbone_checkpoint import file_hash, activation_summary


def labeled_windows(root: Path, dataset: str, label: str, ids):
    """只读取原训练来源，复用原标准化；按完整三元组保留配对与标签。"""
    import torch
    from Datasets import PhysiologicalWindows
    requested = [tuple(x) for x in ids.tolist()]
    found = {}
    for subject in sorted({x[0] for x in requested}):
        data = PhysiologicalWindows(root, dataset, [subject], label)
        lookup = {tuple(x): i for i, x in enumerate(data.source_ids.tolist())}
        for source in {x for x in requested if x[0] == subject}:
            i = lookup[source]
            found[source] = (data.x_c[i].clone(), data.x_a[i].clone(), data.Label[i].clone())
        del data
    return tuple(torch.stack([found[x][i] for x in requested]) for i in range(3))


def measure(model, xc, xa, y, ids, bank, payload, training: bool) -> dict:
    """同一计算图逐项求导；do项只是P1的局部导数，不作为新训练目标。"""
    import torch
    from phase6_learning import gradient_components
    from frontdoor_scm import frontdoor_loss
    model.train(training)
    if any(m.training for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)):
        raise RuntimeError('BN未保持既有冻结规则')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        out = model(xc, xa, reference_bank=bank, source_ids=ids, fd_mode='full',
                    S=model.frontdoor_config.S_train, K=model.frontdoor_config.K_train)
        cfg, args = model.frontdoor_config, payload['arguments']
        beta = cfg.beta*min(payload['epoch']/cfg.beta_warmup_epochs, 1.)
        fw = args.get('fire_weight')
        fw = float(fw if fw is not None else (0. if args['dataset']=='DEAP' else 1.))
        losses = frontdoor_loss(out, y, cfg, payload['epoch'])
        terms = dict(factual_ce=args['lambda_obs']*losses['loss_obs'],
            kl_c=beta*out['kl_c'].mean(), kl_a=beta*out['kl_a'].mean(),
            fire=fw*out['fire_loss'],
            do_P1_local_derivative=out['probs_do'][:,1].mean())
        terms['training_total'] = sum(v for key,v in terms.items() if key!='do_P1_local_derivative')
        names = list(model.named_parameters())
        grouped = gradient_components(terms, names)
        def posterior_summary(mu, lv):
            # ANS先对S求均值，再计算输入差异；与条件网格变化分开解释。
            m, s = mu.double(), (.5*lv.double()).exp()
            if m.ndim == 3:
                m, s = m.mean(1), s.mean(1)
            classes = {str(k): m[y==k].mean(0).tolist() if (y==k).any() else None for k in (0,1)}
            return dict(mean_between_input_std=m.std(0,unbiased=False).mean().item(),
                sigma_between_input_std=s.std(0,unbiased=False).mean().item(),
                mean_posterior_sigma=s.mean().item(), class_mean=classes,
                class_mean_distance=float((m[y==0].mean(0)-m[y==1].mean(0)).norm()) if all((y==k).any() for k in (0,1)) else None)
        return dict(mode='train_frozen_bn' if training else 'eval',
            weighted_terms={k:float(v.detach()) for k,v in terms.items()},
            gradients=grouped, S=cfg.S_train, K=cfg.K_train,
            weights=dict(beta=beta, lambda_obs=args['lambda_obs'], fire=fw),
            posterior=dict(cns=posterior_summary(out['mu_c'],out['logvar_c']),
                           ans=posterior_summary(out['mu_a'],out['logvar_a'])),
            note='do_P1是敏感度泛函，不加入损失，不使用do标签；均值梯度可因样本抵消，不能替代逐样本依赖。train含原图随机性/dropout，eval为对应局部对照。')


def main() -> None:
    p = argparse.ArgumentParser(description='Phase6固定checkpoint分项梯度与类别后验诊断', add_help=False)
    p.add_argument('-h','--help',action='help',help='显示中文帮助')
    p.add_argument('--dataset',choices=['DEAP','HCI'],required=True,help='原数据集')
    p.add_argument('--label',choices=['arousal','valence'],required=True,help='原标签任务')
    p.add_argument('--root_dir',type=Path,default=Path('/home/fss'),help='原数据根目录')
    p.add_argument('--fold_dir',type=Path,required=True,help='大batch完整epoch20原fold目录')
    p.add_argument('--support_summary',type=Path,required=True,help='已完成支持审计的本地probe_summary.json')
    p.add_argument('--output_dir',type=Path,required=True,help='新的phase6输出根目录')
    p.add_argument('--run_name',required=True,help='新的诊断记录名称')
    p.add_argument('--seed',type=int,choices=[42],default=42,help='固定诊断随机流')
    p.add_argument('--device',choices=['cpu'],default='cpu',help='只读CPU诊断')
    p.add_argument('--held_out_subjects',type=int,nargs='+',choices=[1],default=[1],help='当前仅复核原第1折')
    p.add_argument('--epochs',type=int,choices=[0],default=0,help='零表示不训练')
    p.add_argument('--batch_size',type=int,choices=[16],default=16,help='复用原支持探针16个查询窗口，不改原训练batch')
    p.add_argument('--workers',type=int,choices=[0],default=0,help='顺序读取')
    p.add_argument('--diagnostic_level',choices=['full'],default='full',help='分项梯度、夹角和类别统计')
    args = p.parse_args()
    import torch
    from NeuroFDAN import NeuroFDAN
    from frontdoor_scm import FrontDoorConfig, ReferenceBank
    from run_reporting import source_version
    torch.set_num_threads(1)
    destination = args.output_dir/args.dataset/args.label/'seed42'/args.run_name
    if (destination/'probe_summary.json').exists():
        raise FileExistsError('已有诊断，不能覆盖')
    folder = args.fold_dir.resolve()
    if folder.parent == destination.resolve() or folder.parent in destination.resolve().parents:
        raise ValueError('输出不得放入原训练run')
    original = json.loads(args.support_summary.read_text(encoding='utf-8'))
    cfg = json.loads((folder/'fold.json').read_text(encoding='utf-8'))
    path = folder/'last.pt'
    fingerprint = file_hash(path)
    expected = (args.dataset,args.label,1,20)
    if (original['dataset'],original['task'],original['fold'],original['checkpoint_epoch']) != expected or original['checkpoint_sha256'] != fingerprint:
        raise ValueError('不是原支持审计的同一checkpoint')
    if original['source_version']['commit'] != 'c6fbb61f41bb956a64ae2a8d8a4152cbfa420e49':
        raise ValueError('支持审计源码版本不符')
    state = torch.load(path,map_location='cpu',weights_only=False)
    if state['epoch'] != 20 or state['arguments'] != cfg['arguments'] or state['frontdoor_config'] != cfg['model'] or cfg['smoke']:
        raise ValueError('原配置/完整性检查失败')
    if cfg['dataset'] != args.dataset or cfg['arguments']['label'] != args.label or cfg['seed'] != 42 or cfg['held_out_subject'] != 1:
        raise ValueError('任务/种子/fold不匹配')
    query = torch.tensor(original['sources']['query'],dtype=torch.long)
    all_ref = state['reference_source_ids'].cpu()
    if len(torch.unique(all_ref,dim=0))!=len(all_ref) or not set(all_ref[:,0].tolist())<=set(cfg['train_subjects']) or 1 in all_ref[:,0].tolist():
        raise ValueError('原参考库有重复或非训练来源')
    ref = all_ref[torch.randperm(len(all_ref),generator=torch.Generator().manual_seed(42))[:64]]
    combined = torch.cat((query,ref))
    if len(query)!=16 or any(s not in cfg['train_subjects'] or s==1 for s in combined[:,0].tolist()):
        raise ValueError('只能读取原训练来源')
    if {tuple(x) for x in query.tolist()} & {tuple(x) for x in all_ref.tolist()}:
        raise ValueError('查询不应在原参考库内')
    xc,xa,y = labeled_windows(args.root_dir,args.dataset,args.label,combined)
    bank = ReferenceBank(xc[16:],xa[16:],ref,forbidden_subjects=(1,),seed=42)
    bank.raw_windows = True
    torch.manual_seed(42)
    model = NeuroFDAN(ecg_channels=2 if args.dataset=='DEAP' else 3,frontdoor_config=FrontDoorConfig(**cfg['model'])).cpu()
    model.load_state_dict(state['model'],strict=True)
    before = {k:v.detach().clone() for k,v in model.state_dict().items()}
    records = [measure(model,xc[:16],xa[:16],y[:16],query,bank,state,mode) for mode in (True,False)]
    if any(p.grad is not None for p in model.parameters()) or not all(torch.equal(before[k],v) for k,v in model.state_dict().items()) or file_hash(path)!=fingerprint:
        raise RuntimeError('只读状态保护失败')
    result = dict(operation='Phase6现有权重分项梯度与模式复核',dataset=args.dataset,label=args.label,fold=1,
        source_version=source_version(PROJECT), checkpoint_sha256=fingerprint, checkpoint_epoch=20,
        original_training_source=cfg.get('source_version'),checkpoint_arguments=state['arguments'],
        frontdoor_config=state['frontdoor_config'],
        settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        query_sources=query.tolist(), reference_sources=ref.tolist(), label_counts=torch.bincount(y[:16],minlength=2).tolist(),
        selected_reference_label_counts=torch.bincount(y[16:],minlength=2).tolist(),
        input_statistics=dict(cns=activation_summary(xc[:16]),ans=activation_summary(xa[:16])),
        records=records,completed=True,model_state_unchanged=True,checkpoint_file_unchanged=True,optimizer_created=False,
        training_executed=False, reference_note='原2048成员库内无放回选64，仅局部梯度诊断，不是完整训练总体梯度。')
    destination.mkdir(parents=True,exist_ok=True)
    (destination/'probe_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    for row in records:
        print(row['mode'],json.dumps(row['weighted_terms'],ensure_ascii=False),flush=True)
    print(f'只读诊断完成：{destination/"probe_summary.json"}',flush=True)


if __name__=='__main__':
    main()
