"""任意已记录fold的只读分项梯度复核；不训练、不创建优化器、不更改原产物。"""
import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT / 'scripts')]
from phase6_gradient_audit import labeled_windows, measure
from probe_backbone_checkpoint import file_hash, activation_summary


def validate_identity(cfg, state, dataset, label, subject, epoch):
    """不把旧fold1/epoch20探针硬套到新的fold2/epoch300权重。"""
    if cfg.get('smoke') or state.get('format_version') != 2:
        raise ValueError('需要本项目完整训练checkpoint，不能使用smoke')
    if (cfg['dataset'], cfg['arguments']['label'], cfg['held_out_subject'], cfg['seed']) != (dataset, label, subject, 42):
        raise ValueError('数据集、任务、held-out或seed不匹配')
    if state['epoch'] != epoch or state['arguments'] != cfg['arguments'] or state['frontdoor_config'] != cfg['model']:
        raise ValueError('checkpoint实际epoch或原始配置不匹配')
    if cfg['arguments']['fd_mode'] != 'full' or subject in cfg['train_subjects']:
        raise ValueError('需要完整前门模式且训练来源排除held-out')
    report = state['reference_report']
    if report['held_out_subject'] != subject or report['dataset'] != dataset:
        raise ValueError('checkpoint参考库身份不匹配')


def validate_sources(query, references, original_bank, train_subjects, held_out):
    """纯来源检查：查询与参考均为原训练窗口，查询避开整个原参考库。"""
    q, r, bank = [tuple(x) for x in query], [tuple(x) for x in references], [tuple(x) for x in original_bank]
    if len(set(q)) != len(q) or len(set(r)) != len(r) or len(set(bank)) != len(bank):
        raise ValueError('来源含重复窗口')
    if any(x[0] not in train_subjects or x[0] == held_out for x in q + r + bank):
        raise ValueError('包含held-out或非训练被试')
    if not set(r) <= set(bank) or set(q) & set(bank):
        raise ValueError('参考必须来自原库，查询必须在原库之外')


def main():
    p = argparse.ArgumentParser(description='现有checkpoint分项梯度审查；epochs=0，不训练', add_help=False)
    p.add_argument('-h', '--help', action='help', help='显示中文说明')
    p.add_argument('--dataset', choices=['DEAP', 'HCI'], required=True)
    p.add_argument('--label', choices=['arousal', 'valence'], required=True)
    p.add_argument('--root_dir', type=Path, required=True)
    p.add_argument('--fold_dir', type=Path, required=True)
    p.add_argument('--checkpoint_epoch', type=int, required=True, help='必须匹配last.pt实际轮数')
    p.add_argument('--output_dir', type=Path, required=True)
    p.add_argument('--run_name', required=True)
    p.add_argument('--held_out_subjects', type=int, nargs=1, required=True)
    p.add_argument('--seed', type=int, choices=[42], default=42)
    p.add_argument('--device', choices=['cpu'], default='cpu')
    p.add_argument('--epochs', type=int, choices=[0], default=0)
    p.add_argument('--batch_size', type=int, default=16, help='每个局部训练被试的查询窗口数，非优化batch')
    p.add_argument('--reference_samples', type=int, default=64, help='从原参考库无放回取子集，仅局部诊断')
    p.add_argument('--workers', type=int, choices=[0], default=0)
    p.add_argument('--diagnostic_level', choices=['full'], default='full')
    args = p.parse_args()
    if args.batch_size < 2 or args.reference_samples < 1 or args.checkpoint_epoch < 1:
        p.error('查询至少2、参考至少1，checkpoint轮数至少1')
    if not args.run_name or Path(args.run_name).name != args.run_name or args.run_name in ('.', '..'):
        p.error('运行名只能为单个目录名')
    folder = args.fold_dir.resolve()
    destination = args.output_dir / args.dataset / args.label / 'seed42' / args.run_name
    if (destination / 'probe_summary.json').exists() or (destination / 'config.json').exists():
        raise FileExistsError('诊断目录已有产物，请使用新运行名')
    if destination.resolve() == folder.parent or folder.parent in destination.resolve().parents:
        raise ValueError('诊断输出不得放在原训练run内')
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'config.json').write_text(json.dumps(
        {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        ensure_ascii=False, indent=2), encoding='utf-8')
    import torch
    from Datasets import PhysiologicalWindows
    from NeuroFDAN import NeuroFDAN
    from frontdoor_scm import FrontDoorConfig, ReferenceBank
    from run_reporting import source_version
    torch.set_num_threads(1)
    path = folder / 'last.pt'
    fingerprint = file_hash(path)
    cfg = json.loads((folder / 'fold.json').read_text(encoding='utf-8'))
    state = torch.load(path, map_location='cpu', weights_only=False)
    held_out = args.held_out_subjects[0]
    validate_identity(cfg, state, args.dataset, args.label, held_out, args.checkpoint_epoch)
    all_ref = state['reference_source_ids'].cpu().long()
    recorded = json.loads((folder / 'reference_sources.json').read_text(encoding='utf-8'))
    expected_ref = [[x['subject_id'], x['trial_id'], x['window_id']] for x in recorded]
    if all_ref.tolist() != expected_ref or len(all_ref) < args.reference_samples:
        raise ValueError('原参考来源与checkpoint不一致，或小于请求的诊断参考数')
    train_subjects = sorted(cfg['train_subjects'])
    # 固定首/中/末训练被试；不依据测试ACC挑选，不做类别平衡或训练更新。
    subjects = sorted({train_subjects[0], train_subjects[len(train_subjects) // 2], train_subjects[-1]})
    bank_set = {tuple(x) for x in all_ref.tolist()}
    queries = []
    for subject in subjects:
        data = PhysiologicalWindows(args.root_dir, args.dataset, [subject], args.label)
        eligible = [x for x in data.source_ids.tolist() if tuple(x) not in bank_set]
        if len(eligible) < args.batch_size:
            raise ValueError('所选训练被试的库外查询窗口不足，不做静默替换')
        order = torch.randperm(len(eligible), generator=torch.Generator().manual_seed(args.seed + subject))
        queries.append(torch.tensor([eligible[i] for i in order[:args.batch_size].tolist()], dtype=torch.long))
        del data
    query = torch.cat(queries)
    refs = all_ref[torch.randperm(len(all_ref), generator=torch.Generator().manual_seed(args.seed))[:args.reference_samples]]
    validate_sources(query.tolist(), refs.tolist(), all_ref.tolist(), train_subjects, held_out)
    combined = torch.cat((query, refs))
    xc, xa, y = labeled_windows(args.root_dir, args.dataset, args.label, combined)
    n = len(query)
    bank = ReferenceBank(xc[n:], xa[n:], refs, forbidden_subjects=(held_out,), seed=42)
    bank.raw_windows = True
    torch.manual_seed(args.seed)
    model = NeuroFDAN(ecg_channels=2 if args.dataset == 'DEAP' else 3,
                      frontdoor_config=FrontDoorConfig(**state['frontdoor_config'])).cpu()
    model.load_state_dict(state['model'], strict=True)
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    records = []
    for batch, ids in enumerate(queries):
        sl = slice(batch * args.batch_size, (batch + 1) * args.batch_size)
        for training in (True, False):
            row = measure(model, xc[sl], xa[sl], y[sl], ids, bank, state, training)
            row.update(query_subject=subjects[batch], label_counts=torch.bincount(y[sl], minlength=2).tolist())
            records.append(row)
            print(f'被试={subjects[batch]} 模式={row["mode"]} 加权项={row["weighted_terms"]}', flush=True)
    if any(p.grad is not None for p in model.parameters()) or not all(torch.equal(before[k], v) for k, v in model.state_dict().items()):
        raise RuntimeError('只读保护失败：参数、BN或.grad发生改变')
    if file_hash(path) != fingerprint:
        raise RuntimeError('原checkpoint在诊断期间改变')
    result = dict(operation='现有checkpoint只读分项梯度', dataset=args.dataset, label=args.label,
        fold=held_out, checkpoint_epoch=state['epoch'], checkpoint_sha256=fingerprint,
        source_version=source_version(PROJECT), original_training_source=cfg.get('source_version'),
        frontdoor_config=state['frontdoor_config'], checkpoint_arguments=state['arguments'],
        query_sources=query.tolist(), reference_sources=refs.tolist(),
        reference_label_counts=torch.bincount(y[n:], minlength=2).tolist(), records=records,
        input_statistics=dict(cns=activation_summary(xc[:n]), ans=activation_summary(xa[:n])),
        completed=True, model_state_unchanged=True, checkpoint_file_unchanged=True,
        optimizer_created=False, training_executed=False,
        note='3个固定训练被试的局部窗口与64个原参考样本，不能代表全训练总体梯度；不估计因果效应。'
             'do_P1仅为局部敏感度泛函，未加入训练损失；没有新辅助头。')
    (destination / 'probe_summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print('只读审查完成；未执行optimizer step，原参数、BN、checkpoint均未改变。', flush=True)


if __name__ == '__main__':
    main()
