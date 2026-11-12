"""只读审查训练被试原窗口锚点，不构造模型、不读取checkpoint、不训练。"""
import argparse
import json
import pickle
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

import numpy as np
import torch
from Datasets import DATASET_SPECS, normalize_subject
from physiological_anchors import extract_anchors, training_statistics
from phase6_synthetic_experiment import save_json
from run_reporting import source_version


def matrix_windows(value, windows):
    """[trial,channel,time]→[trial×window,channel,768]，与原数据顺序一致。"""
    trials, channels, length = value.shape
    if length != windows*768:
        raise ValueError('窗口长度不符')
    return torch.from_numpy(np.ascontiguousarray(value.reshape(trials, channels, windows, 768)
                                                .transpose(0, 2, 1, 3).reshape(-1, channels, 768)))


def feature_summary(value, mask, normalized, normalized_mask, names):
    """仅保存聚合统计；比较归一化前后，不能把相关性当生理真值验证。"""
    stats = training_statistics(value, mask)
    rows = []
    for j, name in enumerate(names):
        use = mask[:, j] & normalized_mask[:, j]
        a, b = value[use, j].double(), normalized[use, j].double()
        a, b = a-a.mean() if len(a) else a, b-b.mean() if len(b) else b
        norm = a.norm()*b.norm()
        correlation = float((a*b).sum()/norm) if len(a) > 1 and norm > 0 else None
        rows.append(dict(name=name, valid=int(mask[:, j].sum()), active=bool(stats['active'][j]),
                         mean=float(stats['mean'][j]), std=float(stats['scale'][j]),
                         raw_vs_normalized_correlation=correlation))
    return rows


def main():
    p = argparse.ArgumentParser(description='原窗口生理锚点训练来源与数值审查', add_help=False,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('-h', '--help', action='help', help='显示中文参数说明并退出')
    p.add_argument('--dataset', choices=['DEAP', 'HCI'], required=True, help='原数据集')
    p.add_argument('--label', choices=['arousal', 'valence'], default='arousal', help='记录任务；锚点不读取标签')
    p.add_argument('--root_dir', type=Path, default=Path('/home/fss'), help='含Dataset的数据根目录')
    p.add_argument('--output_dir', type=Path, required=True, help='独立结果根目录')
    p.add_argument('--run_name', required=True, help='不可覆盖的运行名')
    p.add_argument('--held_out_subjects', type=int, nargs=1, default=[1], help='只排除一个测试被试，不读取其文件')
    p.add_argument('--seed', type=int, choices=[42], default=42, help='记录固定种子，无随机样本选择')
    p.add_argument('--device', choices=['cpu'], default='cpu', help='只读CPU特征提取')
    p.add_argument('--batch_size', type=int, default=40, help='提取分块大小，不是训练batch')
    p.add_argument('--workers', type=int, choices=[0], default=0, help='单进程顺序读取')
    p.add_argument('--diagnostic_level', choices=['full'], default='full', help='完整聚合特征审查')
    p.add_argument('--anchor_variant', choices=['original','scale_stable'], default='original',
                   help='默认保留原锚点；scale_stable仅审查相对频谱/波形候选，不影响训练')
    p.add_argument('--check_affine_invariance', action='store_true',
                   help='仅scale_stable：逐窗口检查正增益/偏置不变性，失败保存结果并退出非零')
    args = p.parse_args()
    extractor = extract_anchors
    if args.check_affine_invariance and args.anchor_variant != 'scale_stable':
        p.error('不变性守卫仅用于scale_stable候选，不要求原始幅值锚点满足该性质')
    if args.anchor_variant == 'scale_stable':
        from anchor_invariance_candidates import extract_scale_stable, affine_perturbation, update_invariance
        extractor = extract_scale_stable
    spec = DATASET_SPECS[args.dataset]; heldout = args.held_out_subjects[0]
    if heldout not in range(1, spec['subjects']+1) or args.batch_size < 1:
        p.error('被试编号或提取batch无效')
    directory = args.output_dir/args.run_name
    if (directory/'config.json').exists() or (directory/'probe_summary.json').exists():
        raise FileExistsError('已有锚点审查，不覆盖；请更换运行名')
    directory.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    train_subjects = [s for s in range(1, spec['subjects']+1) if s != heldout]
    config = dict(source_version=source_version(PROJECT), arguments={k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                  train_subjects=train_subjects, held_out_subject=heldout, fs=128, points=768,
                  preprocessing='锚点从文件内基线截取后的归一化前窗口计算；另比较原normalize_subject输出，不改变数据管线')
    save_json(directory/'config.json', config)
    result = dict(operation='训练总体原窗口锚点只读审查', completed=False, training_executed=False,
                  real_data_training=False, source_version=config['source_version'], subjects=[],
                  held_out_subject_read=False, labels_field_accessed_or_used=False, config=config,
                  file_note='pickle会反序列化整份原字典；本脚本不访问labels字段或使用标签值')
    if args.check_affine_invariance:
        result['affine_invariance'] = {kind:{} for kind in ('gain','offset','channel_affine')}
        result['invariance_tolerance'] = dict(atol=1e-5, rtol=1e-5)
    save_json(directory/'probe_summary.json', result)
    collected = {k:[] for k in ('c','a','mask_c','mask_a','nc','na','nmask_c','nmask_a')}
    names = {}
    for subject in train_subjects:
        first_block = len(collected['c'])
        path = args.root_dir/'Dataset'/f'{args.dataset}-python'/f'{subject}.npy'
        with path.open('rb') as stream:
            record = pickle.load(stream)
        data = record['data']
        raw = data.detach().cpu().numpy() if torch.is_tensor(data) else np.asarray(data)
        if raw.ndim != 3 or raw.shape[0] != spec['trials'] or raw.shape[-1] != spec['points'] or raw.shape[1] <= max(spec['peripheral']):
            raise ValueError(f'被试{subject}的原始形状不符合当前数据接口')
        offset = 384 if args.dataset == 'DEAP' else 0
        xc = np.asarray(raw[:, :32, offset:], dtype=np.float32)
        xa = np.asarray(raw[:, spec['peripheral'], offset:], dtype=np.float32)
        inputs = [matrix_windows(v, spec['windows']) for v in (xc, xa, normalize_subject(xc), normalize_subject(xa))]
        for start in range(0, len(inputs[0]), args.batch_size):
            xc_batch, xa_batch = inputs[0][start:start+args.batch_size], inputs[1][start:start+args.batch_size]
            ac = extractor(xc_batch, xa_batch, args.dataset)
            norm = extractor(inputs[2][start:start+args.batch_size], inputs[3][start:start+args.batch_size], args.dataset)
            if args.check_affine_invariance:
                for kind, record_invariance in result['affine_invariance'].items():
                    perturbed = extractor(affine_perturbation(xc_batch, kind),
                                          affine_perturbation(xa_batch, kind), args.dataset)
                    update_invariance(record_invariance, ac, perturbed)
            for key in ('c','a','mask_c','mask_a'):
                collected[key].append(ac[key]); collected['n'+key].append(norm[key])
            names = {k:ac[k] for k in ('names_c','names_a')}
        moments = {}
        for side in ('c','a'):
            stats = training_statistics(torch.cat(collected[side][first_block:]),
                                        torch.cat(collected['mask_'+side][first_block:]))
            moments[side] = {key:stats[key].tolist() for key in ('mean','count')}
        result['subjects'].append(dict(subject_id=subject, trials=spec['trials'], windows=len(inputs[0]),
                                       anchor_moments=moments,
                                       source_order='trial_id从0，window_id从0；CNS/ANS同窗口配对'))
        print(f'{args.dataset} 已读取训练被试{subject}，窗口={len(inputs[0])}', flush=True)
        del record, raw, inputs, xc, xa
    values = {k:torch.cat(v) for k,v in collected.items()}
    result.update(completed=True, training_population_windows=len(values['c']),
                  dimensions=dict(cns=len(names['names_c']), ans=len(names['names_a'])),
                  features={s:feature_summary(values[s], values['mask_'+s], values['n'+s], values['nmask_'+s], names['names_'+s]) for s in ('c','a')},
                  interpretation='无原始单位校准/伪迹剔除保证；ECG为波形统计而非HRV，GSR不称SCR分解；与中介真值无等同关系',
                  anchor_variant=args.anchor_variant)
    result['all_modalities_have_active_features'] = all(any(r['active'] for r in result['features'][s]) for s in ('c','a'))
    for side in ('c','a'):
        for j, feature in enumerate(result['features'][side]):
            counts = np.array([r['anchor_moments'][side]['count'][j] for r in result['subjects']])
            means = np.array([r['anchor_moments'][side]['mean'][j] for r in result['subjects']])
            between = float((counts*(means-feature['mean'])**2).sum()/max(1,counts.sum()))
            feature['between_subject_variance_fraction'] = between/feature['std']**2 if feature['active'] else None
    result['subject_note'] = '被试间方差占比用于警惕身份信息；不把被试差异等同于已测U，也不自动去除或改预处理'
    if args.check_affine_invariance:
        rows = [v for r in result['affine_invariance'].values() for v in r.values()]
        result['affine_invariance_passed'] = bool(rows) and all(r['elements'] > 0 and r['violations'] == 0 and r['mask_changes'] == 0 for r in rows)
    save_json(directory/'probe_summary.json', result)
    if args.check_affine_invariance and not result['affine_invariance_passed']:
        raise RuntimeError('候选不变性超出容差或有效掩码改变，已保存审查结果；停止，不修改数据或阈值绕过')
    print(f'锚点审查完成：{directory}/probe_summary.json；没有模型或训练步骤', flush=True)


if __name__ == '__main__':
    main()
