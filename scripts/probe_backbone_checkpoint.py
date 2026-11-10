"""E1b只读骨干逐层探针：不训练、不回传梯度、不保存信号/权重/完整特征。"""
import argparse
import hashlib
import json
import pickle
import random
import re
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def tensor_summary(value) -> dict:
    """先转float64再统计，避免对极小float32权重平方时下溢；不改原张量。"""
    import torch
    x = value.detach().cpu().double()
    if not x.numel() or not torch.isfinite(x).all():
        raise FloatingPointError('遇到空张量或NaN/Inf，停止生成正常统计')
    nonzero = x.abs()[x != 0]
    return dict(shape=list(x.shape), dtype=str(value.dtype),
                mean=x.mean().item(), std=x.std(unbiased=False).item(),
                min=x.min().item(), max=x.max().item(), rms=x.square().mean().sqrt().item(),
                l2=x.square().sum().sqrt().item(), abs_max=x.abs().max().item(),
                min_nonzero_abs=nonzero.min().item() if nonzero.numel() else None,
                zero_fraction=(x == 0).double().mean().item())


def activation_summary(value) -> dict:
    """第0轴是所选窗口；明确区分全元素变化与跨样本变化。"""
    result = tensor_summary(value)
    x = value.detach().cpu().double().reshape(value.shape[0], -1)
    spread = x.std(0, unbiased=False)
    result.update(between_sample_std_mean=spread.mean().item(),
                  between_sample_std_max=spread.max().item(),
                  variable_feature_fraction=(spread > 0).double().mean().item(),
                  exact_constant_across_samples=bool((spread == 0).all()))
    return result


def inspect_backbone(model, xc, xa) -> dict:
    """hook沿真实backbone.forward读取18个现有模块；不手写替代前向。"""
    import torch
    model.cpu().eval()
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    for name, value in before.items():
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise FloatingPointError(f'checkpoint含非有限参数或buffer：{name}')
    trace, modules, handles = {}, {}, []
    for name, module in model.backbone.named_modules():
        if not re.fullmatch(r'(cns_enc|ans_enc)\.\d+', name):
            continue
        details = dict(type=type(module).__name__, training=module.training,
                       parameters={key: tensor_summary(p) for key, p in module.named_parameters(recurse=False)})
        if isinstance(module, torch.nn.Conv1d):
            details.update(groups=module.groups, kernel_size=list(module.kernel_size), stride=list(module.stride))
        if isinstance(module, torch.nn.BatchNorm1d):
            details.update(eps=module.eps,
                buffers={key: tensor_summary(b) for key, b in module.named_buffers(recurse=False)})
            if module.affine and module.running_var is not None:
                if ((module.running_var.double() + module.eps) <= 0).any():
                    raise FloatingPointError(f'{name}的BN方差/eps无效')
                gain = module.weight.detach().double() / (module.running_var.detach().double() + module.eps).sqrt()
                details['effective_bn_gain'] = tensor_summary(gain)
        modules[name] = details
        def record(module, inputs, output, layer=name):
            trace[layer] = activation_summary(output)
        handles.append(module.register_forward_hook(record))
    try:
        with torch.no_grad():
            hc, ha = model.backbone(xc, xa)
            outputs = dict(cns=activation_summary(hc), ans=activation_summary(ha))
            parent = (model.backbone.dag_adj + model.backbone.eval_gumbel).argmax(-1)
            nonself = parent != torch.arange(32)
            graph = dict(nonself_rows=int(nonself.sum()), empty_rows_after_self_removal=int((~nonself).sum()),
                         distinct_selected_nonself_columns=int(torch.unique(parent[nonself]).numel()),
                         logits=tensor_summary(model.backbone.dag_adj))
    finally:
        for handle in handles:
            handle.remove()
    if len(trace) != 18 or set(trace) != set(modules):
        raise RuntimeError('没有覆盖既定两支骨干的18个模块，不能生成完整逐层结论')
    if not all(torch.equal(before[name], value) for name, value in model.state_dict().items()):
        raise RuntimeError('探针前后模型参数或buffer不同，停止')
    if any(p.grad is not None for p in model.parameters()):
        raise RuntimeError('只读探针中出现了参数梯度，停止')
    first_constant = {}
    for branch in ('cns_enc', 'ans_enc'):
        first_constant[branch] = next((name for name, row in trace.items()
                                      if name.startswith(branch + '.') and row['exact_constant_across_samples']), None)
    return dict(layers=trace, module_statistics=modules, outputs=outputs, graph=graph,
                first_exact_constant_layer=first_constant, model_state_unchanged=True,
                gradients_created=False)


def main() -> None:
    parser = argparse.ArgumentParser(description='E1b逐层只读骨干诊断，不运行训练入口', add_help=False)
    parser.add_argument('-h', '--help', action='help', help='显示中文说明')
    parser.add_argument('--root_dir', type=Path, default=Path('/home/fss'), help='原数据根目录，不转换数据')
    parser.add_argument('--dataset', choices=['DEAP', 'HCI'], required=True, help='必须与checkpoint一致')
    parser.add_argument('--task', choices=['arousal', 'valence'], required=True, help='必须与checkpoint任务一致')
    parser.add_argument('--held_out_subject', type=int, required=True, help='只读取该被试文件')
    parser.add_argument('--fold_dir', type=Path, required=True, help='原训练fold目录，读取last.pt/fold.json')
    parser.add_argument('--e1_summary', type=Path, required=True, help='已完成E1的probe_summary.json，用于校验权重与窗口')
    parser.add_argument('--output_dir', type=Path, required=True, help='独立探针结果目录，禁止写到原fold内部')
    parser.add_argument('--expected_epoch', type=int, required=True, help='last.pt必须处于此轮，不自动改用其他checkpoint')
    parser.add_argument('--samples', type=int, default=16, help='与E1相同的窗口数，单批前向')
    parser.add_argument('--seed', type=int, choices=[42], default=42, help='固定种子，仅用于参考初始化')
    parser.add_argument('--device', choices=['cpu'], default='cpu', help='只支持CPU，不执行GPU实验')
    parser.add_argument('--include_best', action='store_true', help='存在兼容best.pt时附加只读对比；缺失则记录')
    parser.add_argument('--include_initial', action='store_true', help='附加未训练seed42参考初始化，不替代已训练权重')
    args = parser.parse_args()
    if not 2 <= args.samples <= 64:
        parser.error('只读探针样本数限制为2–64')

    import numpy as np
    import torch
    from NeuroFDAN import NeuroFDAN
    from Datasets import PhysiologicalWindows
    from frontdoor_scm import FrontDoorConfig
    from run_reporting import source_version
    torch.set_num_threads(1)
    folder, destination = args.fold_dir.resolve(), args.output_dir.resolve()
    if destination == folder or folder in destination.parents:
        raise ValueError('不能把探针输出写入原训练fold')
    if destination == args.e1_summary.resolve().parent:
        raise ValueError('不能覆盖原E1结果目录')
    if (destination / 'probe_summary.json').exists():
        raise FileExistsError('已有探针结果，请使用新目录，不覆盖')
    destination.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((folder / 'fold.json').read_text(encoding='utf-8'))
    e1 = json.loads(args.e1_summary.read_text(encoding='utf-8'))
    expected = (args.dataset, args.task, args.held_out_subject, args.seed)
    if (e1.get('dataset'), e1.get('task'), e1.get('fold'), e1.get('seed')) != expected:
        raise ValueError('E1元数据与本次任务不匹配')
    if (cfg.get('dataset'), cfg.get('arguments', {}).get('label'), cfg.get('held_out_subject'), cfg.get('seed')) != expected:
        raise ValueError('fold.json与本次任务不匹配')
    hashes = {}
    def load(name, expected_epoch=None):
        path = folder / name
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f'缺少普通checkpoint文件：{name}')
        hashes[name] = file_hash(path)
        payload = torch.load(path, map_location='cpu', weights_only=False)
        if not isinstance(payload, dict) or payload.get('format_version') != 2:
            raise ValueError(f'{name}不是格式2的结构checkpoint')
        if not {'arguments', 'reference_report', 'frontdoor_config', 'model', 'epoch'}.issubset(payload):
            raise ValueError(f'{name}字段不完整')
        if not isinstance(payload['arguments'], dict) or not isinstance(payload['reference_report'], dict):
            raise ValueError(f'{name}元数据类型无效')
        settings = payload['arguments']
        identity = (settings.get('dataset'), settings.get('label'),
                    payload['reference_report'].get('held_out_subject'), settings.get('seed'))
        if identity != expected or payload['frontdoor_config'] != cfg['model']:
            raise ValueError(f'{name}的任务/来源/模型配置不匹配')
        if expected_epoch is not None and payload['epoch'] != expected_epoch:
            raise ValueError(f'{name}实际epoch={payload["epoch"]}，与要求不符')
        return payload

    # 先确认真实last权重；失败时绝不以随机模型代替其诊断。
    last = load('last.pt', args.expected_epoch)
    if hashes['last.pt'] != e1.get('checkpoint_sha256') or last['epoch'] != e1.get('checkpoint_epoch'):
        raise ValueError('last.pt不是E1使用的同一份权重，请核对路径/是否续训')
    config = FrontDoorConfig(**last['frontdoor_config'])
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    data = PhysiologicalWindows(args.root_dir, args.dataset, [args.held_out_subject], args.task)
    indices = torch.linspace(0, len(data) - 1, args.samples).long()
    sources = data.source_ids[indices].tolist()
    if e1.get('samples') != args.samples or sources != e1.get('sample_sources'):
        raise ValueError('窗口来源与E1不同，停止配对比较')
    xc, xa = data.x_c[indices], data.x_a[indices]
    result = dict(operation='E1b只读骨干逐层诊断', dataset=args.dataset, task=args.task,
                  fold=args.held_out_subject, seed=args.seed, device='cpu', samples=args.samples,
                  source_version=source_version(PROJECT), sample_sources=sources,
                  input_statistics=dict(cns=activation_summary(xc), ans=activation_summary(xa)),
                  models={}, optimizer_created=False,
                  note='只保存统计，不保存信号/特征/权重；参考初始化不保证与历史启动逐位一致；未训练、未执行中介或前门积分')
    result['e1_input_difference'] = dict(
        cns=result['input_statistics']['cns']['between_sample_std_mean'] - e1['input_std_c'],
        ans=result['input_statistics']['ans']['between_sample_std_mean'] - e1['input_std_a'])
    def inspect(label, payload):
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        model = NeuroFDAN(ecg_channels=2 if args.dataset == 'DEAP' else 3, frontdoor_config=config)
        if payload is not None:
            model.load_state_dict(payload['model'], strict=True)
        print(f'开始只读骨干诊断：{label}', flush=True)
        diagnosis = inspect_backbone(model, xc, xa)
        diagnosis.update(kind='未训练参考初始化' if payload is None else '实际训练checkpoint',
                         epoch=None if payload is None else payload['epoch'])
        if payload is not None:
            diagnosis['optimizer_configuration'] = [
                {key: group.get(key) for key in ('lr', 'weight_decay', 'betas', 'eps', 'amsgrad')}
                for group in payload.get('optimizer', {}).get('param_groups', [])]
        result['models'][label] = diagnosis
        for name, row in diagnosis['layers'].items():
            print(f'  {label}/{name} 跨样本std={row["between_sample_std_mean"]:.6e} '
                  f'RMS={row["rms"]:.6e} 零值比例={row["zero_fraction"]:.4f}', flush=True)
        print(f'  首个精确常量层：{diagnosis["first_exact_constant_layer"]}', flush=True)

    if args.include_initial:
        inspect('initial_seed42_reference', None)
    if args.include_best:
        try:
            best = load('best.pt')
        except (OSError, ValueError, KeyError, RuntimeError, EOFError, pickle.UnpicklingError) as error:
            result['models']['best_checkpoint'] = dict(status='未加载', reason=str(error))
            print(f'可选best未加载：{error}', flush=True)
        else:
            inspect('best_checkpoint', best)
            del best
    inspect('last_checkpoint', last)
    output = result['models']['last_checkpoint']['outputs']
    result['e1_output_difference'] = dict(
        cns=output['cns']['between_sample_std_mean'] - e1['feature_std_c'],
        ans=output['ans']['between_sample_std_mean'] - e1['feature_std_a'])
    for name, fingerprint in hashes.items():
        if file_hash(folder / name) != fingerprint:
            raise RuntimeError(f'{name}在读取期间变化，停止输出成功结果')
    result.update(checkpoint_sha256=hashes, checkpoint_files_unchanged=True)
    content = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    with (destination / 'probe_summary.json').open('x', encoding='utf-8') as stream:
        stream.write(content + '\n')
    print(f'E1b完成；结果：{destination / "probe_summary.json"}', flush=True)


if __name__ == '__main__':
    main()
