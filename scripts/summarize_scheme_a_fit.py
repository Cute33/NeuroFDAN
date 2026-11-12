"""只读取已有方案A JSON，输出有界诊断摘要；不导入torch或训练代码。"""
import argparse
import json
import math
from pathlib import Path

from upload_run_diagnostics import read_snapshot, reject_constant, sanitize

ARGUMENTS = ('dataset', 'label', 'seed', 'held_out_subjects', 'classification_objective',
             'lambda_obs', 'lr', 'weight_decay', 'batch_size', 'eval_batch_size', 'epochs',
             'mediator_dim', 'hidden_dim', 'lif_neurons', 'dropout', 'beta',
             'beta_warmup_epochs', 'fire_weight', 'S_train', 'K_train', 'S_eval', 'K_eval',
             'N_ref', 'reference_chunk_size', 'sampling_mode', 'no_balance')
MODULES = ('backbone.cns_enc', 'backbone.ans_enc', 'cns', 'ans', 'cns2ans', 'outcome',
           'reference_encoder')


def select(value, keys):
    return {key: value[key] for key in keys if key in value}


def confusion_counts(cm):
    if (not isinstance(cm, list) or len(cm) != 2 or
            any(not isinstance(row, list) or len(row) != 2 for row in cm) or
            any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for row in cm for x in row)):
        raise ValueError('混淆矩阵必须为[0,1]行列顺序的2×2非负整数矩阵')
    true = [sum(row) for row in cm]
    pred = [cm[0][k] + cm[1][k] for k in (0, 1)]
    return dict(true_counts=true, predicted_counts=pred,
                single_class_prediction=sum(pred) > 0 and min(pred) == 0)


def gradient_summary(report):
    task = report.get('reference_term', 'integrated_observational_nll')
    gradients = report.get('gradients', {})
    modules = {}
    for module in MODULES:
        task_norm = gradients.get(task, {}).get(module, {}).get('l2')
        terms = {}
        for term in (task, 'kl_c', 'kl_a', 'fire', 'training_total'):
            row = gradients.get(term, {}).get(module)
            if row is None:
                continue
            item = select(row, ('l2', 'zero_tensors', 'missing_tensors', 'cosine_to_classification'))
            norm = row.get('l2')
            ratio = norm / task_norm if isinstance(norm, (float, int)) and isinstance(task_norm, (float, int)) and task_norm > 0 else None
            item['norm_over_task'] = ratio if ratio is not None and math.isfinite(ratio) else None
            terms[term] = item
        if terms:
            modules[module] = terms
    return dict(**select(report, ('epoch', 'weights', 'weighted_terms', 'reference_term')),
                modules=modules,
                note='同一前向的已加权分项梯度、裁剪前范数；比值不是Adam实际更新比，余弦非因果证据')


def build_summary(probe, config):
    prior = probe.get('constant_prior_nll')
    result = select(probe, ('dataset', 'label', 'held_out_subject', 'source_version',
        'completed', 'passed', 'training_executed', 'actual_steps', 'full_epoch_steps',
        'full_fold_training', 'weights_saved', 'weights_transferred_to_fold', 'fit_controls',
        'gate', 'true_counts', 'constant_prior_nll', 'first_full_backward_complete',
        'first_backward_peak_mib', 'peak_allocated_mib', 'error_type'))
    result['arguments'] = select(probe.get('arguments', {}), ARGUMENTS)
    result['workflow'] = select(config, ('workflow_state', 'stages', 'fit_weights_transferred_to_fold'))
    records = probe.get('records', [])
    # 常规300步只有7条；不打印来源ID数组或逐样本张量。
    if len(records) > 100:
        raise ValueError('检查点超过100条，请审查输入文件，不自动截断关键记录')
    result['checks'] = []
    for record in records:
        item = select(record, ('step', 'loss_epoch', 'optimization_nll', 'passed'))
        item['draws'] = []
        draws = record.get('draws', [])
        if len(draws) > 3:
            raise ValueError('方案A摘要预期每次最多3次MC评价')
        for draw in draws:
            local = select(draw, ('draw', 'acc', 'balanced_accuracy', 'macro_f1', 'min_class_recall',
                'class_recalls', 'integrated_observational_nll', 'confusion_matrix',
                'probability_mean', 'probability_std'))
            if 'confusion_matrix' in draw:
                local.update(confusion_counts(draw['confusion_matrix']))
            nll = draw.get('integrated_observational_nll')
            if isinstance(prior, (int, float)) and isinstance(nll, (int, float)):
                local['nll_improvement_over_constant'] = prior - nll
            item['draws'].append(local)
        result['checks'].append(item)
    if records:
        result['first_gradient_report'] = dict(step=records[0].get('step'),
            **gradient_summary(records[0].get('gradient_report') or {}))
        last = next((row for row in reversed(records) if row.get('gradient_report')), {})
        result['last_gradient_report'] = dict(step=last.get('step'),
            **gradient_summary(last.get('gradient_report') or {}))
        result['last_mediator_and_factual'] = [
            select(draw, ('draw', 'posterior', 'posterior_note', 'mediator_statistics',
                          'factual_confusion', 'factual_metrics')) for draw in records[-1].get('draws', [])]
    result['interpretation_limits'] = [
        '只报告现有固定训练子集的拟合，不是held-out或LOSO成绩。',
        'BA=0.5不单独证明单类别预测；以各次完整混淆矩阵和预测计数判断。',
        '均值变化、sigma变化和posterior sigma的绝对大小是不同统计量；不凭一项断言因果中介塌缩。',
        '分项梯度来自本步更新前train模式，MC矩阵与posterior来自更新后eval模式；不是同一模式的对齐比较。',
        '300步未过当前预算门槛，不证明所有优化配置或结构在理论上都不可能拟合。']
    return sanitize(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', help='已有方案A运行根目录，只读')
    args = parser.parse_args()
    run = Path(args.run_dir).expanduser().absolute()
    if not run.is_dir() or run.is_symlink():
        raise ValueError('run_dir必须为真实目录')
    read = lambda name: json.loads(read_snapshot(run / name, 16 * 1024**2).decode('utf-8'), parse_constant=reject_constant)
    probe = read('probe_summary.json')
    config = read('config.json') if (run / 'config.json').exists() else {}
    print('方案A已有结果摘要：只读取JSON；未加载模型、权重或原始信号，未执行训练。', flush=True)
    print(json.dumps(build_summary(probe, config), ensure_ascii=False, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
