"""只读LOSO产物审计与导出；不读取真实数据、不加载或运行模型权重。"""
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

# 与当前数据加载器的固定实验规格一致，测试负责检查两处未发生偏离。
SPECS = {'DEAP': (32, 40, 10), 'HCI': (24, 20, 5)}
STATUS_CN = {'not_started': '未运行', 'incomplete': '未完成：运行中/失败/中断需日志确认',
             'completed_no_best': '完成但未达保存门槛', 'completed': '完整完成且有合格best',
             'smoke': '接口检查，非正式', 'inconsistent': '产物/来源/协议不一致'}
METRICS = ('accuracy', 'sensitivity', 'specificity', 'f1')


def metric_values(matrix, positive_class=0):
    """行真值、列预测；返回比例，零分母沿用原规则记零。"""
    cm = np.asarray(matrix)
    if positive_class not in (0, 1) or cm.shape != (2, 2) or not np.isfinite(cm).all() or (cm < 0).any() or not cm.sum():
        raise ValueError('非空二分类矩阵或正类设置错误')
    p = positive_class
    tp, tn, fp, fn = cm[p, p], cm[1-p, 1-p], cm[1-p, p], cm[p, 1-p]
    return [float((tp + tn) / cm.sum()), float(tp / (tp + fn)) if tp + fn else 0.,
            float(tn / (tn + fp)) if tn + fp else 0.,
            float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.]


def _json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def _finite_numbers(value):
    if isinstance(value, dict):
        return all(_finite_numbers(v) for v in value.values())
    if isinstance(value, list):
        return all(_finite_numbers(v) for v in value)
    return not isinstance(value, (int, float)) or bool(np.isfinite(value))


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def audit_fold(directory, dataset, label, subject, aggregate, positive_class):
    """审计一个预期fold；缺少结束证据时不猜测进程是否仍在运行。"""
    n, trials, windows = SPECS[dataset]
    expected = trials * windows
    folder = directory / f'fold{subject:02d}'
    row = {'subject_id': subject, 'status': 'not_started', 'status_cn': STATUS_CN['not_started'],
           'samples': None, **dict.fromkeys(METRICS), 'reason': '', 'planned_epochs': None,
           'last_recorded_epoch': None, 'reference_count': None}
    signature = None
    def finish(status, reason=''):
        row.update(status=status, status_cn=STATUS_CN[status], reason=reason)
        return row, signature
    if not folder.exists():
        if aggregate is not None and aggregate[subject - 1].sum():
            return finish('inconsistent', '汇总已有数值但缺少该fold来源/结束产物')
        return finish('not_started', '预期fold目录不存在')
    if not (folder / 'fold.json').is_file():
        return finish('incomplete', '已有目录但未写入fold配置，需结合进程/退出码判断')
    try:
        cfg = _json(folder / 'fold.json')
        args = cfg['arguments']
        smoke = bool(cfg.get('smoke') or args.get('max_train_batches', 0) or args.get('max_test_batches', 0))
        _require(cfg['dataset'] == dataset and args['dataset'] == dataset and args['label'] == label, '数据集/任务不匹配')
        _require(cfg['held_out_subject'] == subject and cfg['seed'] == 42 and cfg['no_validation'] is True, 'fold、种子或二分协议不一致')
        _require(set(cfg['train_subjects']) == set(range(1, n + 1)) - {subject}, '训练被试列表不符合完整LOSO')
        _require(cfg['test_windows'] == expected, '完整测试窗口数不符合当前数据接口')
        _require(cfg['population_windows'] == (n - 1) * expected and cfg['population_subjects'] == n - 1, '未平衡参考总体数量不符合完整训练被试')
        sources = _json(folder / 'reference_sources.json')
        count = cfg['actual_N_ref']
        row['reference_count'] = count
        _require(count == len(sources) == min(cfg['requested_N_ref'], cfg['population_windows']) and count > 0, '参考数量或短缺处理不一致')
        _require(cfg['shortfall'] == (cfg['requested_N_ref'] > cfg['population_windows']), '参考短缺标识不一致')
        source_ids = []
        for source in sources:
            s, t, w = source['subject_id'], source['trial_id'], source['window_id']
            _require(all(isinstance(v, int) and not isinstance(v, bool) for v in (s, t, w)), '来源编号必须是整数')
            _require(s in cfg['train_subjects'] and s != subject and 0 <= t < trials and 0 <= w < windows and source['dataset'] == dataset, '参考来源含测试被试或编号/数据集错误')
            source_ids.append((s, t, w))
        _require(len(set(source_ids)) == count and cfg['source_check_passed'] is True, '参考来源重复或原检查未通过')
        _require(cfg['actual_subject_count'] == len({v[0] for v in source_ids}), '参考被试数量与来源记录不一致')
        if smoke:
            return finish('smoke', '来源检查通过；有限batch运行，禁止作为完整LOSO结果')
        keys = ('dataset', 'label', 'seed', 'epochs', 'batch_size', 'lr', 'fd_mode',
                'no_balance', 'lambda_obs', 'fire_weight')
        signature = json.dumps({'arguments': {key: args[key] for key in keys}, 'model': cfg['model']}, sort_keys=True)
        planned = args['epochs']
        row['planned_epochs'] = planned
        _require(isinstance(planned, int) and planned > 0, '计划轮数无效')
        if not (folder / 'summary.json').is_file():
            return finish('incomplete', '尚无正常收尾summary；文件缺失不能单独证明失败')
        summary = _json(folder / 'summary.json')
        if summary.get('smoke'):
            return finish('smoke', 'summary明确标记为接口检查')
        _require(summary['held_out'] == subject and (folder / 'last.pt').is_file(), '结束记录或last checkpoint缺失')
        history = [_json_line(line) for line in (folder / 'history.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
        if not history:
            return finish('incomplete', '无epoch历史')
        row['last_recorded_epoch'] = history[-1]['epoch']
        if history[-1]['epoch'] < planned:
            return finish('incomplete', '尚未达到当前计划轮数，可能在运行或续训中')
        _require(history[-1]['epoch'] == planned and set(range(1, planned + 1)).issubset({h['epoch'] for h in history}), 'epoch记录不完整或超出计划')
        _require(all(_finite_numbers(h['train']) and _finite_numbers(h['test']) for h in history), '历史指标存在NaN/Inf')
        _require(all(h['test']['complete_test'] is True and h['test']['count'] == expected for h in history), '历史混入不完整测试')
        _require(summary['last_test'] == history[-1]['test'] and summary['last_test']['complete_test'] is True and summary['last_test']['count'] == expected, 'summary与最后历史/完整测试不一致，可能收尾中断')
        # 续训可重复写同一epoch；只审计每个epoch的最后记录，不重新选择或覆盖模型。
        latest = {h['epoch']: h['test'] for h in history}
        eligible = [(v['acc'], v['selection_f1'], epoch) for epoch, v in latest.items() if v['acc'] > .5]
        best = summary['best']
        if not summary['has_eligible_best']:
            _require(not eligible, '历史已有ACC>0.5记录却声明从未产生合格best')
            _require(best['epoch'] == 0 and not (folder / 'best.pt').exists() and not (folder / 'best_confusion.npy').exists(), '无best标记与已有最佳产物矛盾')
            _require(aggregate is None or not aggregate[subject - 1].sum(), '无best却有非零汇总行')
            return finish('completed_no_best', '正常结束，但原选模规则未产生合格best')
        _require(0 < best['epoch'] <= planned and best['acc'] > .5, '最佳epoch/ACC不符合原保存门槛')
        _require(bool(eligible), '最佳指标没有对应的合格epoch历史')
        historical_best = max(eligible, key=lambda item: (item[0], item[1], -item[2]))
        _require(best['epoch'] == historical_best[2] and np.isclose(best['acc'], historical_best[0]) and np.isclose(best['f1'], historical_best[1]), '最佳记录与原ACC/F1历史选模规则不一致')
        _require((folder / 'best.pt').is_file(), '合格best checkpoint缺失')
        cm = np.load(folder / 'best_confusion.npy', allow_pickle=False)
        _require(cm.shape == (2, 2) and np.isfinite(cm).all() and (cm >= 0).all() and np.equal(cm, np.floor(cm)).all() and cm.sum() == expected, '最佳矩阵样本数或计数错误')
        _require(aggregate is not None and np.array_equal(cm, aggregate[subject - 1]), '跨被试矩阵缺失或与该fold最佳矩阵不同')
        prediction = _json(folder / 'best_predictions.json')
        _require(len(prediction) == expected, '最佳预测未覆盖完整测试窗口')
        predicted_cm, ids = np.zeros((2, 2), dtype=np.int64), []
        for entry in prediction:
            t, w, y = entry['trial_id'], entry['window_id'], entry['label']
            _require(entry['subject_id'] == subject and entry['dataset'] == dataset and 0 <= t < trials and 0 <= w < windows and y in (0, 1), '预测来源/标签错误')
            probs = np.asarray(entry['probabilities'])
            _require(probs.shape == (2,) and np.isfinite(probs).all() and (probs >= 0).all() and (probs <= 1).all() and np.isclose(probs.sum(), 1., atol=1e-5), '预测概率未归一化或含NaN/Inf')
            predicted_cm[int(y), int(probs.argmax())] += 1
            ids.append((t, w))
        _require(len(set(ids)) == expected and np.array_equal(predicted_cm, cm), '预测窗口重复或概率重建矩阵不一致')
        policy_class = 1 if dataset == 'DEAP' else 0
        policy = metric_values(cm, policy_class)
        _require(np.isclose(best['acc'], policy[0]) and np.isclose(best['f1'], policy[3]), 'summary最佳指标与原选模口径不一致')
        row.update(samples=expected, **dict(zip(METRICS, metric_values(cm, positive_class))))
        return finish('completed')
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        return finish('inconsistent', str(error))


def _json_line(line):
    return json.loads(line)


def audit_run(run_dir, dataset, label, positive_class=0):
    """对全部预期被试审计，绝不只枚举有结果的目录。"""
    directory = Path(run_dir)
    n = SPECS[dataset][0]
    matrix_path = directory / f'NeuroFDAN-{dataset}_{label}_cm.npy'
    matrices, matrix_error = None, ''
    if matrix_path.exists():
        try:
            matrices = np.load(matrix_path, allow_pickle=False)
            _require(matrices.shape == (n, 2, 2) and np.isfinite(matrices).all() and (matrices >= 0).all() and np.equal(matrices, np.floor(matrices)).all(), '汇总矩阵形状或计数无效')
        except (OSError, ValueError, TypeError) as error:
            matrices, matrix_error = None, str(error)
    rows, baseline = [], None
    for subject in range(1, n + 1):
        row, signature = audit_fold(directory, dataset, label, subject, matrices, positive_class)
        if row['status'] in ('completed', 'completed_no_best'):
            if baseline is None:
                baseline = signature
            elif signature != baseline:
                row.update(status='inconsistent', status_cn=STATUS_CN['inconsistent'], reason='各fold的模型/训练/采样配置不一致')
        rows.append(row)
    eligible = [r for r in rows if r['status'] == 'completed']
    complete = len(eligible) == n and not matrix_error
    values = np.array([[r[k] for k in METRICS] for r in eligible]) if eligible else None
    return {'dataset': dataset, 'label': label, 'positive_class': positive_class,
            'expected_subjects': n, 'eligible_subjects': len(eligible), 'complete': complete,
            'execution_complete': all(r['status'] in ('completed', 'completed_no_best') for r in rows),
            'scope': '完整LOSO' if complete else '合格被试子集，非完整LOSO', 'std_ddof': 0,
            'status_counts': dict(Counter(r['status'] for r in rows)), 'matrix_error': matrix_error,
            'mean': values.mean(0).tolist() if eligible else None,
            'std': values.std(0).tolist() if eligible else None, 'folds': rows,
            'checkpoint_contents_verified': False,
            'selection_bias_note': 'held-out测试数据参与checkpoint选择，结果不是独立无偏泛化估计'}


def export_report(report, export_dir):
    """即使拒绝完整性验收，也导出带complete=false的诊断而非旧统计残留。"""
    directory = Path(export_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    with (directory / 'per_subject.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(report['folds'][0]))
        writer.writeheader(); writer.writerows(report['folds'])
    with (directory / 'summary.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        fields = ['dataset', 'label', 'positive_class', 'metric', 'mean', 'std', 'mean_percent', 'std_percent',
                  'eligible_subjects', 'expected_subjects', 'complete', 'scope', 'std_ddof']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i, metric in enumerate(METRICS):
            mean = report['mean'][i] if report['mean'] is not None else None
            std = report['std'][i] if report['std'] is not None else None
            writer.writerow({**{k: report[k] for k in fields if k in report}, 'metric': metric,
                             'mean': mean, 'std': std, 'mean_percent': mean * 100 if mean is not None else None,
                             'std_percent': std * 100 if std is not None else None})
