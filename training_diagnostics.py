"""只读训练诊断：流式统计、裁剪前梯度及逐轮产物；不定义模型或损失。"""
import copy
import json
import logging
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


@contextmanager
def isolated_evaluation(model: torch.nn.Module):
    """隔离额外评估的随机流并恢复各模块模式，不改变参数、梯度或优化器。"""
    modes = [(module, module.training) for module in model.modules()]
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            model.eval()
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        for module, mode in modes:
            module.training = mode


class Moments:
    """以第0轴为样本轴合并Welford矩；内存仅与单条记录维度有关。"""
    def __init__(self):
        self.count, self.mean, self.m2 = 0, None, None

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        if not len(values) or not np.isfinite(values).all():
            raise FloatingPointError('诊断输入为空或含NaN/Inf，停止生成误导统计')
        n, mean = len(values), values.mean(0)
        m2 = np.square(values - mean).sum(0)
        if self.count == 0:
            self.mean, self.m2, self.count = mean, m2, n
            return
        if mean.shape != self.mean.shape:
            raise ValueError('流式统计维度在不同batch间发生变化')
        delta, total = mean - self.mean, self.count + n
        self.m2 += m2 + delta ** 2 * self.count * n / total
        self.mean += delta * n / total
        self.count = total

    def std(self) -> np.ndarray:
        return np.sqrt(np.maximum(self.m2 / self.count, 0.))


def vector_summary(values: np.ndarray) -> dict:
    """小型向量精确描述统计；所有标准差均采用ddof=0。"""
    values = np.asarray(values, dtype=np.float64)
    if not values.size or not np.isfinite(values).all():
        raise FloatingPointError('诊断摘要遇到空值或非有限值')
    return dict(mean=float(values.mean()), median=float(np.median(values)),
                min=float(values.min()), max=float(values.max()))


class BoundedDistribution:
    """posterior标量分布的流式直方图；中位数近似且记录分辨率，不缓存全量张量。"""
    def __init__(self, low: float, high: float, bins: int = 4096):
        self.low, self.high, self.bins = low, high, bins
        self.hist = np.zeros(bins, dtype=np.int64)
        self.count, self.total, self.minimum, self.maximum = 0, 0., None, None

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if not values.size or not np.isfinite(values).all():
            raise FloatingPointError('posterior分布统计遇到非有限值')
        if values.min() < self.low - 1e-6 or values.max() > self.high + 1e-6:
            raise ValueError('posterior统计超出配置的logvar边界')
        clipped = np.clip(values, self.low, self.high)
        self.hist += np.histogram(clipped, bins=self.bins, range=(self.low, self.high))[0]
        self.count += len(values)
        self.total += float(values.sum())
        self.minimum = float(values.min()) if self.minimum is None else min(self.minimum, float(values.min()))
        self.maximum = float(values.max()) if self.maximum is None else max(self.maximum, float(values.max()))

    def summary(self) -> dict:
        width = (self.high - self.low) / self.bins
        cumulative = self.hist.cumsum()
        positions = [(self.count - 1) // 2 + 1, self.count // 2 + 1]
        centers = [self.low + (np.searchsorted(cumulative, p) + .5) * width for p in positions]
        median = self.minimum if self.minimum == self.maximum else float(np.mean(centers))
        return dict(mean=self.total / self.count, median=median, min=self.minimum, max=self.maximum,
                    count=self.count, median_method='4096等宽箱近似，常量分布精确', median_bin_width=width)


class MediatorDiagnostics:
    """完整评估流的中介统计；B为样本、S为层次MC、D为潜变量维度。"""
    def __init__(self, config):
        self.branches = {}
        for name in ('cns', 'ans'):
            self.branches[name] = dict(mu=Moments(), draw_mean=Moments(), draw_by_slot=Moments(),
                mu_mc=Moments(), draw_mc=Moments(), latent=Moments(), classes=[Moments(), Moments()],
                std=BoundedDistribution(np.exp(.5 * config.logvar_min), np.exp(.5 * config.logvar_max)),
                variance=BoundedDistribution(np.exp(config.logvar_min), np.exp(config.logvar_max)))
        self.samples = None

    def update(self, output: dict, labels: torch.Tensor) -> None:
        y = labels.detach().cpu().numpy()
        for name, suffix in [('cns', 'c'), ('ans', 'a')]:
            state = self.branches[name]
            mu, logvar, draw = [output[f'{key}_{suffix}'].detach().cpu().double().numpy()
                               for key in ('mu', 'logvar', 'm')]
            # CNS参数[B,D]，ANS条件参数[B,S,D]，采样均为[B,S,D]。
            if draw.ndim != 3 or len(draw) != len(y) or mu.shape != logvar.shape:
                raise ValueError('诊断中介形状不符合[B,D]/[B,S,D]约定')
            conditional = mu.ndim == 3
            if mu.shape != (draw.shape if conditional else (len(y), draw.shape[-1])):
                raise ValueError('中介均值轴与实际采样轴不一致')
            self.samples = draw.shape[1]
            location = mu.mean(1) if conditional else mu
            state['mu'].update(location)
            state['draw_mean'].update(draw.mean(1))
            state['draw_by_slot'].update(draw)
            state['mu_mc'].update(mu.std(1) if conditional else np.zeros_like(mu))
            state['draw_mc'].update(draw.std(1))
            state['latent'].update(location.std(-1)[:, None])
            state['std'].update(np.exp(.5 * logvar))
            state['variance'].update(np.exp(logvar))
            for label in (0, 1):
                if (y == label).any():
                    state['classes'][label].update(location[y == label])

    def finish(self) -> dict:
        result = {'S': self.samples, 'std_ddof': 0,
                  'axes': '跨样本std_N(E_S mu)；条件变化E_N std_S(mu)；CNS参数无S轴',
                  'note': 'ANS均值含有限MC误差；采样变化不能证明携带输入信息；类别差异不是因果效应'}
        for name, state in self.branches.items():
            spread = state['mu'].std()
            classes = state['classes']
            class_info = {f'class{label}': dict(count=c.count, mean_per_dimension=c.mean.tolist() if c.count else None)
                          for label, c in enumerate(classes)}
            distance, fisher = None, None
            if all(c.count for c in classes):
                difference = classes[0].mean - classes[1].mean
                distance = float(np.linalg.norm(difference))
                within = float(np.mean(classes[0].std() ** 2 + classes[1].std() ** 2))
                # 零类内方差时不伪造无穷大的“可分性”。
                fisher = float(np.mean(difference ** 2) / within) if within > 0 else None
            result[name] = dict(count=state['mu'].count,
                posterior_mean_per_dimension=state['mu'].mean.tolist(),
                posterior_mean_between_sample_std_per_dimension=spread.tolist(),
                posterior_mean_between_sample_std=vector_summary(spread),
                variable_dimension_fraction_at_1e_3=float((spread >= 1e-3).mean()),
                between_mean_to_posterior_std_rms_ratio=float(np.sqrt(np.mean(spread ** 2)) /
                    np.sqrt(state['variance'].total / state['variance'].count)),
                within_sample_latent_dimension_std=state['latent'].mean.item(),
                conditional_mu_within_sample_mc_std=vector_summary(state['mu_mc'].mean),
                posterior_std=state['std'].summary(), posterior_variance=state['variance'].summary(),
                sampled_mean_between_sample_std=vector_summary(state['draw_mean'].std()),
                sampled_between_sample_std_by_draw=vector_summary(state['draw_by_slot'].std().mean(0)),
                sampled_within_sample_std=vector_summary(state['draw_mc'].mean),
                within_sample_mc_available=self.samples > 1,
                class_means=class_info, class_mean_distance_l2=distance,
                class_fisher_ratio=fisher, fisher_note='两类均存在且类内方差非零才定义',
                low_mean_variation=bool(spread.mean() < 1e-3))
        return result


class PredictionDiagnostics:
    """只累计2×2计数、概率矩和熵，不保存全量窗口或参考组合。"""
    def __init__(self):
        self.paths = {key: dict(cm=np.zeros((2, 2), dtype=np.int64), moments=Moments(), entropy=0.)
                      for key in ('primary', 'factual')}
        self.disagree, self.difference, self.count = 0, 0., 0

    def update(self, primary: torch.Tensor, factual: torch.Tensor, labels: torch.Tensor) -> None:
        y = labels.detach().cpu().long().numpy()
        arrays = [p.detach().cpu().double().numpy() for p in (primary, factual)]
        for state, p in zip(self.paths.values(), arrays):
            if p.shape != (len(y), 2) or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any() or not np.allclose(p.sum(-1), 1., atol=1e-5):
                raise FloatingPointError('分类诊断概率无效')
            state['cm'] += np.bincount(2 * y + p.argmax(-1), minlength=4).reshape(2, 2)
            state['moments'].update(p)
            state['entropy'] += float(-(p * np.log(np.maximum(p, np.finfo(float).tiny))).sum())
        self.disagree += int((arrays[0].argmax(-1) != arrays[1].argmax(-1)).sum())
        self.difference += float(np.abs(arrays[0] - arrays[1]).sum(-1).sum())
        self.count += len(y)

    def finish(self) -> dict:
        result = {}
        for name, state in self.paths.items():
            cm = state['cm']
            predicted = cm.sum(0)
            result[name] = dict(confusion_matrix=cm.tolist(), true_counts=cm.sum(1).tolist(),
                predicted_counts=predicted.tolist(), probability_mean=state['moments'].mean.tolist(),
                probability_std=state['moments'].std().tolist(), mean_entropy=state['entropy'] / self.count,
                single_class=bool(predicted.min() == 0),
                extreme_imbalance=bool(predicted.min() / self.count < .01), imbalance_threshold=.01)
        result.update(factual_primary_disagreement=self.disagree / self.count,
                      factual_primary_probability_l1=self.difference / self.count)
        return result


class GradientDiagnostics:
    """backward后、裁剪与step前读取模块梯度；缺失梯度与精确零梯度分开记录。"""
    def __init__(self, model: torch.nn.Module):
        prefixes = {'cns_mediator': 'frontdoor.cns.', 'ans_mediator': 'frontdoor.ans.',
                    'cns2ans': 'frontdoor.ans.cns_current.', 'outcome': 'frontdoor.outcome.',
                    'reference': 'frontdoor.reference_encoder.', 'cns_backbone': 'backbone.cns_enc.',
                    'ans_backbone': 'backbone.ans_enc.', 'graph': 'backbone.dag_adj'}
        self.groups = {key: [p for name, p in model.named_parameters() if p.requires_grad and name.startswith(prefix)
                            and not (key == 'ans_mediator' and name.startswith(prefixes['cns2ans']))]
                       for key, prefix in prefixes.items()}
        self.records = {key: [] for key in prefixes}
        self.missing = {key: [] for key in prefixes}

    def update(self) -> None:
        for name, params in self.groups.items():
            grads = [p.grad.detach() for p in params if p.grad is not None]
            self.missing[name].append(len(params) - len(grads))
            norm = float(torch.stack([g.double().square().sum() for g in grads]).sum().sqrt()) if grads else 0.
            if not np.isfinite(norm):
                logger.error('模块%s梯度出现NaN/Inf，停止更新参数', name)
                raise FloatingPointError(f'{name}梯度非有限，发生在裁剪/optimizer.step之前')
            self.records[name].append(norm)

    def finish(self) -> dict:
        return {name: dict(active=bool(self.groups[name]), parameter_tensors=len(self.groups[name]),
                 pre_clip_l2=vector_summary(values), zero_batches=sum(v == 0 for v in values),
                 batches=len(values), missing_grad_tensors_mean=float(np.mean(self.missing[name])),
                 all_zero=bool(self.groups[name]) and all(v == 0 for v in values))
                for name, values in self.records.items()}


def _replace_epoch(path: Path, record: dict) -> list[dict]:
    """续训时保留更早epoch，替换当前并去除回退点之后的旧诊断，不伪造空缺。"""
    records = {}
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            if line.strip():
                row = json.loads(line)
                if row['epoch'] < record['epoch']:
                    records[row['epoch']] = row
    records[record['epoch']] = record
    ordered = [records[e] for e in sorted(records)]
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in ordered), encoding='utf-8')
    temporary.replace(path)
    return ordered


def save_epoch_diagnostics(folder: Path, record: dict, mediators: dict) -> dict:
    """四类产物只保存统计；全量中介向量仅在mediator文件出现一次。"""
    # 历史/summary/checkpoint复用原metrics；诊断连续计数不能反向修改这些对象。
    record = copy.deepcopy(record)
    collapse_path = folder / 'collapse_diagnostics.jsonl'
    previous = []
    if collapse_path.exists():
        previous = [json.loads(line) for line in collapse_path.read_text(encoding='utf-8').splitlines() if line.strip()]
        previous = [row for row in previous if row['epoch'] == record['epoch'] - 1]
    collapse = dict(epoch=record['epoch'], fold=record['fold'], smoke=record['smoke'],
                    complete=record['complete'], prediction_path=record['prediction_path'])
    for stage in ('train', 'test'):
        diagnostic = record[stage]['prediction_diagnostics']
        old = previous[-1][stage]['single_class_streak'] if previous else 0
        diagnostic['single_class_streak'] = old + 1 if diagnostic['primary']['single_class'] else 0
        diagnostic['mean_probability_change_from_previous_epoch'] = (
            (np.asarray(diagnostic['primary']['probability_mean']) -
             np.asarray(previous[-1][stage]['primary']['probability_mean'])).tolist() if previous else None)
        collapse[stage] = diagnostic
        old_primary = previous[-1][stage]['primary'] if previous else {}
        if diagnostic['primary']['single_class'] and (old == 0 or
                diagnostic['primary']['predicted_counts'].index(0) != old_primary.get('predicted_counts', [0, 0]).index(0)):
            logger.warning('%s单类别预测：fold=%s epoch=%s 预测=%s 连续%s轮（single-class prediction detected）',
                           '训练集' if stage == 'train' else '测试集', record['fold'], record['epoch'],
                           diagnostic['primary']['predicted_counts'], diagnostic['single_class_streak'])
        elif diagnostic['primary']['extreme_imbalance'] and not diagnostic['primary']['single_class'] and not old_primary.get('extreme_imbalance'):
            logger.warning('%s预测极度失衡：最少类别比例<1%%；该阈值仅用于诊断', stage)
    gradient = record['optimization'].get('gradients', {})
    for name, values in gradient.items():
        prior = previous[-1].get('zero_gradient_streak', {}).get(name, 0) if previous else 0
        streak = prior + 1 if values['all_zero'] else 0
        collapse.setdefault('zero_gradient_streak', {})[name] = streak
        if values['all_zero'] and prior == 0:
            logger.warning('模块%s该轮所有batch梯度范数为零，连续%s轮；需结合缺失梯度和消融配置判断', name, streak)
    records = _replace_epoch(folder / 'epoch_metrics.jsonl', record)
    _replace_epoch(collapse_path, collapse)
    _replace_epoch(folder / 'mediator_diagnostics.jsonl', dict(epoch=record['epoch'], fold=record['fold'],
                   level=record['diagnostic_level'], complete=record['complete'], **mediators))
    temporary = folder / 'epoch_confusion_matrices.npz.tmp'
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, epochs=np.array([r['epoch'] for r in records], dtype=np.int64),
            train=np.array([r['train_confusion_matrix'] for r in records], dtype=np.int64),
            test=np.array([r['test_confusion_matrix'] for r in records], dtype=np.int64),
            complete=np.array([r['complete'] for r in records], dtype=bool))
    temporary.replace(folder / 'epoch_confusion_matrices.npz')
    return collapse


def log_epoch_summary(record: dict, collapse: dict, total_epochs: int, interval: int = 10) -> None:
    """每轮一行含两组紧凑矩阵；周期只打印关键量，不倾倒完整字典。"""
    train, test = record['train'], record['test']
    counts = test['prediction_diagnostics']['primary']['predicted_counts']
    compact = lambda cm: f'[[{cm[0][0]},{cm[0][1]}],[{cm[1][0]},{cm[1][1]}]]'
    logger.info('fold=%s 轮次%03d/%03d | Loss=%.6f | 训练ACC=%.4f F1=%.4f CM=%s | '
                '测试ACC=%.4f F1=%.4f CM=%s | 测试预测=%s 单类连续=%s | 保存最佳=%s 完整=%s',
                record['fold'], record['epoch'], total_epochs, record['optimization']['loss'],
                train['acc'], train['selection_f1'], compact(record['train_confusion_matrix']),
                test['acc'], test['selection_f1'], compact(record['test_confusion_matrix']),
                counts, collapse['test']['single_class_streak'], record['selected_best'], record['complete'])
    if interval and record['epoch'] % interval == 0:
        primary = test['prediction_diagnostics']['primary']
        logger.info('周期诊断：事实训练/测试ACC=%.4f/%.4f；两路径测试不一致率=%.4f；'
                    '测试P1均值/std=%.6f/%.3e；CE=%.6f CNS_KL=%.3e ANS_KL=%.3e',
                    train['prediction_diagnostics']['factual']['metrics']['acc'],
                    test['prediction_diagnostics']['factual']['metrics']['acc'],
                    test['prediction_diagnostics']['factual_primary_disagreement'],
                    primary['probability_mean'][1], primary['probability_std'][1],
                    record['optimization']['factual_ce'], record['optimization']['kl_c'], record['optimization']['kl_a'])
        if record['diagnostic_level'] == 'full':
            summary = record['mediator_summary']['test']
            grad = record['optimization']['gradients']
            logger.info('周期中介诊断：测试mu跨样本std CNS=%.3e ANS=%.3e；posterior std CNS=%.4f ANS=%.4f；'
                        '裁剪前梯度均值 CNS=%.3e ANS=%.3e CNS→ANS=%.3e；完整统计见JSONL',
                        summary['cns']['posterior_mean_between_sample_std']['mean'],
                        summary['ans']['posterior_mean_between_sample_std']['mean'],
                        summary['cns']['posterior_std']['mean'], summary['ans']['posterior_std']['mean'],
                        grad['cns_mediator']['pre_clip_l2']['mean'], grad['ans_mediator']['pre_clip_l2']['mean'],
                        grad['cns2ans']['pre_clip_l2']['mean'])


def log_evaluation(stage: str, metrics: dict, matrix: torch.Tensor) -> None:
    """一条logging记录保留完整多行矩阵；正类改变指标定义，不改变矩阵顺序。"""
    p = metrics['prediction_diagnostics']['primary']
    factual = metrics['prediction_diagnostics']['factual']
    logger.info('[%s] 主路径=%s 完整评估=%s 正类=%s\n混淆矩阵（行=真实，列=预测，顺序[0,1]）：\n%s\n'
                'ACC=%.4f Precision=%.4f Recall=%.4f Specificity=%.4f F1=%.4f\n'
                '真实计数=%s 预测计数=%s 平均概率=%s 概率标准差=%s 平均熵=%.6f\n'
                '事实路径对照矩阵：\n%s\n事实ACC=%.4f 事实F1=%.4f 两路径判决不一致率=%.4f',
                stage, metrics['prediction_path'], metrics['complete_evaluation'], metrics['selection_positive_class'],
                np.array2string(matrix.cpu().numpy()), metrics['acc'], metrics['precision'], metrics['recall'],
                metrics['specificity'], metrics['selection_f1'], p['true_counts'], p['predicted_counts'],
                p['probability_mean'], p['probability_std'], p['mean_entropy'],
                np.array2string(np.asarray(factual['confusion_matrix'])), factual['metrics']['acc'],
                factual['metrics']['selection_f1'], metrics['prediction_diagnostics']['factual_primary_disagreement'])
