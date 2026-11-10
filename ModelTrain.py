"""LOSO二分训练：事实监督、训练总体参考、held-out测试选择checkpoint。"""
import json
import logging
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from Datasets import (DATASET_SPECS, PhysiologicalWindows, WindowSelection,
                      cal_train_index, canonical_dataset, canonical_label, make_loaders, seed_worker)
from NeuroFDAN import NeuroFDAN
from frontdoor_scm import FrontDoorConfig, ReferenceBank, frontdoor_loss
from training_diagnostics import (GradientDiagnostics, MediatorDiagnostics, PredictionDiagnostics,
                                  isolated_evaluation, log_epoch_summary, save_epoch_diagnostics)
from run_reporting import source_version, write_run_reports

logger = logging.getLogger(__name__)


def seed_everything(seed=42):
    if seed != 42:
        raise ValueError('本阶段真实训练固定使用seed=42')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(value='auto'):
    value = str(value)
    if value.isdigit():
        value = f'cuda:{value}' if torch.cuda.is_available() else 'cpu'
    if value == 'auto':
        value = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(value)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('指定了CUDA，但当前环境不可用；可显式使用--device cpu')
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('当前训练入口仅支持CPU或CUDA')
    return device


def optimizer_weight_decay(args):
    """旧Namespace没有此字段时沿用原值，不隐式关闭耦合L2。"""
    value = float(getattr(args, 'weight_decay', .01))
    if not math.isfinite(value) or value < 0:
        raise ValueError('weight_decay必须为有限非负数')
    return value


def build_optimizer(model, args):
    """沿用原单组Adam及全部参数；仅公开已有耦合L2配置。"""
    return torch.optim.Adam(model.parameters(), lr=args.lr, betas=(.9, .999), eps=1e-8,
                            weight_decay=optimizer_weight_decay(args), amsgrad=False)


def validate_resume_weight_decay(state, args):
    """保存的优化器会覆盖构造配置，故禁止以续训静默切换衰减系数。"""
    expected = optimizer_weight_decay(args)
    groups = state['optimizer']['param_groups']
    if not groups or any(group.get('weight_decay') != expected for group in groups):
        raise ValueError('checkpoint的Adam weight_decay与本次参数不一致；对照必须新run从头训练')


def parameter_rms(model):
    """只读各参数RMS；full诊断保留逐参数尺度，避免模块范数掩盖卷积核收缩。"""
    result = {}
    with torch.no_grad():
        for name, value in model.named_parameters():
            rms = float(value.detach().double().square().mean().sqrt())
            if not math.isfinite(rms):
                raise FloatingPointError(f'参数RMS含NaN/Inf：{name}')
            result[name] = rms
    return result


def confusion_metrics(confusion, dataset):
    """混淆矩阵行是真值、列是预测；两种正类F1均记录，选择指标保持旧设置。"""
    cm = torch.as_tensor(confusion).cpu().long()
    count = int(cm.sum())
    if cm.shape != (2, 2) or count < 1:
        raise ValueError('评价需要非空二分类混淆矩阵')
    scores = {}
    for positive in (0, 1):
        tp = int(cm[positive, positive])
        fp = int(cm[:, positive].sum()) - tp
        fn = int(cm[positive].sum()) - tp
        scores[f'f1_class{positive}'] = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.
    scores.update(acc=float(cm.diag().sum()) / count, count=count, correct=int(cm.diag().sum()))
    positive = 1 if canonical_dataset(dataset) == 'DEAP' else 0
    scores['selection_f1'] = scores[f'f1_class{positive}']
    scores['selection_positive_class'] = positive
    tp, tn = int(cm[positive, positive]), int(cm[1-positive, 1-positive])
    fp, fn = int(cm[1-positive, positive]), int(cm[positive, 1-positive])
    scores.update(precision=tp / (tp + fp) if tp + fp else 0.,
                  recall=tp / (tp + fn) if tp + fn else 0.,
                  sensitivity=tp / (tp + fn) if tp + fn else 0.,
                  specificity=tn / (tn + fp) if tn + fp else 0.)
    return scores


@dataclass
class BestCheckpoint:
    """原规则：ACC上升且>0.5；ACC相等时按对应数据集F1上升保存。"""
    acc: float = 0.
    f1: float = 0.
    epoch: int = 0

    def consider(self, metrics, epoch):
        acc, f1 = metrics['acc'], metrics['selection_f1']
        better = (acc > self.acc and acc > .5) or (acc == self.acc and f1 > self.f1)
        if better:
            self.acc, self.f1, self.epoch = acc, f1, epoch
        return better


def _assert_reference_sources(ids, train_subjects, held_out):
    observed = set(ids[:, 0].cpu().tolist())
    if held_out in train_subjects or len(set(train_subjects)) != len(train_subjects) or held_out in observed or not observed.issubset(set(train_subjects)):
        raise ValueError('参考来源检查失败：含held-out或非训练被试')


def build_reference_bank(population, train_subjects, held_out, N_ref=2048, seed=42):
    """从未平衡的全部训练窗口抽取经验库；只复用Phase 1来源校验与抽样。"""
    if not isinstance(population, PhysiologicalWindows):
        raise ValueError('参考总体必须是未做标签下采样的PhysiologicalWindows')
    if N_ref < 1 or not len(population):
        raise ValueError('参考容量和训练总体必须非空')
    _assert_reference_sources(population.source_ids, train_subjects, held_out)
    bank = ReferenceBank(population.x_c, population.x_a, population.source_ids,
                         forbidden_subjects=(held_out,), max_size=N_ref, seed=seed,
                         dataset=population.dataset)
    bank.raw_windows = True
    bank.population_count = len(population)
    _assert_reference_sources(bank.source_ids, train_subjects, held_out)
    bank.report = {
        'dataset': population.dataset, 'train_subjects': list(train_subjects), 'held_out_subject': held_out,
        'population_windows': len(population), 'population_subjects': len(set(population.source_ids[:, 0].tolist())),
        'requested_N_ref': N_ref, 'actual_N_ref': bank.used_count,
        'actual_subject_count': len(set(bank.source_ids[:, 0].tolist())),
        'shortfall': bank.shortfall, 'source_check_passed': True,
        'population': '当前fold全部未做标签平衡的训练窗口',
        'sampling': '库成员无放回均匀抽取；积分参考有放回并排除当前来源',
    }
    return bank


def cache_reference_features(model, raw_bank, device, batch_size=40):
    """每次评估前按当前权重重建缓存；不缓存训练期可学习表示。"""
    was_training = model.training
    model.eval()
    c, a = [], []
    with torch.no_grad():
        for start in range(0, raw_bank.used_count, batch_size):
            hc, ha = model.backbone(raw_bank.x_c[start:start + batch_size].to(device),
                                     raw_bank.x_a[start:start + batch_size].to(device))
            c.append(hc); a.append(ha)
    bank = ReferenceBank(torch.cat(c), torch.cat(a), raw_bank.source_ids.to(device),
                         forbidden_subjects=(raw_bank.report['held_out_subject'],), seed=42,
                         dataset=raw_bank.dataset)
    bank.shortfall = raw_bank.shortfall
    bank.population_count = raw_bank.population_count
    bank.feature_version, bank.model_identity = model.feature_version, id(model)
    bank.parameter_versions = tuple(p._version for p in model.parameters())
    bank.report = dict(raw_bank.report, feature_version=model.feature_version)
    model.train(was_training)
    return bank


def _record_sources(path, bank):
    records = [dict(subject_id=s, trial_id=t, window_id=w, dataset=bank.dataset)
               for s, t, w in bank.source_ids.cpu().tolist()]
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')


def _add_confusion(confusion, y, probs):
    prediction = probs.detach().argmax(-1)
    confusion += torch.bincount((y.detach().cpu().long() * 2 + prediction.cpu()), minlength=4).reshape(2, 2)


def prediction_probabilities(outputs, fd_mode):
    """训练诊断、测试、选模始终使用同一输出；消融明确改用观察路径。"""
    probs = outputs['probs_do'] if fd_mode == 'full' else outputs['probs_obs']
    if probs.ndim != 2 or probs.shape[1] != 2 or not torch.isfinite(probs).all():
        raise FloatingPointError('主输出必须为有限的[B,2]概率；发现NaN/Inf或形状错误')
    if not ((probs >= 0) & (probs <= 1)).all() or not torch.allclose(
            probs.sum(-1), torch.ones_like(probs[:, 0]), atol=1e-5, rtol=1e-5):
        raise FloatingPointError('主输出概率范围或行归一化错误，停止记录指标')
    return probs


def train_one_epoch(model, loader, raw_bank, optimizer, device, dataset, epoch,
                    fd_mode='full', lambda_obs=1., fire_weight=0., max_batches=0,
                    diagnostic_level='basic'):
    model.train()
    gradients = GradientDiagnostics(model) if diagnostic_level == 'full' else None
    cm, sums, count = torch.zeros(2, 2, dtype=torch.long), {}, 0
    for batch_index, (batch, _) in enumerate(loader):
        xc, xa, y = batch['x_c'].to(device), batch['x_a'].to(device), batch['Labels'].to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(xc, xa, reference_bank=raw_bank if fd_mode == 'full' else None,
                       source_ids=batch['source_ids'].to(device), fd_mode=fd_mode)
        primary_probs = prediction_probabilities(output, fd_mode)
        terms = frontdoor_loss(output, y, model.frontdoor_config, epoch=epoch)
        # 真实观察标签从不用于监督probs_do，也不传入任何oracle概率。
        loss = terms['loss'] + (lambda_obs - 1.) * terms['loss_obs'] + fire_weight * output['fire_loss']
        if not torch.isfinite(loss):
            raise FloatingPointError('训练损失出现非有限数值，请检查输入和模型设置')
        loss.backward()
        if gradients is not None:
            gradients.update()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
        optimizer.step()
        model.weights_updated()
        _add_confusion(cm, y, primary_probs)
        values = dict(output['mediator_statistics'], loss=loss.detach(),
                      factual_ce=terms['loss_obs'].detach(), kl=terms['loss_kl'].detach(),
                      fire_loss=output['fire_loss'].detach())
        size = len(y)
        for key, value in values.items():
            sums[key] = sums.get(key, 0.) + float(value) * size
        count += size
        if max_batches and batch_index + 1 >= max_batches:
            break
    if not count:
        raise ValueError('训练没有执行任何batch')
    metrics = dict(confusion_metrics(cm, dataset), **{k: v / count for k, v in sums.items()})
    if gradients is not None:
        metrics['gradients'] = gradients.finish()
        metrics['parameter_rms'] = parameter_rms(model)
    return metrics, cm


def evaluate(model, loader, raw_bank, device, dataset, fd_mode='full', max_batches=0, *,
             diagnostic_level='basic', cached_bank=None, cache_holder=None, collect_predictions=True):
    model.eval()
    cached = cached_bank
    if fd_mode == 'full' and cached is None:
        cached = cache_reference_features(model, raw_bank, device, loader.batch_size)
    if cache_holder is not None:
        cache_holder['bank'] = cached
    diagnostics = PredictionDiagnostics()
    mediators = MediatorDiagnostics(model.frontdoor_config) if diagnostic_level == 'full' else None
    cm, predictions, sums, count = torch.zeros(2, 2, dtype=torch.long), [], {}, 0
    # 固定评估MC随机流并恢复训练随机流，减少不同epoch的选模采样噪声。
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices), torch.no_grad():
        torch.manual_seed(42)
        for batch_index, (batch, _) in enumerate(loader):
            y = batch['Labels'].to(device)
            output = model(batch['x_c'].to(device), batch['x_a'].to(device), reference_bank=cached,
                           source_ids=batch['source_ids'].to(device), fd_mode=fd_mode)
            probs = prediction_probabilities(output, fd_mode)
            _add_confusion(cm, y, probs)
            diagnostics.update(probs, output['probs_obs'], y)
            if mediators is not None:
                mediators.update(output, y)
            for key, value in output['mediator_statistics'].items():
                sums[key] = sums.get(key, 0.) + float(value) * len(y)
            sums['factual_ce'] = sums.get('factual_ce', 0.) + float(torch.nn.functional.cross_entropy(output['logits_obs'], y)) * len(y)
            count += len(y)
            if collect_predictions:
                for ids, truth, probability in zip(batch['source_ids'].tolist(), y.cpu().tolist(), probs.cpu().tolist()):
                    predictions.append({'subject_id': ids[0], 'trial_id': ids[1], 'window_id': ids[2],
                                        'dataset': dataset, 'label': truth, 'probabilities': probability})
            if max_batches and batch_index + 1 >= max_batches:
                break
    metrics = confusion_metrics(cm, dataset)
    metrics.update({k: v / count for k, v in sums.items()})
    metrics['complete_test'] = count == len(loader.dataset)
    metrics.update(complete_evaluation=metrics['complete_test'],
                   prediction_path='probs_do' if fd_mode == 'full' else 'probs_obs',
                   prediction_diagnostics=diagnostics.finish())
    for path in ('primary', 'factual'):
        entry = metrics['prediction_diagnostics'][path]
        entry['metrics'] = confusion_metrics(entry['confusion_matrix'], dataset)
    if mediators is not None:
        metrics['mediator_diagnostics'] = mediators.finish()
    return metrics, cm, predictions


def make_train_evaluation_loader(dataset, batch_size=40, workers=0):
    """完整优化训练子集，只读、不打乱、不丢尾batch，独立于优化和测试加载器。"""
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                      num_workers=workers, worker_init_fn=seed_worker,
                      generator=torch.Generator().manual_seed(42))


def evaluate_training_set(model, loader, raw_bank, device, dataset, fd_mode='full', max_batches=0, **kwargs):
    """额外诊断不step、不更新BN，恢复模式与全局随机流；调用方提供独立加载器。"""
    generator_state = loader.generator.get_state() if loader.generator is not None else None
    try:
        with isolated_evaluation(model):
            return evaluate(model, loader, raw_bank, device, dataset, fd_mode, max_batches,
                            collect_predictions=False, **kwargs)
    finally:
        if generator_state is not None:
            loader.generator.set_state(generator_state)


def save_checkpoint(path, model, optimizer, epoch, best, args, raw_bank, metrics):
    """保存模型/优化器/配置和真实来源；不保存数据或带图缓存。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {'format_version': 2, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
               'epoch': epoch, 'best': asdict(best), 'arguments': vars(args),
               'frontdoor_config': asdict(model.frontdoor_config), 'metrics': metrics,
               'reference_report': raw_bank.report, 'reference_source_ids': raw_bank.source_ids.cpu()}
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path, model, optimizer=None, map_location='cpu'):
    """仅加载本项目自身产生的checkpoint；map_location支持CPU/CUDA切换。"""
    state = torch.load(path, map_location=map_location, weights_only=False)
    if state.get('format_version') != 2 or state['frontdoor_config'] != asdict(model.frontdoor_config):
        raise ValueError('checkpoint格式或模型配置不匹配')
    model.load_state_dict(state['model'])
    model.weights_updated()
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer'])
    return state


def save_fold_confusion(run_dir, fold_dir, dataset, label, held_out, confusion):
    """Linux服务器不同fold并行时加锁更新汇总，避免覆盖其他被试行。"""
    import fcntl
    run_dir, fold_dir = Path(run_dir), Path(fold_dir)
    np.save(fold_dir / 'best_confusion.npy', confusion.cpu().numpy())
    path = run_dir / f'NeuroFDAN-{dataset}_{label}_cm.npy'
    with (run_dir / '.matrix.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        matrices = np.load(path) if path.exists() else np.zeros((DATASET_SPECS[dataset]['subjects'], 2, 2), dtype=np.int64)
        matrices[held_out - 1] = confusion.cpu().numpy()
        temporary = path.with_suffix('.npy.tmp')
        with temporary.open('wb') as stream:
            np.save(stream, matrices)
        temporary.replace(path)


def frontdoor_config(args):
    return FrontDoorConfig(mediator_dim=args.mediator_dim, hidden_dim=args.hidden_dim,
                          dropout=args.dropout, S_train=args.S_train, K_train=args.K_train,
                          S_eval=args.S_eval, K_eval=args.K_eval, N_ref=args.N_ref,
                          beta=args.beta, beta_warmup_epochs=args.beta_warmup_epochs,
                          ref_dim=args.ref_dim, no_cns2ans=args.no_cns2ans,
                          reference_chunk_size=args.reference_chunk_size, time_steps=args.time_steps,
                          logvar_min=args.logvar_min, logvar_max=args.logvar_max)


def run_fold(args, held_out, *, population=None, test_dataset=None, model=None, optimizer=None):
    """额外对象仅供轻量级接口测试和兼容调用注入，不提供替代真实实验模式。"""
    seed_everything(args.seed)
    optimizer_weight_decay(args)
    dataset, label = canonical_dataset(args.dataset), canonical_label(args.label)
    spec = DATASET_SPECS[dataset]
    train_subjects = cal_train_index(held_out, spec['subjects'])
    population = population if population is not None else PhysiologicalWindows(args.root_dir, dataset, train_subjects, label)
    test_dataset = test_dataset if test_dataset is not None else PhysiologicalWindows(args.root_dir, dataset, [held_out], label)
    if any(data.dataset != dataset or data.label != label for data in (population, test_dataset)):
        raise ValueError('训练/测试数据集或任务与当前fold配置不匹配')
    _assert_reference_sources(population.source_ids, train_subjects, held_out)
    if set(test_dataset.source_ids[:, 0].tolist()) != {held_out}:
        raise ValueError('测试数据来源并非当前held-out被试')
    selected = population.balanced_indices(args.seed) if not args.no_balance else torch.arange(len(population))
    train_dataset = WindowSelection(population, selected)
    train_loader, test_loader = make_loaders(train_dataset, test_dataset, args.batch_size, args.workers, args.seed)
    train_eval_loader = make_train_evaluation_loader(train_dataset, args.batch_size, args.workers)
    diagnostic_level = getattr(args, 'diagnostic_level', 'basic')
    raw_bank = build_reference_bank(population, train_subjects, held_out, args.N_ref, args.seed)
    device = resolve_device(args.device)
    model = model if model is not None else NeuroFDAN(ecg_channels=2 if dataset == 'DEAP' else 3,
                                                       frontdoor_config=frontdoor_config(args))
    model.to(device)
    optimizer = optimizer if optimizer is not None else build_optimizer(model, args)
    run_name = args.run_name or f'{args.fd_mode}_cns_{"off" if args.no_cns2ans else "on"}_ref{args.ref_dim}'
    run_dir = Path(args.output_dir) / dataset / label / 'seed42' / run_name
    fold_dir = run_dir / f'fold{held_out:02d}'
    fold_dir.mkdir(parents=True, exist_ok=True)
    if (fold_dir / 'fold.json').exists() and not args.resume:
        raise FileExistsError('该fold已存在；请使用不同--run_name，或显式--resume，不覆盖既有实验')
    if args.resume and Path(args.resume).resolve().parent != fold_dir.resolve():
        raise ValueError('续训必须使用同一输出目录和run_name，避免继承最佳指标却遗漏原best文件')
    fold_config = dict(raw_bank.report, optimization_windows=len(train_dataset),
                       train_drop_last_windows=len(train_dataset) % args.batch_size,
                       test_windows=len(test_dataset), no_validation=True, seed=args.seed,
                       prediction='probs_do' if args.fd_mode == 'full' else 'probs_obs',
                       smoke=bool(args.max_train_batches or args.max_test_batches),
                       selection='held-out ACC上升且>0.5，ACC相等时F1上升；DEAP正类1/HCI正类0',
                       arguments=vars(args), model=asdict(model.frontdoor_config),
                       source_version=source_version(Path(__file__).resolve().parent))
    best, first_epoch = BestCheckpoint(), 1
    if args.resume:
        state = load_checkpoint(args.resume, model, optimizer, device)
        validate_resume_weight_decay(state, args)
        old = state['reference_report']
        if old['held_out_subject'] != held_out or old['dataset'] != dataset or state['arguments']['label'] != label:
            raise ValueError('checkpoint数据集、任务或held-out不匹配')
        if state['arguments']['fd_mode'] != args.fd_mode or state['arguments']['no_balance'] != args.no_balance or any(
                state['arguments'][k] != getattr(args, k) for k in ('max_train_batches', 'max_test_batches')):
            raise ValueError('checkpoint的预测模式或分类平衡设置不匹配')
        if not torch.equal(state['reference_source_ids'].cpu(), raw_bank.source_ids):
            raise ValueError('checkpoint参考来源与当前训练总体不一致')
        best, first_epoch = BestCheckpoint(**state['best']), state['epoch'] + 1
        if args.epochs < first_epoch:
            raise ValueError('指定epochs不大于checkpoint已完成轮数，没有可继续训练的epoch')
        logger.info('已加载checkpoint；继续epoch=%s；续训不是逐位等同未中断训练的承诺', first_epoch)
    fold_config['optimizer'] = dict(name=type(optimizer).__name__,
        parameter_groups=[{key: group.get(key) for key in ('lr', 'betas', 'eps', 'weight_decay', 'amsgrad')}
                          for group in optimizer.param_groups])
    if diagnostic_level == 'full':
        fold_config['start_parameter_rms'] = parameter_rms(model)
    logger.info('优化器=%s；各组weight_decay=%s；观察标签仅监督事实路径',
                type(optimizer).__name__, [group.get('weight_decay') for group in optimizer.param_groups])
    (fold_dir / 'fold.json').write_text(json.dumps(fold_config, ensure_ascii=False, indent=2), encoding='utf-8')
    _record_sources(fold_dir / 'reference_sources.json', raw_bank)
    logger.info('当前fold=%s；训练被试=%s；测试被试=%s；参考库=%s/%s；来源检查通过',
                held_out, train_subjects, held_out, raw_bank.used_count, len(population))
    logger.info('训练集评估覆盖全部%s个优化窗口（不丢尾batch）；未平衡参考总体%s窗口；诊断级别=%s',
                len(train_dataset), len(population), diagnostic_level)
    if raw_bank.shortfall:
        logger.info('参考总体不足N_ref，使用全部%s个训练窗口', raw_bank.used_count)
    if fold_config['smoke']:
        logger.info('当前是有限batch接口检查，输出不得作为正式LOSO实验结果')
    # TensorBoard只在实际运行fold时导入，命令行help不要求启动任何服务。
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(str(fold_dir / 'tensorboard'))
    fire_weight = args.fire_weight if args.fire_weight is not None else (1. if dataset == 'HCI' else 0.)
    last_metrics = None
    try:
        for epoch in range(first_epoch, args.epochs + 1):
            train_metrics, _ = train_one_epoch(model, train_loader, raw_bank, optimizer, device, dataset,
                           epoch, args.fd_mode, args.lambda_obs, fire_weight, args.max_train_batches,
                           diagnostic_level=diagnostic_level)
            cache_holder = {}
            # 原测试路径先执行；额外训练集评估随后隔离随机流，不参与选模。
            metrics, cm, predictions = evaluate(model, test_loader, raw_bank, device, dataset,
                                                args.fd_mode, args.max_test_batches,
                                                diagnostic_level=diagnostic_level, cache_holder=cache_holder)
            train_evaluation, train_cm, _ = evaluate_training_set(model, train_eval_loader, raw_bank,
                device, dataset, args.fd_mode, args.max_test_batches, diagnostic_level=diagnostic_level,
                cached_bank=cache_holder['bank'])
            del cache_holder
            mediator_records = {stage: value.pop('mediator_diagnostics', None)
                                for stage, value in [('train', train_evaluation), ('test', metrics)]}
            selected_best = best.consider(metrics, epoch)
            last_metrics = metrics
            history = dict(epoch=epoch, train=train_metrics, test=metrics, selected_best=selected_best,
                           confusion_matrix=cm.tolist(), reference=raw_bank.report)
            with (fold_dir / 'history.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(history, ensure_ascii=False) + '\n')
            for stage, values in [('训练', train_metrics), ('训练集评估', train_evaluation), ('测试', metrics)]:
                for key, value in values.items():
                    if isinstance(value, (int, float)):
                        writer.add_scalar(f'{stage}/{key}', value, epoch)
            probabilities = torch.tensor([p['probabilities'][1] for p in predictions])
            writer.add_histogram('测试/主输出正类概率', probabilities, epoch)
            writer.add_scalar('参考库/实际样本数', raw_bank.used_count, epoch)
            writer.add_scalar('参考库/样本不足', int(raw_bank.shortfall), epoch)
            writer.add_scalar('配置/切断CNS路径', int(args.no_cns2ans), epoch)
            save_checkpoint(fold_dir / 'last.pt', model, optimizer, epoch, best, args, raw_bank, metrics)
            if selected_best:
                # 保留原有含数据集/任务/被试/正确数的文件名，另提供稳定best.pt入口。
                legacy_name = f'NeuroFDAN-{dataset}-{label}-{held_out}-{metrics["correct"]}.pth'
                save_checkpoint(fold_dir / legacy_name, model, optimizer, epoch, best, args, raw_bank, metrics)
                save_checkpoint(fold_dir / 'best.pt', model, optimizer, epoch, best, args, raw_bank, metrics)
                (fold_dir / 'best_predictions.json').write_text(json.dumps(predictions, ensure_ascii=False), encoding='utf-8')
                save_fold_confusion(run_dir, fold_dir, dataset, label, held_out, cm)
            mediator_summary = {}
            for stage, diagnosis in mediator_records.items():
                mediator_summary[stage] = None if diagnosis is None else {
                    branch: {key: diagnosis[branch][key] for key in
                             ('posterior_mean_between_sample_std', 'posterior_std', 'low_mean_variation')}
                    for branch in ('cns', 'ans')}
            record = dict(epoch=epoch, fold=held_out, dataset=dataset, label=label,
                          source_version=fold_config['source_version'],
                          smoke=fold_config['smoke'], diagnostic_level=diagnostic_level,
                          complete=train_evaluation['complete_evaluation'] and metrics['complete_test'],
                          prediction_path=fold_config['prediction'],
                          training_population='实际用于优化的训练窗口子集，评估不丢尾batch',
                          optimization_windows=len(train_dataset), reference_population_windows=len(population),
                          train=train_evaluation, test=metrics, optimization=train_metrics,
                          train_confusion_matrix=train_cm.tolist(), test_confusion_matrix=cm.tolist(),
                          selected_best=selected_best, mediator_summary=mediator_summary,
                          mediator_details_file='mediator_diagnostics.jsonl')
            collapse = save_epoch_diagnostics(fold_dir, record, mediator_records)
            log_epoch_summary(record, collapse, args.epochs, getattr(args, 'log_detail_interval', 10))
    finally:
        writer.close()
    summary = dict(held_out=held_out, best=asdict(best), last_test=last_metrics,
                   has_eligible_best=best.epoch > 0, smoke=fold_config['smoke'])
    (fold_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    if best.epoch == 0:
        logger.info('该fold没有满足原ACC>0.5规则的最佳checkpoint；仅保留last.pt，不伪造best')
    try:
        write_run_reports(run_dir)
    except (OSError,ValueError,TypeError) as error:
        logger.warning('训练已结束但摘要未完成：%s；原summary/checkpoint保持原样', type(error).__name__)
    return summary


def run_loso(args):
    dataset = canonical_dataset(args.dataset)
    args.dataset, args.label = dataset, canonical_label(args.label)
    count = DATASET_SPECS[dataset]['subjects']
    subjects = args.held_out_subjects or list(range(args.start_subject, (args.end_subject or count) + 1))
    if not subjects or len(set(subjects)) != len(subjects) or any(s < 1 or s > count for s in subjects):
        raise ValueError('LOSO被试列表为空、重复或越界')
    if args.resume and len(subjects) != 1:
        raise ValueError('--resume只能用于单个fold')
    try:
        return [run_fold(args, subject) for subject in subjects]
    finally:
        run_name = args.run_name or f'{args.fd_mode}_cns_{"off" if args.no_cns2ans else "on"}_ref{args.ref_dim}'
        try:
            write_run_reports(Path(args.output_dir) / dataset / args.label / 'seed42' / run_name)
        except (OSError, ValueError, TypeError) as error:
            logger.warning('运行摘要未完成：%s；保留原异常/训练产物，请结合日志判断状态', type(error).__name__)


class DEAPModelTrainer:
    """保留旧构造名；旧元组模型应显式迁移为本阶段NeuroFDAN。"""
    dataset = 'DEAP'
    def __init__(self, model, root_dir, optimizer, scheduler=None, test_index=1,
                 batch_size=40, k_fold=32, n_epoch=20, temperature=13.544,
                 Trainer_name='Trainer', label='arousal', device_id=0):
        if scheduler is not None:
            raise ValueError('兼容入口不隐式改变原无scheduler设置，请在独立训练代码中显式接入')
        self.model, self.optimizer = model, optimizer
        self.arguments = (root_dir, test_index, batch_size, n_epoch, Trainer_name, label, device_id)

    def run(self):
        from mainNeuroFDAN import get_args
        root, held, batch, epochs, name, label, device = self.arguments
        args = get_args(argv=['--dataset', self.dataset, '--label', label, '--root_dir', str(root),
                            '--held_out_subjects', str(held), '--batch_size', str(batch),
                            '--epochs', str(epochs), '--device', str(device), '--run_name', name])
        for key, value in asdict(self.model.frontdoor_config).items():
            if hasattr(args, key):
                setattr(args, key, value)
        return run_fold(args, held, model=self.model, optimizer=self.optimizer)


class HCIModelTrainer(DEAPModelTrainer):
    dataset = 'HCI'

    def __init__(self, model, root_dir, optimizer, scheduler=None, test_index=1,
                 batch_size=20, k_fold=24, n_epoch=20, temperature=13.544,
                 Trainer_name='Trainer', label='arousal', device_id=0):
        super().__init__(model, root_dir, optimizer, scheduler, test_index,
                         batch_size, k_fold, n_epoch, temperature, Trainer_name, label, device_id)
