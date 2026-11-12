"""方案A的有限预算训练子集验收；会更新私有模型，不读写旧checkpoint。"""
import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


def fit_gate(records, prior_nll):
    """预先固定的工程验收，不是因果识别标准或LOSO成绩。"""
    return len(records) == 3 and all(r['balanced_accuracy'] >= .8 and r['min_class_recall'] >= .6
                                and r['integrated_observational_nll'] <= prior_nll - .05 for r in records)


def save_private_checkpoint(model, result, destination):
    """仅显式诊断时保存末端私有权重；独立格式、不含信号、不覆盖任何已有文件。"""
    import os
    import tempfile
    import torch
    target = destination / 'private_fit_last.pt'
    payload = dict(format_version='neurofdan_private_fit_v1', role='training_subset_diagnostic_only',
        model=model.state_dict(), frontdoor_config=result['model'], arguments=result['arguments'],
        source_version=result['source_version'], actual_steps=result['actual_steps'],
        selected_source_ids=result['selected_source_ids'], reference_source_ids=result['reference_source_ids'],
        reference_report=result['reference_report'], passed=result['passed'])
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination, prefix='.private-fit-', delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(payload, stream)
        # 同文件系统原子创建新名字，目标已有时失败，不使用会覆盖目标的replace。
        os.link(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    result.update(weights_saved=True, private_checkpoint_file=target.name,
                  private_checkpoint_format=payload['format_version'])


def measure(model, batch, bank, device, dataset, seed):
    import torch
    from ModelTrain import cache_reference_features, confusion_metrics, prediction_probabilities
    from classification_objectives import integrated_observational_nll
    from training_diagnostics import isolated_evaluation
    before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    records = []
    with isolated_evaluation(model):
        torch.manual_seed(seed)
        cached = cache_reference_features(model, bank, device, len(batch['Labels']))
        for draw in range(3):
            out = model(batch['x_c'], batch['x_a'], reference_bank=cached,
                        source_ids=batch['source_ids'], fd_mode='full')
            p = prediction_probabilities(out, 'full')
            y = batch['Labels']
            cm = torch.bincount((2 * y + p.argmax(-1)).cpu(), minlength=4).reshape(2, 2)
            factual_cm = torch.bincount((2 * y + out['probs_obs'].argmax(-1)).cpu(), minlength=4).reshape(2, 2)
            m = confusion_metrics(cm, dataset)
            recalls = [float(cm[k, k]) / int(cm[k].sum()) for k in (0, 1)]
            posterior = {}
            for branch, mu_key, lv_key in (('cns', 'mu_c', 'logvar_c'), ('ans', 'mu_a', 'logvar_a')):
                mu, sigma = out[mu_key].double(), (.5 * out[lv_key].double()).exp()
                if mu.ndim == 3:
                    mu, sigma = mu.mean(1), sigma.mean(1)
                posterior[branch] = dict(mean_between_input_std=float(mu.std(0, unbiased=False).mean()),
                    sigma_between_input_std=float(sigma.std(0, unbiased=False).mean()),
                    mean_posterior_sigma=float(sigma.mean()),
                    class_mean_distance=float((mu[y == 0].mean(0) - mu[y == 1].mean(0)).norm()))
            records.append(dict(draw=draw + 1, **m, confusion_matrix=cm.tolist(),
                min_class_recall=min(recalls), class_recalls=recalls,
                integrated_observational_nll=float(integrated_observational_nll(out, y)),
                probability_mean=p.double().mean(0).tolist(),
                probability_std=p.double().std(0, unbiased=False).tolist(),
                mediator_statistics={k: float(v) for k, v in out['mediator_statistics'].items()},
                posterior=posterior, posterior_note='ANS先沿有限S平均，再计算窗口差异，仍含MC误差；不解释为因果效应',
                factual_confusion=factual_cm.tolist(), factual_metrics=confusion_metrics(factual_cm, dataset)))
    if not all(torch.equal(before[k], v.detach().cpu()) for k, v in model.state_dict().items()):
        raise RuntimeError('验收评估修改了参数或BN状态')
    return records


def main():
    controls = argparse.ArgumentParser(add_help=False)
    controls.add_argument('--fit_max_steps', type=int, default=300)
    controls.add_argument('--fit_check_every', type=int, default=50)
    controls.add_argument('--fit_baseline_summary', default=None,
                          help='可选KL单因素对照原摘要；启动前、来源和首步NLL严格核对')
    controls.add_argument('--fit_save_checkpoint', action='store_true',
                          help='只保存新目录中的私有末端权重，供后续只读探针，不用于正式续训')
    fit, training_argv = controls.parse_known_args()
    from mainNeuroFDAN import get_args
    args = get_args(argv=training_argv)
    if fit.fit_max_steps < 1 or fit.fit_check_every < 1:
        raise ValueError('拟合步数和检查间隔至少为1')
    if (args.classification_objective != 'integrated_observational' or args.resume or args.no_cns2ans or
            not args.held_out_subjects or len(args.held_out_subjects) != 1 or
            args.max_train_batches or args.max_test_batches):
        raise ValueError('验收仅支持方案A、单fold、全新私有模型；不接受旧权重或smoke上限')
    dest = Path(args.output_dir) / args.dataset / args.label / 'seed42' / args.run_name
    path = dest / 'probe_summary.json'
    if path.exists():
        raise FileExistsError('验收结果已存在，拒绝覆盖')
    if fit.fit_save_checkpoint and ((dest / 'private_fit_last.pt').exists() or (dest / 'private_fit_last.pt').is_symlink()):
        raise FileExistsError('私有权重已存在，拒绝覆盖')
    baseline = None
    if fit.fit_baseline_summary:
        from scheme_a_control_evidence import read_json, check_arguments, summary_sha256
        baseline = read_json(fit.fit_baseline_summary)
        baseline_hash = summary_sha256(fit.fit_baseline_summary)
        check_arguments(baseline, vars(args), vars(fit))
    dest.mkdir(parents=True, exist_ok=True)
    from run_reporting import source_version
    result = dict(operation='方案A固定自然训练批次拟合验收', source_version=source_version(PROJECT),
        dataset=args.dataset, label=args.label, held_out_subject=args.held_out_subjects[0],
        arguments=vars(args), fit_controls=vars(fit), completed=False, passed=False,
        training_executed=False, actual_steps=0, full_fold_training=False,
        weights_saved=False, weights_transferred_to_fold=False, records=[],
        gate=dict(balanced_accuracy_min=.8, min_class_recall=.6, nll_below_prior_by=.05,
                  monte_carlo_draws=3, all_draws_must_pass=True),
        note='仅训练来源子集的工程拟合；不是测试性能、完整LOSO或因果识别证明；主训练另从seed42初始化。')

    def save():
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        temporary.replace(path)

    save()
    try:
        import torch
        from Datasets import DATASET_SPECS, PhysiologicalWindows, cal_train_index
        from ModelTrain import (seed_everything, resolve_device, build_reference_bank, frontdoor_config,
                                build_optimizer, loss_component_gradient_report)
        from NeuroFDAN import NeuroFDAN
        from classification_objectives import classification_terms
        seed_everything(args.seed)
        if args.cpu_threads:
            torch.set_num_threads(args.cpu_threads)
        subject = args.held_out_subjects[0]
        subjects = cal_train_index(subject, DATASET_SPECS[args.dataset]['subjects'])
        population = PhysiologicalWindows(args.root_dir, args.dataset, subjects, args.label)
        if len(population) < args.batch_size:
            raise ValueError('自然训练总体不足一个完整batch')
        bank = build_reference_bank(population, subjects, subject, args.N_ref, args.seed)
        selected = torch.randperm(len(population), generator=torch.Generator().manual_seed(args.seed))[:args.batch_size]
        device = resolve_device(args.device)
        batch = dict(x_c=population.x_c[selected].to(device), x_a=population.x_a[selected].to(device),
                     Labels=population.Label[selected].to(device), source_ids=population.source_ids[selected].to(device))
        counts = torch.bincount(batch['Labels'].cpu(), minlength=2).tolist()
        if min(counts) < 1:
            raise ValueError('固定自然批次只有一类；停止，不做标签平衡或重新挑样本')
        pi = counts[1] / sum(counts)
        prior_nll = -pi * math.log(pi) - (1 - pi) * math.log(1 - pi)
        model = NeuroFDAN(ecg_channels=2 if args.dataset == 'DEAP' else 3,
                          frontdoor_config=frontdoor_config(args)).to(device)
        optimizer = build_optimizer(model, args)
        epoch_steps = len(population) // args.batch_size
        result.update(model=asdict(model.frontdoor_config), reference_report=bank.report,
            selected_source_ids=batch['source_ids'].cpu().tolist(), reference_source_ids=bank.source_ids.tolist(),
            true_counts=counts, constant_prior_nll=prior_nll, full_epoch_steps=epoch_steps,
            loss_epoch_mapping='1+(私有step-1)//完整训练总体每轮步数；不是LOSO实际epoch')
        if baseline is not None:
            from scheme_a_control_evidence import check_sources
            check_sources(baseline, result)
            result['control_protocol'] = dict(changed_training_factor='beta:0.001->0',
                source_ids_and_reference_ids_equal=True, baseline_summary_sha256=baseline_hash)
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        save()
        for step in range(1, fit.fit_max_steps + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            out = model(batch['x_c'], batch['x_a'], reference_bank=bank,
                        source_ids=batch['source_ids'], fd_mode='full', factual_ce_mode=args.factual_ce_mode)
            epoch = 1 + (step - 1) // epoch_steps
            terms = classification_terms(out, batch['Labels'], model.frontdoor_config, epoch,
                                         objective=args.classification_objective, lambda_obs=args.lambda_obs)
            loss = terms['loss'] + args.fire_weight * out['fire_loss']
            if step == 1 and baseline is not None:
                from scheme_a_control_evidence import check_initial_nll
                check_initial_nll(baseline, float(terms['classification_loss'].detach()))
                result['control_protocol']['initial_task_nll_matches'] = True
            if not torch.isfinite(loss):
                raise FloatingPointError('私有拟合损失含NaN/Inf')
            report = None
            if step == 1 or step % fit.fit_check_every == 0:
                report = loss_component_gradient_report(model, out, terms, loss, epoch,
                    args.lambda_obs, args.fire_weight, args.classification_objective)
                if step == 1:
                    task = report['gradients']['integrated_observational_nll']
                    result['first_batch_task_gradients'] = task
                    if not all(task[key]['l2'] > 0 and math.isfinite(task[key]['l2']) for key in ('cns', 'ans', 'cns2ans', 'outcome')):
                        raise RuntimeError('主NLL没有到达全部关键模块，停止晋级')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(norm):
                raise FloatingPointError('反向梯度含NaN/Inf')
            optimizer.step(); model.weights_updated()
            result.update(training_executed=True, actual_steps=step)
            if step == 1:
                result['first_full_backward_complete'] = True
                if device.type == 'cuda':
                    result['first_backward_peak_mib'] = torch.cuda.max_memory_allocated(device) / 1024**2
            task_loss = float(terms['classification_loss'].detach())
            del loss, out, terms
            if step == 1 or step % fit.fit_check_every == 0 or step == fit.fit_max_steps:
                draws = measure(model, batch, bank, device, args.dataset, args.seed)
                passed = step >= fit.fit_check_every and fit_gate(draws, prior_nll)
                result['records'].append(dict(step=step, loss_epoch=epoch, optimization_nll=task_loss,
                    draws=draws, gradient_report=report, passed=passed))
                result['passed'] = passed
                if device.type == 'cuda':
                    result['peak_allocated_mib'] = torch.cuda.max_memory_allocated(device) / 1024**2
                save()
                print(f'私有拟合step={step}/{fit.fit_max_steps} 主NLL={task_loss:.6f} '
                      f'BA={[round(d["balanced_accuracy"],4) for d in draws]} '
                      f'三次MC均通过={passed}', flush=True)
                if passed:
                    break
            else:
                save()
        if baseline is not None:
            if summary_sha256(fit.fit_baseline_summary) != baseline_hash:
                raise RuntimeError('原基线摘要读取期间发生变化，停止解释配对比较')
            result['control_protocol']['original_baseline_unchanged'] = True
        if fit.fit_save_checkpoint:
            save_private_checkpoint(model, result, dest)
        result['completed'] = True
        save()
        print('拟合门槛通过；本脚本不启动正式fold训练，由调用流程决定后续。' if result['passed'] else
              '预算内未通过拟合验收，停止，不启动20轮；不据此宣称结构理论上不可能训练。', flush=True)
        return 0 if result['passed'] else 3
    except BaseException as error:
        result.update(error_type=type(error).__name__, error=str(error), passed=False)
        save()
        raise


if __name__ == '__main__':
    sys.exit(main())
