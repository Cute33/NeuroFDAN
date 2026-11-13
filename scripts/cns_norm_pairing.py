"""N1只允许CNS入口归一化改变；来源和初始化守卫，不定义新损失。"""
import hashlib


def check_arguments(base, args, controls):
    if (base.get('source_version', {}).get('commit') != 'bfe108e703b5d089fc53d59d254b61d6fcbbb885'
            or not base.get('completed') or base.get('passed') or base.get('actual_steps') != 300):
        raise ValueError('需要bfe108e已完成300步、未通过的beta0原始摘要')
    old, new = dict(base['arguments']), dict(args)
    old.setdefault('cns_input_norm','none'); new.setdefault('cns_input_norm','none')
    if (old['dataset'],old['label'],old['held_out_subjects'],old['seed'],old['beta']) != ('DEAP','arousal',[2],42,0.):
        raise ValueError('基线身份不符')
    changed={k for k in set(old)|set(new) if old.get(k)!=new.get(k)}
    if changed-{'output_dir','run_name'} != {'cns_input_norm'} or new['cns_input_norm']!='layernorm':
        raise ValueError('N1只允许norm由none改为layernorm，不允许改lr/beta/fire等')
    for key in ('fit_max_steps','fit_check_every'):
        if controls[key] != base['fit_controls'][key]: raise ValueError('预算必须相同')
    if controls['fit_gate_metric'] != 'acc': raise ValueError('N1使用ACC主验收指标')


def check_sources(base, result):
    for key in ('selected_source_ids','reference_source_ids','true_counts','full_epoch_steps'):
        if base[key]!=result[key]: raise ValueError('查询/参考来源或标签分布不同：'+key)
    old,new=dict(base['model']),dict(result['model'])
    old.setdefault('cns_input_norm','none')
    changed={k for k in set(old)|set(new) if old.get(k)!=new.get(k)}
    if changed!={'cns_input_norm'} or new['cns_input_norm']!='layernorm':
        raise ValueError('模型配置不只改变norm')


def parameter_hash(model):
    return hashlib.sha256(b''.join(p.detach().cpu().contiguous().numpy().tobytes() for p in model.parameters())).hexdigest()


def main():
    import argparse,json
    from pathlib import Path
    from scheme_a_control_evidence import read_json, sanitize
    p=argparse.ArgumentParser(description='N1配对结果，只读取已生成JSON')
    p.add_argument('--baseline',required=True);p.add_argument('--result',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();base,new=read_json(args.baseline),read_json(args.result)
    check_arguments(base,new['arguments'],new['fit_controls']);check_sources(base,new)
    if not new.get('initial_parameters',{}).get('matched'):raise ValueError('未核验初始化')
    def last(v):
        r=v['records'][-1]
        return dict(actual_steps=v['actual_steps'],completed=v['completed'],passed=v['passed'],
            draws=[{k:d[k] for k in ('acc','selection_f1','confusion_matrix','class_recalls',
                'integrated_observational_nll','probability_std','posterior')} for d in r['draws']])
    result=dict(operation='N1固定训练子集容量比较，非正式fold成绩',protocol_checks_passed=True,
        changed_training_factor='cns_input_norm:none->layernorm',true_counts=base['true_counts'],
        constant_prior_nll=base['constant_prior_nll'],baseline=last(base),normalization=last(new),
        initial_parameters=new['initial_parameters'],main_prediction='probs_do',primary_metric='ACC',
        note='当前none模板和norm初始化相同；历史基线无初始指纹，不称历史逐参数核验。拟合按ACC验收，旧BA门槛不参与新放行。',
        automatic_full_fold_training=False)
    encoded=json.dumps(sanitize(result),ensure_ascii=False,indent=2,allow_nan=False)
    with Path(args.output).open('x',encoding='utf-8') as f:f.write(encoded+'\n')
    print(encoded,flush=True)


if __name__=='__main__':main()
