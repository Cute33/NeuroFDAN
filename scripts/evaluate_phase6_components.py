"""只读拆分已保存的合成教师模型；不训练、不接入真实数据流程。"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

import numpy as np
import torch
from scipy.special import expit

from frontdoor_scm import FrontDoorConfig, FrontDoorEstimator, ReferenceBank
from phase6_synthetic_experiment import data_digest, save_json, training_view
from run_reporting import source_version
from synthetic_scm import SCMConfig, SyntheticSCM, probability_metrics

TRAINING_COMMIT = '15b79769535027f4dfb7de0cb5ab1c11252dec49'
CELLS = ('true_m_true_h', 'true_m_learned_h', 'learned_m_true_h', 'learned_m_learned_h')


def file_digest(path: Path) -> str:
    """逐块读取源文件指纹，不写入源目录。"""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def state_digest(model: FrontDoorEstimator) -> str:
    """包含参数与持久缓冲区；评估前后必须一致。"""
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def true_conditional_grid(scm: SyntheticSCM, ma: np.ndarray, rc: np.ndarray,
                          ra: np.ndarray, gamma: float) -> np.ndarray:
    """[B,S,D]×[B,K,d_x]→[B,S,K]；积分U|X'，不调用直接干预函数。"""
    if ma.ndim != 3 or rc.ndim != 3 or ra.ndim != 3 or rc.shape[:2] != ra.shape[:2] or ma.shape[0] != rc.shape[0]:
        raise ValueError('中介与配对参考必须有明确的B/S/K轴')
    b, s, d = ma.shape
    k = rc.shape[1]
    if d != scm.config.d_ma or rc.shape[2] != scm.config.d_xc or ra.shape[2] != scm.config.d_xa:
        raise ValueError('中介或参考特征维度与生成器不一致')
    if not all(np.isfinite(value).all() for value in (ma, rc, ra)) or np.any(abs(ma) > 1):
        raise ValueError('不得把无界高斯中介代入tanh机制的真实条件分布')
    posterior, covariance = scm.posterior_u(rc.reshape(b*k, -1), ra.reshape(b*k, -1), gamma)
    offsets = (gamma * (posterior @ scm.w_u) + scm.b).reshape(b, k)
    spread = np.sqrt(2 * gamma**2 * (scm.w_u @ covariance @ scm.w_u))
    output = np.empty((b, s, k))
    for start in range(0, s, 32):
        location = (ma[:, start:start+32] @ scm.w_a)[:, :, None] + offsets[:, None, :]
        output[:, start:start+32] = expit(location[..., None] + spread * scm.nodes) @ scm.weights
    return output


def paired_mediators(model: FrontDoorEstimator, scm: SyntheticSCM, xc: torch.Tensor,
                     xa: torch.Tensor, S: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """两条分层路径共用[B,S,D]独立标准噪声；条件机制各自保留。"""
    if model.config.mediator_family != 'tanh_normal' or model.config.no_cns2ans:
        raise ValueError('只支持坐标锚定且保留CNS→ANS的tanh教师模型')
    b, d = len(xc), model.config.mediator_dim
    if d != scm.config.d_mc or d != scm.config.d_ma:
        raise ValueError('教师与真实中介坐标维度不匹配')
    rng = np.random.default_rng(seed)
    ec = rng.normal(size=(b, S, d))
    ea = rng.normal(size=(b, S, d))
    true_mc = np.tanh((xc.double().numpy() @ scm.B_c.T)[:, None, :] + ec)
    true_ma = np.tanh((xa.double().numpy() @ scm.B_a.T)[:, None, :] + true_mc @ scm.C_ca.T + ea)
    mu, lv = model.cns.distribution(xc)
    mc = (mu[:, None] + (.5*lv[:, None]).exp() * torch.tensor(ec, dtype=mu.dtype)).tanh()
    ans = model.ans(xa, mc)
    # ANSMediator的自带抽样不用于此配对评价；用同一ea重采，参数仍来自原模块。
    learned_ma = (ans['mu_a'] + (.5*ans['logvar_a']).exp() * torch.tensor(ea, dtype=mu.dtype)).tanh()
    return true_ma, learned_ma.numpy()


def evaluate_gamma(model: FrontDoorEstimator, scm: SyntheticSCM, train: dict,
                   points: dict, observed: dict, gamma: float, target: dict, args) -> dict:
    """四格共享参考索引和噪声；不把误差差异解释为可加因果贡献。"""
    n = len(train['y'])
    ids = torch.stack((torch.zeros(n, dtype=torch.long), torch.arange(n), torch.zeros(n, dtype=torch.long)), 1)
    bank = ReferenceBank(train['x_c'], train['x_a'], ids, max_size=model.config.N_ref,
                         seed=args.seed, forbidden_subjects=(1, 2), dataset='synthetic')
    xc = torch.tensor(points['x_c'], dtype=torch.float32)
    xa = torch.tensor(points['x_a'], dtype=torch.float32)
    truth = np.asarray(target['direct_probability'])
    oracle = np.asarray(target['oracle_probability'])
    if truth.shape != (len(xc),) or oracle.shape != truth.shape or not np.isfinite(truth).all():
        raise ValueError('保存真值与重新生成的干预点数量不匹配')
    values = {cell: [] for cell in CELLS}
    repeats = []
    with torch.no_grad(), torch.random.fork_rng(devices=[]):
        own = model.reference_encoder(torch.tensor(observed['x_c'], dtype=torch.float32),
                                      torch.tensor(observed['x_a'], dtype=torch.float32))
        true_ma = torch.tensor(observed['m_a'], dtype=torch.float32)
        factual_h = model.outcome(true_ma, own).softmax(-1)[:, 1].numpy()
        factual_truth = scm.conditional_outcome(observed['m_a'], observed['x_c'], observed['x_a'], gamma)
        factual_error = probability_metrics(factual_h, factual_truth)
        for repeat in range(args.repeats):
            started = time.perf_counter()
            torch.manual_seed(args.seed + 90000 + repeat)
            ma_t, ma_l = paired_mediators(model, scm, xc, xa, args.S, args.seed + 91000 + repeat)
            rc, ra, sources = bank.sample(len(xc), args.K)
            b, k = rc.shape[:2]
            references = model.reference_encoder(rc.reshape(b*k, -1), ra.reshape(b*k, -1)).reshape(b, k, -1)
            predictions, conditional_errors, seconds = {}, {}, {}
            for label, ma in (('true_m', ma_t), ('learned_m', ma_l)):
                step = time.perf_counter()
                known = true_conditional_grid(scm, ma, rc.numpy(), ra.numpy(), gamma)
                predictions[label+'_true_h'] = known.mean(2).mean(1)
                seconds[label+'_true_h'] = time.perf_counter() - step
                step = time.perf_counter()
                # 学习结果模型始终调用唯一的softmax→K均值→S均值实现。
                p, logits = model.marginalize(torch.tensor(ma, dtype=torch.float32), references)
                predictions[label+'_learned_h'] = p[:, 1].numpy()
                seconds[label+'_learned_h'] = time.perf_counter() - step
                conditional_errors[label] = probability_metrics(logits.softmax(-1)[..., 1].numpy(), known)
            for cell in CELLS:
                p = predictions[cell]
                if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
                    raise FloatingPointError('四格评价出现非法概率')
                values[cell].append(p)
            repeats.append(dict(repeat=repeat, seconds=time.perf_counter()-started, cell_seconds=seconds,
                conditional_head_error_on_cross_pairs=conditional_errors,
                errors={cell: probability_metrics(predictions[cell], truth) for cell in CELLS},
                reference_indices_sha256=hashlib.sha256(sources.numpy().tobytes()).hexdigest()))
            print(f'gamma={gamma} 积分重复={repeat+1}/{args.repeats} 四格MSE=' +
                  json.dumps({cell: repeats[-1]['errors'][cell]['probability_mse'] for cell in CELLS}), flush=True)
    arrays = {cell: np.stack(v) for cell, v in values.items()}
    cells = {}
    for cell, value in arrays.items():
        cells[cell] = dict(probability_per_repeat=value.tolist(),
            mean_probability=value.mean(0).tolist(), against_direct=probability_metrics(value.mean(0), truth),
            against_population_oracle=probability_metrics(value.mean(0), oracle),
            against_empirical_true_true=probability_metrics(value.mean(0), arrays[CELLS[0]].mean(0)),
            mc_std_per_point=value.std(0, ddof=1).tolist(),
            mc_standard_error_of_mean_per_point=(value.std(0, ddof=1)/np.sqrt(args.repeats)).tolist())
    return dict(gamma=gamma, cells=cells, repeats=repeats, factual_true_ma_head_error=factual_error,
        reference=dict(available=n, used=bank.used_count, shortfall=bank.shortfall,
                       distribution='原合成自然训练总体，逐点有放回配对抽样'),
        saved_direct_probability=truth.tolist(), saved_direct_standard_error=target['direct_standard_error'],
        saved_population_oracle_probability=oracle.tolist(),
        note='MC标准误仅条件于固定模型/经验库，未覆盖训练或参考总体抽样不确定性；四格误差不是可加因果分解。')


def main() -> None:
    p = argparse.ArgumentParser(description='只读拆分已有合成教师checkpoint，不执行训练', add_help=False)
    p.add_argument('-h', '--help', action='help', help='显示中文帮助')
    p.add_argument('--run_dir', type=Path, required=True, help='原paired_tanh_normal_20ep目录，含本地models和生成参数')
    p.add_argument('--output_dir', type=Path, required=True, help='新的独立phase6评价目录，不得覆盖')
    p.add_argument('--S', type=int, default=256, help='仅本次评价中介抽样数，不改checkpoint配置')
    p.add_argument('--K', type=int, default=128, help='仅本次评价每点联合参考抽样数')
    p.add_argument('--repeats', type=int, default=4, help='固定checkpoint的MC重复，至少两次')
    p.add_argument('--seed', type=int, choices=[42], default=42, help='固定原数据种子42')
    p.add_argument('--device', choices=['cpu'], default='cpu', help='只读CPU合成评价')
    p.add_argument('--workers', type=int, choices=[0], default=0, help='顺序评价不创建数据worker')
    p.add_argument('--diagnostic_level', choices=['full'], default='full', help='保存四格和配对条件误差')
    args = p.parse_args()
    if min(args.S, args.K, args.repeats) < 2:
        p.error('S、K与MC重复至少为2')
    run, out = args.run_dir.resolve(), args.output_dir.resolve()
    if out == run or run in out.parents or out in run.parents:
        p.error('评价目录必须独立于原实验目录')
    out.mkdir(parents=True, exist_ok=True)
    # 允许日志包装器先创建目录；任何旧评价产物均禁止覆盖。
    if any(path.name not in ('train.log', 'train.log.exit') for path in out.iterdir()):
        raise FileExistsError('评价目录已有产物，请使用新的运行名')
    result = dict(operation='Phase6只读教师四格评价', completed=False, training_executed=False,
                  real_data_training=False, source_version=source_version(PROJECT), gammas=[])
    save_json(out/'probe_summary.json', result)
    files = [run/'config.json', run/'probe_summary.json', run/'scm_parameters.npz']
    originals = {str(path): file_digest(path) for path in files}
    cfg = json.loads(files[0].read_text(encoding='utf-8'))
    saved = json.loads(files[1].read_text(encoding='utf-8'))
    a = cfg['arguments']
    if cfg['source_version']['commit'] != TRAINING_COMMIT or saved['source_version']['commit'] != TRAINING_COMMIT or not saved['completed'] or saved['smoke']:
        raise ValueError('原实验必须是已完成的15b7976非smoke配对实验')
    if [row['gamma'] for row in saved['gammas']] != a['gammas'] or not saved['gammas']:
        raise ValueError('原实验的混杂强度记录不完整或顺序不一致')
    if a['mediator_family'] != 'tanh_normal' or a['no_cns2ans'] or a['seed'] != args.seed:
        raise ValueError('只允许原tanh_normal完整路径、seed42教师对照；不混用任意潜在坐标')
    scm = SyntheticSCM(SCMConfig(**cfg['scm']))
    with np.load(files[2], allow_pickle=False) as parameters:
        for key in ('A_c', 'A_a', 'B_c', 'B_a', 'C_ca', 'w_a', 'w_u', 'b'):
            if not np.array_equal(parameters[key], np.asarray(getattr(scm, key))):
                raise ValueError('保存生成参数与再生机制不一致：'+key)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    points = scm.sample(a['points'], 0., 78901)
    save_json(out/'config.json', dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        source_version=result['source_version'], original_source=cfg['source_version'],
        original_arguments=a, scm=cfg['scm'],
        target='固定原干预点、原真值；只替换合成教师的条件模块，不进行选模或调参'))
    for target in saved['gammas']:
        gamma = target['gamma']
        record = next(row for row in target['conditions'] if row['condition'] == 'oracle_mediator')
        checkpoint = run/'models'/f'gamma{gamma}_oracle_mediator_last.pt'
        originals[str(checkpoint)] = file_digest(checkpoint)
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if state['condition'] != 'oracle_mediator' or state['epoch'] != a['epochs'] or state['seed'] != args.seed or state['config'] != record['config']:
            raise ValueError('checkpoint条件、轮数、种子或配置与原报告不匹配')
        model_cfg = FrontDoorConfig(**state['config'])
        if model_cfg.mediator_family != 'tanh_normal' or model_cfg.no_cns2ans or model_cfg.ref_dim != 0:
            raise ValueError('本次拆分要求tanh中介和未额外压缩的完整参考')
        train = training_view(scm.sample(a['train_size'], gamma, args.seed+1000), 'observation_only')
        if data_digest(train) != record['training_data_sha256']:
            raise ValueError('重新生成的训练X/Y指纹与原实验不一致')
        observed = scm.sample(a['test_size'], gamma, args.seed+2000)
        model = FrontDoorEstimator(scm.config.d_xc, scm.config.d_xa, model_cfg).cpu().eval()
        model.load_state_dict(state['model'], strict=True)
        before = state_digest(model)
        row = evaluate_gamma(model, scm, train, points, observed, gamma, target, args)
        if state_digest(model) != before or model.training or any(p.grad is not None for p in model.parameters()):
            raise RuntimeError('只读状态保护失败')
        row.update(checkpoint_sha256=originals[str(checkpoint)], epoch=state['epoch'],
                   training_data_sha256=data_digest(train), state_unchanged=True, parameter_gradients_absent=True)
        result['gammas'].append(row)
        save_json(out/'probe_summary.json', result)
    if any(file_digest(Path(path)) != digest for path, digest in originals.items()):
        raise RuntimeError('源checkpoint或实验产物在评价期间发生变化')
    result.update(completed=True, source_files_unchanged=True, source_sha256=originals)
    save_json(out/'probe_summary.json', result)
    (out/'run_report.txt').write_text('只读合成教师四格评价完成。未训练；原模型和源文件指纹保持一致。\n'
        '逐gamma的干预概率误差、经验库oracle及MC标准误见probe_summary.json。\n', encoding='utf-8')
    print('只读评价完成；不需要运行mainNeuroFDAN.py。结果目录：'+str(out), flush=True)


if __name__ == '__main__':
    main()
