"""生理窗口读取与来源追踪；分类下采样与训练总体参考分布明确分离。"""
import os
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler


DATASET_SPECS = {
    'DEAP': {'subjects': 32, 'trials': 40, 'windows': 10, 'peripheral': (34, 35, 36), 'points': 8064},
    'HCI': {'subjects': 24, 'trials': 20, 'windows': 5, 'peripheral': (32, 33, 34, 35), 'points': 3840},
}


def canonical_dataset(name: str) -> str:
    name = name.upper().replace('-', '_')
    if name in ('HCI', 'MAHNOB_HCI'):
        return 'HCI'
    if name != 'DEAP':
        raise ValueError('数据集只能是DEAP或HCI/MAHNOB_HCI')
    return name


def canonical_label(name: str) -> str:
    name = name.lower()
    if name not in ('arousal', 'valence'):
        raise ValueError('任务只能是arousal或valence')
    return name


def _numpy(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def normalize_subject(data: np.ndarray) -> np.ndarray:
    """保留原 zscore 默认axis=0：各被试内部沿trial轴计算，不混合其他被试。"""
    data = np.asarray(data, dtype=np.float32)
    if not np.isfinite(data).all():
        raise ValueError('生理输入含NaN或无穷值，请检查服务器预处理文件')
    mean, std = data.mean(axis=0, keepdims=True), data.std(axis=0, keepdims=True)
    # 常量位置原实现产生NaN；明确归零，仅修复未定义的零方差情况。
    return (data - mean) / np.where(std == 0, 1., std)


def DEAP_eliminate_individual_differences(data):
    return torch.from_numpy(normalize_subject(_numpy(data)))


def HCI_eliminate_individual_differences(data):
    return torch.from_numpy(normalize_subject(_numpy(data)))


class PhysiologicalWindows(Dataset):
    """未做标签选择的配对训练总体；每条记录有被试、trial、窗口和数据集标识。"""
    def __init__(self, root_dir, dataset, subjects, label='arousal'):
        self.dataset = canonical_dataset(dataset)
        self.label = canonical_label(label)
        spec = DATASET_SPECS[self.dataset]
        self.subjects = sorted(int(s) for s in subjects)
        if not self.subjects or len(set(self.subjects)) != len(self.subjects) or any(s < 1 or s > spec['subjects'] for s in self.subjects):
            raise ValueError('被试列表为空、重复或超出数据集范围')
        self.windows_per_trial = spec['windows']
        self.trials_per_subject = spec['trials']
        root_dir = root_dir if root_dir is not None else os.environ.get('NEURO_ROOT', '/home/fss')
        self.root_dir = Path(root_dir) / 'Dataset' / f'{self.dataset}-python'
        self.data_files = [f'{s}.npy' for s in self.subjects]
        cns, ans, labels, sources = [], [], [], []
        for subject in self.subjects:
            path = self.root_dir / f'{subject}.npy'
            with path.open('rb') as stream:
                # 仓库既有文件虽名为npy，实际为pickle字典；不改变格式。
                record = pickle.load(stream)
            raw, ratings = _numpy(record['data']), _numpy(record['labels'])
            if raw.ndim != 3 or raw.shape[0] != spec['trials'] or raw.shape[1] <= max(spec['peripheral']) or raw.shape[2] != spec['points']:
                raise ValueError(f'{path}形状错误：实际{raw.shape}，要求{spec["trials"]}个trial及{spec["points"]}个时间点')
            if ratings.ndim != 2 or ratings.shape[0] != spec['trials'] or ratings.shape[1] < 2 or not np.isfinite(ratings[:, :2]).all():
                raise ValueError(f'{path}标签形状或数值错误')
            offset = 384 if self.dataset == 'DEAP' else 0
            eeg = normalize_subject(raw[:, :32, offset:])
            peripheral = normalize_subject(raw[:, spec['peripheral'], offset:])
            def windowize(array):
                count, channels, length = array.shape
                if length != spec['windows'] * 768:
                    raise ValueError('窗口长度不满足6秒×128Hz')
                windows = array.reshape(count, channels, spec['windows'], 768).transpose(0, 2, 1, 3)
                return torch.from_numpy(np.ascontiguousarray(windows.reshape(-1, channels, 768)))
            cns.append(windowize(eeg))
            ans.append(windowize(peripheral))
            column = 1 if self.label == 'arousal' else 0
            binary = (ratings[:, column] > 5).astype(np.int64)
            labels.append(torch.from_numpy(np.repeat(binary, spec['windows'])))
            sources.extend((subject, trial, window) for trial in range(spec['trials']) for window in range(spec['windows']))
        self.x_c, self.x_a = torch.cat(cns), torch.cat(ans)
        self.Label = torch.cat(labels)
        self.source_ids = torch.tensor(sources, dtype=torch.long)

    def __len__(self):
        return len(self.Label)

    def __getitem__(self, index):
        subject, trial, window = self.source_ids[index].tolist()
        peripheral_key = 'EMG' if self.dataset == 'DEAP' else 'ECG'
        payload = {
            'x_c': self.x_c[index], 'x_a': self.x_a[index], 'EEG': self.x_c[index],
            peripheral_key: self.x_a[index, :-1], 'GSR': self.x_a[index, -1:],
            'Labels': self.Label[index], 'source_ids': self.source_ids[index],
            'subject_id': subject, 'trial_id': trial, 'window_id': window, 'dataset': self.dataset,
        }
        return payload, int(index)

    def balanced_indices(self, seed=42):
        """沿用trial级二类下采样及整组截断；不会修改未平衡总体或来源。"""
        w, group = self.windows_per_trial, self.trials_per_subject
        trial_labels = self.Label[::w].numpy()
        zero, one = np.flatnonzero(trial_labels == 0), np.flatnonzero(trial_labels == 1)
        size = min(len(zero), len(one))
        size = size // group * group
        if size == 0:
            raise ValueError('原trial平衡规则截断后无样本；请检查标签或明确使用--no_balance')
        rng = np.random.RandomState(seed)
        # 先抽取min_len再截断，与既有选择顺序一致；使用当前任务的标签。
        minimum = min(len(zero), len(one))
        selected = np.concatenate((rng.choice(zero, minimum, replace=False)[:size],
                                   rng.choice(one, minimum, replace=False)[:size]))
        rng.shuffle(selected)
        return torch.tensor((selected[:, None] * w + np.arange(w)[None, :]).reshape(-1), dtype=torch.long)


class WindowSelection(Dataset):
    """只保存总体窗口索引，共享原始张量，不复制或丢失来源。"""
    def __init__(self, population, indices):
        self.population = population
        self.indices = torch.as_tensor(indices, dtype=torch.long)
        if self.indices.ndim != 1 or not len(self.indices) or self.indices.min() < 0 or self.indices.max() >= len(population):
            raise ValueError('窗口子集索引错误')
        self.dataset, self.label = population.dataset, population.label
        self.windows_per_trial = population.windows_per_trial

    @property
    def source_ids(self):
        return self.population.source_ids[self.indices]

    @property
    def Label(self):
        return self.population.Label[self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.population[int(self.indices[index])]


class GroupedSampler(Sampler):
    """保留原有按trial窗口分组打乱；使用独立固定种子生成器逐epoch推进。"""
    def __init__(self, data_source, group_size, shuffle=True, seed=42):
        self.data_source, self.group_size, self.shuffle = data_source, group_size, shuffle
        self.generator = torch.Generator().manual_seed(seed)

    def __iter__(self):
        groups = list(range(0, len(self.data_source), self.group_size))
        if self.shuffle:
            groups = [groups[i] for i in torch.randperm(len(groups), generator=self.generator).tolist()]
        return iter([j for start in groups for j in range(start, min(start + self.group_size, len(self.data_source)))])

    def __len__(self):
        return len(self.data_source)


def seed_worker(worker_id):
    """DataLoader提供的确定性worker种子同时应用于NumPy和Python。"""
    import random
    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed)
    random.seed(seed)


def make_loaders(train_dataset, test_dataset, batch_size=40, workers=0, seed=42,
                 sampling_mode='grouped'):
    """默认沿用trial窗口整组顺序；window只改变优化窗口排列，不改变样本或参考总体。"""
    if sampling_mode not in ('grouped', 'window'):
        raise ValueError('sampling_mode只能是grouped或window')
    if batch_size < 2 or len(train_dataset) < 2:
        raise ValueError('训练batch及训练集至少包含两个窗口，以便排除自身参考')
    common = {'num_workers': workers, 'worker_init_fn': seed_worker,
              'generator': torch.Generator().manual_seed(seed)}
    group_size = train_dataset.windows_per_trial if sampling_mode == 'grouped' else 1
    sampler = GroupedSampler(train_dataset, group_size, seed=seed)
    train = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, drop_last=True, **common)
    if len(train) == 0:
        raise ValueError('训练数据少于batch_size；请显式减小batch_size')
    test = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, **common)
    return train, test


def cal_train_index(test_index, total_subjects=32):
    tests = to_list(test_index)
    if not tests or min(tests) < 1 or max(tests) > total_subjects:
        raise ValueError('held-out被试超出范围')
    return [s for s in range(1, total_subjects + 1) if s not in tests]


def to_list(value):
    return value if isinstance(value, list) else [int(value)]


def _compat_dataset(root_dir, data_files, label, dataset):
    directory = Path(root_dir if root_dir is not None else os.environ.get('NEURO_ROOT', '/home/fss')) / 'Dataset' / f'{dataset}-python'
    subjects = [int(Path(str(f)).stem) for f in data_files] if data_files is not None else [int(f.stem) for f in directory.glob('*.npy')]
    population = PhysiologicalWindows(root_dir, dataset, subjects, label)
    if len(subjects) > 1:
        return WindowSelection(population, population.balanced_indices())
    return population


class DEAPDataset_Python(Dataset):
    """保留既有构造名和getitem格式，valence参数正确传入trial平衡。"""
    def __init__(self, root_dir=None, data_files=None, label='arousal'):
        self.wrapped = _compat_dataset(root_dir, data_files, label, 'DEAP')

    def __len__(self):
        return len(self.wrapped)

    def __getitem__(self, index):
        return self.wrapped[index]

    @property
    def source_ids(self):
        return self.wrapped.source_ids


class HCIDataset_Python(DEAPDataset_Python):
    def __init__(self, root_dir=None, data_files=None, label='arousal'):
        self.wrapped = _compat_dataset(root_dir, data_files, label, 'HCI')


def DEAPSubdataset(root_dir, index, label):
    return _compat_dataset(root_dir, [f'{i}.npy' for i in to_list(index)], label, 'DEAP')


def HCISubdataset(root_dir, index, label):
    return _compat_dataset(root_dir, [f'{i}.npy' for i in to_list(index)], label, 'HCI')


def _compat_loaders(root_dir, test_index, label, batch_size, dataset):
    tests = to_list(test_index)
    if len(tests) != 1:
        raise ValueError('LOSO只允许一个held-out被试')
    count = DATASET_SPECS[dataset]['subjects']
    population = PhysiologicalWindows(root_dir, dataset, cal_train_index(tests, count), label)
    train = WindowSelection(population, population.balanced_indices())
    test = PhysiologicalWindows(root_dir, dataset, tests, label)
    return make_loaders(train, test, batch_size)


def DEAPDataLoader(root_dir, test_index, label, batch_size=40):
    return _compat_loaders(root_dir, test_index, label, batch_size, 'DEAP')


def HCIDataLoader(root_dir, test_index, label, batch_size=20):
    return _compat_loaders(root_dir, test_index, label, batch_size, 'HCI')
