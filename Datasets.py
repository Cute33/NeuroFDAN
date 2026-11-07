import os
import torch
import pickle
import scipy.io
from scipy.stats import zscore
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np

'''
The DEAP dataset and HCI dataset
'''
def DEAP_eliminate_individual_differences(data):
    # shape: (trails 40, channels, time, points)
    data = zscore(data)
    data = torch.tensor(data)
    return data

def HCI_eliminate_individual_differences(data):
    # shape: (trails 20, channels, time 30, points 128)
    data = data.numpy()
    data = zscore(data)
    data = torch.tensor(data)
    return data

class HCIDataset_Python(Dataset):
    def __init__(self, root_dir = None, data_files = None, label = 'arousal'):
        if root_dir is None:
            root_dir = os.environ.get('NEURO_ROOT', '/home/fss')
        # Get the data files list
        self.root_dir = os.path.join(root_dir, 'Dataset/HCI-python')
        if data_files is None:
            self.data_files = os.listdir(self.root_dir)
        else:
            self.data_files = data_files
        self.data_files.sort(key=lambda x:int(x[:-4]))
        self.time_window = 6
        # Loading data
        print('loading data...')
        self.EEG = []
        self.ECG = []
        self.GSR = []
        self.Label = []
        for data_file in self.data_files:
            if data_file.endswith('.npy'):
                file_path = os.path.join(self.root_dir, data_file)
            else:
                data_file = data_file + '.npy'
                file_path = os.path.join(self.root_dir, data_file)
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            tmp_EEG = data['data'][:, 0:32, :]
            self.EEG.append(tmp_EEG.reshape(20, 32, 30, 128))
            tmp_ECG = data['data'][:, 32:35, :]
            self.ECG.append(tmp_ECG.reshape(20, 3, 30, 128))
            tmp_GSR = (data['data'][:, 35, :])[:,None,:]
            self.GSR.append(tmp_GSR.reshape(20, 1, 30, 128))
            self.Label.append(torch.tensor(data['labels']))

        # 去除个体差异性
        for index in range(len(data_files)):
            self.EEG[index] = HCI_eliminate_individual_differences(self.EEG[index])
            self.ECG[index] = HCI_eliminate_individual_differences(self.ECG[index])
            self.GSR[index] = HCI_eliminate_individual_differences(self.GSR[index])
        
        # 为了平衡训练集的样本分布
        if len(data_files) != 1:         
            # 合并所有样本到一个列表，方便下采样
            all_EEG = []
            all_ECG = []
            all_GSR = []
            all_Label = []
            for i in range(len(self.EEG)):
                all_EEG.extend(list(self.EEG[i]))
                all_ECG.extend(list(self.ECG[i]))
                all_GSR.extend(list(self.GSR[i]))
                all_Label.extend(list(self.Label[i]))
            
            # 转为 numpy 数组，才能用索引
            all_EEG = np.array(all_EEG)      # [N, 32, 30, 128]
            all_ECG = np.array(all_ECG)      # [N, 3, 30, 128]
            all_GSR = np.array(all_GSR)      # [N, 1, 30, 128]
            all_Label = np.array(all_Label)  # [N, 2]

            # 下采样代码插入点
            # 统计每个类别的索引  Arousal: 1  Valence: 0
            # 检查标签类型并选择相应的列索引
            column_index = 1 if label == 'arousal' else 0  # 如果是arousal用第二列(索引1)，否则用第一列(索引0)
            labels_np = np.array(all_Label)
            class0_indices = np.where(labels_np[:,column_index] <= 5)[0]
            class1_indices = np.where(labels_np[:,column_index] > 5)[0]
            
            # 取两类中数量较少的那个，作为下采样目标
            min_len = min(len(class0_indices), len(class1_indices))
            np.random.seed(42)  # 保持可复现
            sampled_class0 = np.random.choice(class0_indices, min_len, replace=False)
            sampled_class1 = np.random.choice(class1_indices, min_len, replace=False)
            
            # 分别丢弃不能整除20的部分
            valid_trial_num = (min_len // 20) * 20
            sampled_class0 = sampled_class0[:valid_trial_num]
            sampled_class1 = sampled_class1[:valid_trial_num]

            # 再拼接和 shuffle
            sampled_indices = np.concatenate([sampled_class0, sampled_class1])
            np.random.shuffle(sampled_indices)
            
            num_subjects = valid_trial_num // 20 * 2 # 两个类别

            EEG_final = all_EEG[sampled_indices].reshape(num_subjects, 20, *all_EEG.shape[1:])
            ECG_final = all_ECG[sampled_indices].reshape(num_subjects, 20, *all_ECG.shape[1:])
            GSR_final = all_GSR[sampled_indices].reshape(num_subjects, 20, *all_GSR.shape[1:])
            Label_final = all_Label[sampled_indices].reshape(num_subjects, 20, *all_Label.shape[1:])  # 若只要第二列可用 [:, 1]
                    
            # 用下采样后的索引更新数据
            self.EEG = [torch.tensor(EEG_final[i]) for i in range(num_subjects)]      # 每个被试20个trial
            self.ECG = [torch.tensor(ECG_final[i]) for i in range(num_subjects)]
            self.GSR = [torch.tensor(GSR_final[i]) for i in range(num_subjects)]
            self.Label = [Label_final[i] for i in range(num_subjects)]
        
    def __getitem__(self, index):
        # Read a single sample of data from the data array
        # Return the tensor data
        data = {
            'EEG': self.EEG[index],
            'ECG': self.ECG[index],
            'GSR': self.GSR[index],
            'Labels': self.Label[index],
            'data_files': self.data_files,
        }
        # Return the tensor data
        return (data, index)

    def __reshape_label__(self):
        tmp_label = self.Label
        self.Label = []
        for index in range(self.__len__()):
            subject_index = index // (20*(30//self.time_window))
            video_index = (index % (20*(30//self.time_window))) // (30//self.time_window)
            self.Label.append(tmp_label[subject_index][video_index])

    def __len__(self):
        return len(self.EEG)



class DEAPDataset_Python(Dataset):
    def __init__(self, root_dir = None, data_files = None, label = 'arousal'):
        if root_dir is None:
            root_dir = os.environ.get('NEURO_ROOT', '/home/fss')
        # Get the data files list
        self.root_dir = os.path.join(root_dir, 'Dataset/DEAP-python')
        if data_files is None:
            self.data_files = os.listdir(self.root_dir)
        else:
            self.data_files = data_files
        self.data_files.sort(key=lambda x:int(x[:-4]))
        self.time_window = 6
        # Loading data
        print('loading data...')
        self.EEG = []
        self.EMG = []
        self.GSR = []
        self.Label = []
        for data_file in self.data_files:
            if data_file.endswith('.npy'):
                file_path = os.path.join(self.root_dir, data_file)
            else:
                data_file = data_file + '.npy'
                file_path = os.path.join(self.root_dir, data_file)
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            tmp_EEG = data['data'][:,0:32,384:]
            self.EEG.append(tmp_EEG.reshape(40, 32, 60, 128))
            tmp_EMG = data['data'][:,34:36,384:]
            self.EMG.append(tmp_EMG.reshape(40, 2, 60, 128))
            tmp_GSR = (data['data'][:,36,384:])[:,None,:]
            self.GSR.append(tmp_GSR.reshape(40, 1, 60, 128))
            self.Label.append(torch.tensor(data['labels']))

        # 去除个体差异性
        for index in range(len(data_files)):
            self.EEG[index] = DEAP_eliminate_individual_differences(self.EEG[index])
            self.EMG[index] = DEAP_eliminate_individual_differences(self.EMG[index])
            self.GSR[index] = DEAP_eliminate_individual_differences(self.GSR[index])

        # 为了平衡训练集的样本分布
        if len(data_files) != 1:
            # 合并所有样本到一个列表，方便下采样
            all_EEG = []
            all_EMG = []
            all_GSR = []
            all_Label = []
            for i in range(len(self.EEG)):
                all_EEG.extend(list(self.EEG[i]))
                all_EMG.extend(list(self.EMG[i]))
                all_GSR.extend(list(self.GSR[i]))
                all_Label.extend(list(self.Label[i]))

            # 转为 numpy 数组，才能用索引
            all_EEG = np.array(all_EEG)  # [N, 32, 60, 128]
            all_EMG = np.array(all_EMG)  # [N, 2, 60, 128]
            all_GSR = np.array(all_GSR)  # [N, 1, 60, 128]
            all_Label = np.array(all_Label)  # [N, 4]

            # 下采样代码插入点
            # 统计每个类别的索引  Arousal: 1  Valence: 0
            # 检查标签类型并选择相应的列索引
            column_index = 1 if label == 'arousal' else 0  # 如果是arousal用第二列(索引1)，否则用第一列(索引0)
            labels_np = np.array(all_Label)
            class0_indices = np.where(labels_np[:, column_index] <= 5)[0]
            class1_indices = np.where(labels_np[:, column_index] > 5)[0]

            # 取两类中数量较少的那个，作为下采样目标
            min_len = min(len(class0_indices), len(class1_indices))
            np.random.seed(42)  # 保持可复现
            sampled_class0 = np.random.choice(class0_indices, min_len, replace=False)
            sampled_class1 = np.random.choice(class1_indices, min_len, replace=False)

            # 分别丢弃不能整除40的部分
            valid_trial_num = (min_len // 40) * 40
            sampled_class0 = sampled_class0[:valid_trial_num]
            sampled_class1 = sampled_class1[:valid_trial_num]

            # 再拼接和 shuffle
            sampled_indices = np.concatenate([sampled_class0, sampled_class1])
            np.random.shuffle(sampled_indices)

            num_subjects = valid_trial_num // 40 * 2  # 两个类别

            EEG_final = all_EEG[sampled_indices].reshape(num_subjects, 40, *all_EEG.shape[1:])
            EMG_final = all_EMG[sampled_indices].reshape(num_subjects, 40, *all_EMG.shape[1:])
            GSR_final = all_GSR[sampled_indices].reshape(num_subjects, 40, *all_GSR.shape[1:])
            Label_final = all_Label[sampled_indices].reshape(num_subjects, 40, *all_Label.shape[1:])  # 若只要第二列可用 [:, 1]

            # 用下采样后的索引更新数据
            self.EEG = [torch.tensor(EEG_final[i]) for i in range(num_subjects)]  # 每个被试40个trial
            self.EMG = [torch.tensor(EMG_final[i]) for i in range(num_subjects)]
            self.GSR = [torch.tensor(GSR_final[i]) for i in range(num_subjects)]
            self.Label = [Label_final[i] for i in range(num_subjects)]

    def __getitem__(self, index):
        # Read a single sample of data from the data array
        # Return the tensor data
        data = {
            'EEG': self.EEG[index],
            'EMG': self.EMG[index],
            'GSR': self.GSR[index],
            'Labels': self.Label[index],
            'data_files': self.data_files,
        }
        # Return the tensor data
        return (data, index)

    def __reshape_label__(self):
        tmp_label = self.Label
        self.Label = []
        for index in range(self.__len__()):
            subject_index = index // (40*(60//self.time_window))
            video_index = (index % (40*(60//self.time_window))) // (60//self.time_window)
            self.Label.append(tmp_label[subject_index][video_index])
            
    def __len__(self):
        return len(self.EEG)

# make sure input is list
def to_list(input):
    if isinstance(input, list):
        return input
    elif isinstance(input, int):
        return [input]
    else:
        raise ValueError("Input must be an integer or a list")


from collections import Counter


def HCISubdataset(root_dir, index, label):
    if index == []:
        return []
    # args:
    #   [1, 3, 5]
    file_index = [f"{str(item)}.npy" for item in index]
    #   ['1.npy', '3.npy', '5.npy']
    subset = HCIDataset_Python(root_dir, file_index, label)
    
    def Reshape_data(data_list, time_window = 6):
        if data_list == []:
            return []
        # data_list[index] : videos * channels * (windows * split_size) * timepoints
        # eg. 20, 32, 30(5*6), 128
        videos, channel, time, timepoints = data_list[0].shape

        # Ensure that the time dimension is divisible by time_window
        assert time % time_window == 0, "Time dimension must be divisible by time_window"
        new_data_list = []

        for index in range(0, len(data_list)):
            # Reshape the tensor
            reshaped = data_list[index].reshape(videos, channel, -1, time_window, timepoints)
            reshaped = reshaped.permute(0, 2, 1, 3, 4)
            reshaped = reshaped.reshape(-1, channel, time_window * timepoints)
            new_data_list.extend(reshaped)

        return new_data_list

    subset.EEG, subset.ECG, subset.GSR = Reshape_data(subset.EEG), Reshape_data(subset.ECG), Reshape_data(subset.GSR)

    # deal Label: n * 20 * 2
    # need to be (n*20) * 2
    # this is Coarse Label -> Arousal Valence
    for index in range(0, len(subset.Label)):
        # drop last two dimensions
        if label == 'arousal':
            subset.Label[index] = subset.Label[index][:,1]
        elif label == 'valence':
            subset.Label[index] = subset.Label[index][:,0]

    # 目前Label [1,9]，需要划分至[0,1]
    # https://ieeexplore.ieee.org/document/9096541
    # https://www.mdpi.com/1424-8220/18/8/2739
    # Label     HVHA    HVLA    LVHA    LVLA
    # Valence   >5      >5      <=5     <=5
    # Arousal   >5      <=5     >5      <=5
    for index in range(0, len(subset.Label)):
        for video_index in range(20):
            if subset.Label[index][video_index] <= 5:
                subset.Label[index][video_index] = 0
            else:
                subset.Label[index][video_index] = 1

    # 这一步是扩展原来的类别标签到每一个窗口上
    subset.__reshape_label__()
    return subset




def DEAPSubdataset(root_dir, index, label):
    if index == []:
        return []
    # args:
    #   [1, 3, 5]
    file_index = [f"{str(item)}.npy" for item in index]
    #   ['1.npy', '3.npy', '5.npy']
    subset = DEAPDataset_Python(root_dir, file_index)

    def Reshape_data(data_list, time_window = 6):
        if data_list == []:
            return []
        # data_list[index] : videos * channels * (windows * split_size) * timepoints
        # eg. 40, 32, 60(10*6), 128
        videos, channel, time, timepoints = data_list[0].shape

        # Ensure that the time dimension is divisible by time_window
        assert time % time_window == 0, "Time dimension must be divisible by time_window"
        new_data_list = []

        for index in range(0, len(data_list)):
            # Reshape the tensor
            reshaped = data_list[index].reshape(videos, channel, -1, time_window, timepoints)
            reshaped = reshaped.permute(0, 2, 1, 3, 4)
            reshaped = reshaped.reshape(-1, channel, time_window * timepoints)
            new_data_list.extend(reshaped)

        return new_data_list

    subset.EEG, subset.EMG, subset.GSR = Reshape_data(subset.EEG), Reshape_data(subset.EMG), Reshape_data(subset.GSR)

    # deal Label: n * 40 * 4 
    # need to be (n*40) * 2
    # this is Coarse Label -> Arousal Valence
    for index in range(0, len(subset.Label)):
        # drop last two dimensions
        if label == 'arousal':
            subset.Label[index] = subset.Label[index][:,1]
        elif label == 'valence':
            subset.Label[index] = subset.Label[index][:,0]

    # 目前Label [1,9]，需要划分至[0,1]
    # https://ieeexplore.ieee.org/document/9096541
    # https://www.mdpi.com/1424-8220/18/8/2739
    # Label     HVHA    HVLA    LVHA    LVLA
    # Valence   >5      >5      <=5     <=5
    # Arousal   >5      <=5     >5      <=5
    for index in range(0, len(subset.Label)):
        for video_index in range(40):
            if subset.Label[index][video_index] <= 5:
                subset.Label[index][video_index] = 0
            else:
                subset.Label[index][video_index] = 1

    # 这一步是扩展原来的类别标签到每一个窗口上
    subset.__reshape_label__()
    return subset


class GroupedSampler(Sampler):
    def __init__(self, data_source, group_size, shuffle=True):
        self.data_source = data_source
        self.group_size = group_size
        self.shuffle = shuffle
        self.num_groups = len(data_source) // group_size  # 确定组数

    def __iter__(self):
        # 创建按每10个一组的索引
        indices = np.arange(len(self.data_source))
        grouped_indices = [indices[i:i + self.group_size] for i in range(0, len(self.data_source), self.group_size)]

        # 打乱这些组
        if self.shuffle:
            np.random.shuffle(grouped_indices)

        # 将打乱后的组展开成一个列表，保持原始长度
        shuffled_indices = [idx for group in grouped_indices for idx in group]

        return iter(shuffled_indices)

    def __len__(self):
        return len(self.data_source)

def HCIDataLoader(root_dir, test_index, label, batch_size = 20):
    # don't need to input dataset, instead, input the root dir of dataset
    test_index = to_list(test_index)
    # Leave one subject out
    assert 1 <= min(test_index) and max(test_index) <= 24, "Index must be between 1 and 24 (inclusive)"

    train_index = [i for i in range(1, 25) if i != test_index[0]]
    # train_index = [3]
    test_index = test_index

    group_size = 5

    train_dataloader = DataLoader(HCISubdataset(root_dir, train_index, label), batch_size=batch_size, sampler=GroupedSampler(HCISubdataset(root_dir, train_index, label), group_size))
    test_dataloader = DataLoader(HCISubdataset(root_dir, test_index, label), batch_size=batch_size, sampler=GroupedSampler(HCISubdataset(root_dir, test_index, label), group_size))

    return train_dataloader, test_dataloader


def DEAPDataLoader(root_dir, test_index, label, batch_size = 40):
    # don't need to input dataset, instead, input the root dir of dataset
    test_index = to_list(test_index)
    # Leave one subject out
    assert 1 <= min(test_index) and max(test_index) <= 32, "Index must be between 1 and 32 (inclusive)"

    train_index = [i for i in range(1, 33) if i != test_index[0]]
    # train_index = [14]
    test_index = test_index

    group_size = 10

    train_dataloader = DataLoader(DEAPSubdataset(root_dir, train_index, label), batch_size=batch_size, sampler=GroupedSampler(DEAPSubdataset(root_dir, train_index, label), group_size))
    test_dataloader = DataLoader(DEAPSubdataset(root_dir, test_index, label), batch_size=batch_size, sampler=GroupedSampler(DEAPSubdataset(root_dir, test_index, label), group_size))

    return train_dataloader, test_dataloader



# 根据输入测试集索引计算训练集索引
def cal_train_index(test_index, total_subjects = 32):
    def to_list(input):
        if isinstance(input, list):
            return input
        elif isinstance(input, int):
            return [input]
        else:
            raise ValueError("Input must be an integer or a list")

    test_index = to_list(test_index)
    # Leave one subject out
    assert 1 <= min(test_index) and max(test_index) <= total_subjects, "Index must be between 1 and 32 (inclusive)"
    train_index = [i for i in range(1, total_subjects + 1) if i != test_index[0]]
    # return [3]
    return train_index