import os
import gc
from queue import PriorityQueue
import sys
import math
import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import plotly.express as px
import matplotlib.pyplot as plt

from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from copy import deepcopy
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm
from abc import ABC, abstractmethod
from itertools import cycle

sys.path.extend(["/home/zkx/NeuroFDAN"])
from NeuroFDAN import *
from Datasets import *
from loss import CausalMultiLoss, calculate_accuracy, update_confusion_matrix

LOG_DIR = "./logs"
TRAIN_STAGE = "train"
TEST_STAGE = "test"
VAL_STAGE = "val"
AROUSAL = "arousal"
VALENCE = "valence"

lr = []

class DEAPModelTrainer(ABC):
    def __init__(self,
        model,
        root_dir,
        optimizer,
        scheduler=None,
        test_index: int = 1,
        batch_size=40,
        k_fold=32,
        n_epoch=20,
        temperature=13.544,
        Trainer_name="Trainer",
        label=AROUSAL,
        device_id = 0
    ) -> None:
        self.model = model
        self.root_dir = root_dir
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.test_index = test_index
        self.batch_size = batch_size
        self.k_fold = k_fold
        self.n_epoch = n_epoch
        self.temperature = temperature
        self.Trainer_name = Trainer_name
        self.label = label
        self.device_id = device_id
        self.plots = {}
        self.initialization()

    def initialization(self, drop_last=False):
        """
        Define a standard initialization for all modals
            :param drop_last: default False
                              bool, whether to drop the last batch in the dataloader
        """
        model = self.model
        torch.cuda.empty_cache()

        if torch.cuda.is_available():
            gpu_id = self.device_id
            torch.cuda.set_device(gpu_id)
            device = torch.device(f"cuda:{gpu_id}")
        else:
            device = torch.device("cpu")
        # if torch.cuda.device_count() > 1:
        #    model = nn.DataParallel(model)  # Wrap the model with DataParallel
        self.setup_seed(42)
        print(f"Using Device : {torch.cuda.get_device_name()}")
        self.model = model.to(device)
        self.device = device

        # Initialize tensorboard
        from datetime import datetime

        formatted_date = datetime.now().strftime("%y%m%d")
        self.writer = SummaryWriter(log_dir=f"{LOG_DIR}/{formatted_date}/{self.Trainer_name}")

        # 用于知道当前在训练集上是否最优
        self.Best_ACC = 0

        self.loss_type = 'bce'
        self.criterion = CausalMultiLoss(weights={'main':1.0, 'backdoor':0.0, 'frontdoor':0.0}, loss_type=self.loss_type)
        self.criterion = self.criterion.to(self.device)

        # 每次开始运行的时候检查暂存混淆矩阵文件是否为空，如果空的话则建立，不空的话将混淆矩阵值读入self.EVAL
        tmp_confusion_matrix = os.path.join(self.root_dir, 'NeuroFDAN', f'NeuroFDAN-DEAP_{self.label}_cm.npy')
        if not os.path.exists(tmp_confusion_matrix):
            # 存储每个测试被试的混淆矩阵，用于评估模型
            self.EVAL = torch.zeros((self.k_fold), 2, 2)
            np.save(tmp_confusion_matrix, self.EVAL)
        else:
            # 文件存在，加载现有的混淆矩阵
            self.EVAL = np.load(tmp_confusion_matrix)
            self.EVAL = torch.tensor(self.EVAL)
        self.EVAL = self.EVAL.to(self.device)

    def write_logs(self, stage, losses, epoch):
        for key, value in losses.items():
            self.writer.add_scalar(f"{self.test_index}_loss/{stage}_{key}", value, epoch)


    def run(self):
        """
        Define a standard train/val loop  for models
        """
        model = self.model

        # Leave one subject out
        print("training...")
        # 对于每个test_index，训练模型，记录训练过程中Best_ACC
        self.Best_ACC = (self.EVAL[self.test_index-1][0,0] + self.EVAL[self.test_index-1][1,1]) / (40*5)
        print(f"***************Test {self.test_index}***************")
        print(self.EVAL[self.test_index-1])

        # test_index: 1, train_index是余下 23 个
        train_dataloader, test_dataloader = DEAPDataLoader(self.root_dir, self.test_index, self.label, self.batch_size)

        for epoch in range(1, self.n_epoch+1):
            print(f"***************Test {self.test_index}, Epoch {epoch} start training***************")

            # 训练
            model.train()
            self.clear_plots()
            loss = self.model_evaluate(TRAIN_STAGE, train_dataloader, epoch, cal_train_index(self.test_index, total_subjects=32))
            self.write_logs(TRAIN_STAGE, loss, epoch)

            # 测试
            model.eval()
            with torch.no_grad():
                # Test stage
                loss = self.model_evaluate(TEST_STAGE, test_dataloader, epoch, self.test_index)
                self.write_logs(TEST_STAGE, loss, epoch)

            self.plot(f"Epoch_{epoch}.html")

        print(f"*************** Test {self.test_index} confusiont matrix ***************")
        print(self.EVAL[self.test_index - 1])
        # 所有epoch结束，保存测试用例混淆矩阵
        self.tmp_save_confusion_matrix(self.test_index)

        del train_dataloader, test_dataloader
        self.writer.close()

        # 绘制学习率变化曲线
        plt.plot(lr)
        plt.title("Learning Rate Schedule (OneCycleLR)")
        plt.xlabel("Training Steps")
        plt.ylabel("Learning Rate")
        plt.grid(True)
        plt.savefig("epoch.png")

    def model_evaluate(self, stage, data_loader, epoch, index):
        """
        Define a standard evaluate function for all test stages
        args:
            stage: str, 'train', 'test'
            data_loader: torch.utils.data.DataLoader
            epoch: int, current epoch
            index: train_index, test_index
        return:
            loss: dict
        """
        num_trail = len(data_loader.dataset.EEG)
        num_subject = num_trail // 20 // 5

        optimizer = self.optimizer
        enumerator = tqdm(enumerate(data_loader)) if stage == TRAIN_STAGE else enumerate(data_loader)
        criterion = self.criterion
        # epoch: 共n_epoch轮，每轮训练/测试的结果
        epoch_correct = 0
        epoch_loss = 0.0
        epoch_acc = 0.0  # epoch_acc在epoch_correct统计完后计算
        epoch_confusion_matrix = torch.zeros(2, 2)

        epoch_cns_loss = 0.0
        epoch_ans_loss = 0.0
        epoch_cls_loss = 0.0

        epoch_front_loss = 0.0
        epoch_back_loss = 0.0
        epoch_dag_loss = 0.0

        # batch
        for i, data in enumerator:
            # Move the inputs and targets to the GPU (if available)
            EEG_inputs = (data[0]["EEG"]).to(self.device)
            EMG_inputs = (data[0]["EMG"]).to(self.device)
            GSR_inputs = (data[0]["GSR"]).to(self.device)
            labels = (data[0]["Labels"]).to(self.device)

            # 清除梯度
            if stage == TRAIN_STAGE:
                self.optimizer.zero_grad()

            out, cns_dag_penalty, ans_dag_penalty = self.model(EEG_inputs.float(), EMG_inputs.float(), GSR_inputs.float())

            # 当前这个batch的loss
            position_loss, loss_dict = criterion(out, labels, cns_dag_penalty, ans_dag_penalty)
            epoch_loss += position_loss.item()
            epoch_cns_loss += loss_dict['cns']
            epoch_ans_loss += loss_dict['ans']
            epoch_cls_loss += loss_dict['main']

            # 当前这个batch的acc和correct
            _, position_correct = calculate_accuracy(out, labels, loss_type=self.loss_type)
            # epoch, 训练：1个epoch对应 310 个batch
            epoch_correct += position_correct

            # Compute the gradients and update the parameters
            if stage == TRAIN_STAGE:
                position_loss.backward()
                # 梯度裁剪（L2范数）
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()
                # 每个 batch 更新学习率，暂时不是每个 batch 更新了，改成每个 epoch 更新，更新代码在后面
                if self.scheduler is not None:
                    # self.scheduler.step()
                    
                    current_lr = self.scheduler.get_last_lr()
                    lr.append(current_lr)

                # 用于测试代码的，测试的时候加上
                epoch_confusion_matrix = update_confusion_matrix(out, labels, epoch_confusion_matrix, loss_type=self.loss_type)

            # 测试时候记录混淆矩阵，每个batch更新混淆矩阵。测试会进行10个batch
            if stage == TEST_STAGE:
                epoch_confusion_matrix = update_confusion_matrix(out, labels, epoch_confusion_matrix, loss_type=self.loss_type)

        epoch_acc = epoch_correct / num_trail
        epoch_loss = epoch_loss / num_subject
        epoch_cls_loss = epoch_cls_loss / num_subject
        epoch_cns_loss = epoch_cns_loss / num_subject
        epoch_ans_loss = epoch_ans_loss / num_subject

        loss = {
            "epoch_loss": epoch_loss,
            "epoch_cls_loss": epoch_cls_loss,
            "epoch_cns_loss": epoch_cns_loss,
            "epoch_ans_loss": epoch_ans_loss,
            "epoch_acc": epoch_acc,
        }

        print(f"Epoch {epoch}, {stage} Loss: {default_round(loss['epoch_loss'])}, Correct: {epoch_correct}")
        print(epoch_confusion_matrix)

        if stage == TEST_STAGE:

            # 每个epoch更新学习率
            if self.scheduler is not None:
                self.scheduler.step()

            if epoch_acc > self.Best_ACC and epoch_acc > 0.5:
                self.Best_ACC = epoch_acc
                self.EVAL[index - 1] = epoch_confusion_matrix
                # 保存一下混淆矩阵
                self.tmp_save_confusion_matrix(self.test_index)
                # 存一下好结果的模型参数
                torch.save(self.model.state_dict(), "NeuroFDAN-DEAP-{}-{}-{}.pth".format(self.label, index, epoch_correct))

            if epoch_acc == self.Best_ACC:
                # 准确率相同时，根据F1来决定要不要更新
                def F1(confusion):
                    TP = confusion[1, 1]
                    TN = confusion[0, 0]
                    FP = confusion[1, 0]
                    FN = confusion[0, 1]

                    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
                    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
                    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
                    return f1

                if F1(self.EVAL[index-1]) < F1(epoch_confusion_matrix):
                    self.EVAL[index - 1] = epoch_confusion_matrix
                    # 保存一下混淆矩阵
                    self.tmp_save_confusion_matrix(self.test_index)
                    # 存一下好结果的模型参数
                    torch.save(self.model.state_dict(), "NeuroFDAN-DEAP-{}-{}-{}.pth".format(self.label, index, epoch_correct))
        return loss

    def to_device(self, tensors, device=None):
        """
        Send them all to device
        args:
            tensors: list of torch.Tensor
        """
        device = self.device if device is None else device

        return [tensor.to(device) for tensor in tensors]

    # 用于在每个测试用例结束后保存混淆矩阵，避免代码终止造成数据丢失
    def tmp_save_confusion_matrix(self, test_index):
        file_confusion_matrix = os.path.join(self.root_dir, 'NeuroFDAN', f'NeuroFDAN-DEAP_{self.label}_cm.npy')
        if not os.path.exists(file_confusion_matrix):
            # 文件不存在时， 存储进去
            confusion_matrix = self.EVAL
            confusion_matrix = confusion_matrix.cpu()
            confusion_matrix = confusion_matrix.numpy()
            np.save(file_confusion_matrix, confusion_matrix)
        else:
            # 文件存在，只替换文件混淆矩阵的test index位置
            confusion_matrix = np.load(file_confusion_matrix)
            confusion_matrix[test_index - 1] = self.EVAL[test_index - 1].cpu().numpy()
            np.save(file_confusion_matrix, confusion_matrix)

    def setup_seed(self, seed):
        if seed == False:
            return
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    # Plotting functions
    def clear_plots(self):
        self.plots = {}

    def plot(self, file_name):
        """
        a function to plot all plots in self.plots
        args:
            plot_name: str, the name of the plot
        """
        if not self.plots:
            return
        from datetime import datetime

        formatted_date = datetime.now().strftime("%y%m%d")
        for plot_name, plot_eles in self.plots.items():
            plot_save_dir = os.path.join(
                LOG_DIR, formatted_date, self.Trainer_name, plot_name
            )
            os.makedirs(plot_save_dir, exist_ok=True)
            plot_save_path = os.path.join(plot_save_dir, file_name)
            label_mapping = {}
            eles, lables = self.to_device([torch.tensor([]), torch.tensor([])])
            for i, (key, values) in enumerate(plot_eles.items()):
                eles = torch.cat([eles, values])
                lables = torch.cat(
                    [lables, torch.ones(values.shape[0]).to(self.device) * i]
                )
                label_mapping[i] = key
            eles = eles.cpu().detach().numpy()
            lables = lables.cpu().detach().numpy()
            if eles.shape[1] == 2:
                df = pd.DataFrame(data=eles, columns=["x", "y"])
            elif eles.shape[1] == 3:
                df = pd.DataFrame(data=eles, columns=["x", "y", "z"])
            else:
                scaler = StandardScaler()
                eles_normalized = scaler.fit_transform(eles)
                # Apply PCA
                n_components = 3
                pca = PCA(n_components=n_components)
                eles_pca = pca.fit_transform(eles_normalized)
                df = pd.DataFrame(data=eles_pca, columns=["PC1", "PC2", "PC3"])
                df["Labels"] = lables
                explained_var_ratio = pca.explained_variance_ratio_
                info_percentage = [
                    f"{round(ratio * 100, 2)}%" for ratio in explained_var_ratio
                ]
            df["Labels"] = lables
            # 使用 map 方法应用映射
            df["Labels"] = df["Labels"].map(label_mapping)
            # 将 'Domain Labels' 列转换为分类数据类型
            df["Labels"] = df["Labels"].astype("category")

            if eles.shape[1] == 2:
                # Plotting with Plotly
                fig = px.scatter(df, x="x", y="y", color="Labels", symbol="Labels")

            elif eles.shape[1] == 3:
                fig = px.scatter_3d(
                    df, x="x", y="y", z="z", color="Labels", symbol="Labels"
                )
            else:
                # Plotting with Plotly
                fig = px.scatter_3d(
                    df,
                    x="PC1",
                    y="PC2",
                    z="PC3",
                    color="Labels",
                    symbol="Labels",
                    labels={
                        "PC1": f"Principal Component 1 ({info_percentage[0]})",
                        "PC2": f"Principal Component 2 ({info_percentage[1]})",
                        "PC3": f"Principal Component 3 ({info_percentage[2]})",
                    },
                    title="PCA of Normalized Shear Features with Domain Labels",
                )
            fig.write_html(plot_save_path)


class HCIModelTrainer(ABC):
    def __init__(self,
        model,
        root_dir,
        optimizer,
        scheduler=None,
        test_index: int = 1,
        batch_size=20,
        k_fold=24,
        n_epoch=20,
        temperature=13.544,
        Trainer_name="Trainer",
        label=AROUSAL,
        device_id = 0
    ) -> None:
        self.model = model
        self.root_dir = root_dir
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.test_index = test_index
        self.batch_size = batch_size
        self.k_fold = k_fold
        self.n_epoch = n_epoch
        self.temperature = temperature
        self.Trainer_name = Trainer_name
        self.label = label
        self.device_id = device_id
        self.plots = {}
        self.initialization()

    def initialization(self, drop_last=False):
        """
        Define a standard initialization for all modals
            :param drop_last: default False
                              bool, whether to drop the last batch in the dataloader
        """
        model = self.model
        torch.cuda.empty_cache()

        if torch.cuda.is_available():
            gpu_id = self.device_id
            torch.cuda.set_device(gpu_id)
            device = torch.device(f"cuda:{gpu_id}")
        else:
            device = torch.device("cpu")
        # if torch.cuda.device_count() > 1:
        #    model = nn.DataParallel(model)  # Wrap the model with DataParallel
        self.setup_seed(42)
        print(f"Using Device : {torch.cuda.get_device_name()}")
        self.model = model.to(device)
        self.device = device

        # Initialize tensorboard
        from datetime import datetime

        formatted_date = datetime.now().strftime("%y%m%d")
        self.writer = SummaryWriter(log_dir=f"{LOG_DIR}/{formatted_date}/{self.Trainer_name}")

        # 用于知道当前在训练集上是否最优
        self.Best_ACC = 0

        self.loss_type = 'bce'
        self.criterion = CausalMultiLoss(weights={'main':1.0, 'backdoor':0.0, 'frontdoor':0.0}, loss_type=self.loss_type)
        self.criterion = self.criterion.to(self.device)

        # 每次开始运行的时候检查暂存混淆矩阵文件是否为空，如果空的话则建立，不空的话将混淆矩阵值读入self.EVAL
        tmp_confusion_matrix = os.path.join(self.root_dir, 'NeuroFDAN', f'NeuroFDAN-HCI_{self.label}_cm.npy')
        if not os.path.exists(tmp_confusion_matrix):
            # 存储每个测试被试的混淆矩阵，用于评估模型
            self.EVAL = torch.zeros((self.k_fold), 2, 2)
            np.save(tmp_confusion_matrix, self.EVAL)
        else:
            # 文件存在，加载现有的混淆矩阵
            self.EVAL = np.load(tmp_confusion_matrix)
            self.EVAL = torch.tensor(self.EVAL)
        self.EVAL = self.EVAL.to(self.device)

    def write_logs(self, stage, losses, epoch):
        for key, value in losses.items():
            self.writer.add_scalar(f"{self.test_index}_loss/{stage}_{key}", value, epoch)


    def run(self):
        """
        Define a standard train/val loop  for models
        """
        model = self.model

        # Leave one subject out
        print("training...")
        # 对于每个test_index，训练模型，记录训练过程中Best_ACC
        self.Best_ACC = (self.EVAL[self.test_index-1][0,0] + self.EVAL[self.test_index-1][1,1]) / (20*5)
        print(f"***************Test {self.test_index}***************")
        print(self.EVAL[self.test_index-1])

        # test_index: 1, train_index是余下 23 个
        train_dataloader, test_dataloader = HCIDataLoader(self.root_dir, self.test_index, self.label, self.batch_size)

        for epoch in range(1, self.n_epoch+1):
            print(f"***************Test {self.test_index}, Epoch {epoch} start training***************")

            # 训练
            model.train()
            self.clear_plots()
            loss = self.model_evaluate(TRAIN_STAGE, train_dataloader, epoch, cal_train_index(self.test_index, total_subjects=24))
            self.write_logs(TRAIN_STAGE, loss, epoch)

            # 测试
            model.eval()
            with torch.no_grad():
                # Test stage
                loss = self.model_evaluate(TEST_STAGE, test_dataloader, epoch, self.test_index)
                self.write_logs(TEST_STAGE, loss, epoch)

            self.plot(f"Epoch_{epoch}.html")

        print(f"*************** Test {self.test_index} confusiont matrix ***************")
        print(self.EVAL[self.test_index - 1])
        # 所有epoch结束，保存测试用例混淆矩阵
        self.tmp_save_confusion_matrix(self.test_index)

        del train_dataloader, test_dataloader
        self.writer.close()

        # 绘制学习率变化曲线
        plt.plot(lr)
        plt.title("Learning Rate Schedule (OneCycleLR)")
        plt.xlabel("Training Steps")
        plt.ylabel("Learning Rate")
        plt.grid(True)
        plt.savefig("epoch.png")

    def model_evaluate(self, stage, data_loader, epoch, index):
        """
        Define a standard evaluate function for all test stages
        args:
            stage: str, 'train', 'test'
            data_loader: torch.utils.data.DataLoader
            epoch: int, current epoch
            index: train_index, test_index
        return:
            loss: dict
        """
        num_trail = len(data_loader.dataset.EEG)
        num_subject = num_trail // 20 // 5

        optimizer = self.optimizer
        enumerator = tqdm(enumerate(data_loader)) if stage == TRAIN_STAGE else enumerate(data_loader)
        criterion = self.criterion
        # epoch: 共n_epoch轮，每轮训练/测试的结果
        epoch_correct = 0
        epoch_loss = 0.0
        epoch_acc = 0.0  # epoch_acc在epoch_correct统计完后计算
        epoch_confusion_matrix = torch.zeros(2, 2)
        
        epoch_cns_loss = 0.0
        epoch_ans_loss = 0.0
        epoch_cls_loss = 0.0
        
        epoch_front_loss = 0.0
        epoch_back_loss = 0.0
        epoch_dag_loss = 0.0

        # batch
        for i, data in enumerator:
            # Move the inputs and targets to the GPU (if available)
            EEG_inputs = (data[0]["EEG"]).to(self.device)
            ECG_inputs = (data[0]["ECG"]).to(self.device)
            GSR_inputs = (data[0]["GSR"]).to(self.device)
            labels = (data[0]["Labels"]).to(self.device)

            # 清除梯度
            if stage == TRAIN_STAGE:
                self.optimizer.zero_grad()

            out, cns_dag_penalty, ans_dag_penalty, attn, rec_loss, div_loss, fire_rate_loss = self.model(EEG_inputs.float(), ECG_inputs.float(), GSR_inputs.float())

            # 当前这个batch的loss
            position_loss, loss_dict = criterion(out, labels, cns_dag_penalty, ans_dag_penalty)

            position_loss += 0.5 * rec_loss - 0.01 * div_loss + 1.0 * fire_rate_loss

            epoch_loss += position_loss.item()
            epoch_cns_loss += loss_dict['cns']
            epoch_ans_loss += loss_dict['ans']
            epoch_cls_loss += loss_dict['main']

            # 当前这个batch的acc和correct
            _, position_correct = calculate_accuracy(out, labels, loss_type=self.loss_type)
            # epoch, 训练：1个epoch对应 310 个batch
            epoch_correct += position_correct

            # Compute the gradients and update the parameters
            if stage == TRAIN_STAGE:
                position_loss.backward()
                # 梯度裁剪（L2范数）
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()
                # 每个 batch 更新学习率，暂时不是每个 batch 更新了，改成每个 epoch 更新，更新代码在后面
                if self.scheduler is not None:
                    # self.scheduler.step()
                    
                    current_lr = self.scheduler.get_last_lr()
                    lr.append(current_lr)

                # 用于测试代码的，测试的时候加上
                epoch_confusion_matrix = update_confusion_matrix(out, labels, epoch_confusion_matrix, loss_type=self.loss_type)

            # 测试时候记录混淆矩阵，每个batch更新混淆矩阵。测试会进行10个batch
            if stage == TEST_STAGE:
                epoch_confusion_matrix = update_confusion_matrix(out, labels, epoch_confusion_matrix, loss_type=self.loss_type)

        epoch_acc = epoch_correct / num_trail
        epoch_loss = epoch_loss / num_subject
        epoch_cls_loss = epoch_cls_loss / num_subject
        epoch_cns_loss = epoch_cns_loss / num_subject
        epoch_ans_loss = epoch_ans_loss / num_subject

        loss = {
            "epoch_loss": epoch_loss,
            "epoch_cls_loss": epoch_cls_loss,
            "epoch_cns_loss": epoch_cns_loss,
            "epoch_ans_loss": epoch_ans_loss,
            "epoch_acc": epoch_acc,
        }

        print(f"Epoch {epoch}, {stage} Loss: {default_round(loss['epoch_loss'])}, Correct: {epoch_correct}")
        print(epoch_confusion_matrix) 

        if stage == TEST_STAGE:

            # 每个epoch更新学习率
            if self.scheduler is not None:
                self.scheduler.step()
                
            if epoch_acc > self.Best_ACC and epoch_acc > 0.5:
                self.Best_ACC = epoch_acc
                self.EVAL[index - 1] = epoch_confusion_matrix
                # 保存一下混淆矩阵
                self.tmp_save_confusion_matrix(self.test_index)
                # 存一下好结果的模型参数
                torch.save(self.model.state_dict(), "NeuroFDAN-HCI-{}-{}-{}.pth".format(self.label, index, epoch_correct))

            if epoch_acc == self.Best_ACC:
                # 准确率相同时，根据F1来决定要不要更新
                def F1(confusion):
                    TP = confusion[1, 1]
                    TN = confusion[0, 0]
                    FP = confusion[1, 0]
                    FN = confusion[0, 1]

                    TP = confusion[0, 0]
                    TN = confusion[1, 1]
                    FP = confusion[0, 1]
                    FN = confusion[1, 0]
                    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
                    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
                    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
                    return f1

                if F1(self.EVAL[index-1]) < F1(epoch_confusion_matrix):
                    self.EVAL[index - 1] = epoch_confusion_matrix
                    # 保存一下混淆矩阵
                    self.tmp_save_confusion_matrix(self.test_index)
                    # 存一下好结果的模型参数
                    torch.save(self.model.state_dict(), "NeuroFDAN-HCI-{}-{}-{}.pth".format(self.label, index, epoch_correct))
        return loss

    def to_device(self, tensors, device=None):
        """
        Send them all to device
        args:
            tensors: list of torch.Tensor
        """
        device = self.device if device is None else device

        return [tensor.to(device) for tensor in tensors]

    # 用于在每个测试用例结束后保存混淆矩阵，避免代码终止造成数据丢失
    def tmp_save_confusion_matrix(self, test_index):
        file_confusion_matrix = os.path.join(self.root_dir, 'NeuroFDAN', f'NeuroFDAN-HCI_{self.label}_cm.npy')
        if not os.path.exists(file_confusion_matrix):
            # 文件不存在时， 存储进去
            confusion_matrix = self.EVAL
            confusion_matrix = confusion_matrix.cpu()
            confusion_matrix = confusion_matrix.numpy()
            np.save(file_confusion_matrix, confusion_matrix)
        else:
            # 文件存在，只替换文件混淆矩阵的test index位置
            confusion_matrix = np.load(file_confusion_matrix)
            confusion_matrix[test_index - 1] = self.EVAL[test_index - 1].cpu().numpy()
            np.save(file_confusion_matrix, confusion_matrix)

    def setup_seed(self, seed):
        if seed == False:
            return
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    # Plotting functions
    def clear_plots(self):
        self.plots = {}

    def plot(self, file_name):
        """
        a function to plot all plots in self.plots
        args:
            plot_name: str, the name of the plot
        """
        if not self.plots:
            return
        from datetime import datetime

        formatted_date = datetime.now().strftime("%y%m%d")
        for plot_name, plot_eles in self.plots.items():
            plot_save_dir = os.path.join(
                LOG_DIR, formatted_date, self.Trainer_name, plot_name
            )
            os.makedirs(plot_save_dir, exist_ok=True)
            plot_save_path = os.path.join(plot_save_dir, file_name)
            label_mapping = {}
            eles, lables = self.to_device([torch.tensor([]), torch.tensor([])])
            for i, (key, values) in enumerate(plot_eles.items()):
                eles = torch.cat([eles, values])
                lables = torch.cat(
                    [lables, torch.ones(values.shape[0]).to(self.device) * i]
                )
                label_mapping[i] = key
            eles = eles.cpu().detach().numpy()
            lables = lables.cpu().detach().numpy()
            if eles.shape[1] == 2:
                df = pd.DataFrame(data=eles, columns=["x", "y"])
            elif eles.shape[1] == 3:
                df = pd.DataFrame(data=eles, columns=["x", "y", "z"])
            else:
                scaler = StandardScaler()
                eles_normalized = scaler.fit_transform(eles)
                # Apply PCA
                n_components = 3
                pca = PCA(n_components=n_components)
                eles_pca = pca.fit_transform(eles_normalized)
                df = pd.DataFrame(data=eles_pca, columns=["PC1", "PC2", "PC3"])
                df["Labels"] = lables
                explained_var_ratio = pca.explained_variance_ratio_
                info_percentage = [
                    f"{round(ratio * 100, 2)}%" for ratio in explained_var_ratio
                ]
            df["Labels"] = lables
            # 使用 map 方法应用映射
            df["Labels"] = df["Labels"].map(label_mapping)
            # 将 'Domain Labels' 列转换为分类数据类型
            df["Labels"] = df["Labels"].astype("category")

            if eles.shape[1] == 2:
                # Plotting with Plotly
                fig = px.scatter(df, x="x", y="y", color="Labels", symbol="Labels")

            elif eles.shape[1] == 3:
                fig = px.scatter_3d(
                    df, x="x", y="y", z="z", color="Labels", symbol="Labels"
                )
            else:
                # Plotting with Plotly
                fig = px.scatter_3d(
                    df,
                    x="PC1",
                    y="PC2",
                    z="PC3",
                    color="Labels",
                    symbol="Labels",
                    labels={
                        "PC1": f"Principal Component 1 ({info_percentage[0]})",
                        "PC2": f"Principal Component 2 ({info_percentage[1]})",
                        "PC3": f"Principal Component 3 ({info_percentage[2]})",
                    },
                    title="PCA of Normalized Shear Features with Domain Labels",
                )
            fig.write_html(plot_save_path)

def default_round(a):
    return round(a, 4)