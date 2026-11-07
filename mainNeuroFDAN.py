import os
import gc
import sys
import torch
import argparse
from datetime import datetime
from torch.optim.lr_scheduler import OneCycleLR, ExponentialLR
sys.path.extend(["/home/fss/NeuroFDAN"])
from NeuroFDAN import NeuroFDAN
from ModelTrain import *


global logger
def get_args(description='CPCC on Emotion Recognition'):
    parser = argparse.ArgumentParser(description=description)
    
    # 添加文件路径参数
    parser.add_argument('--label', type=str, required=True, help="Emotion recognition label, optional: 'arousal' and 'valence'.")
    parser.add_argument('--root_dir', type=str, default='/home/fss', help='The root driectory of the framework.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate.')
    parser.add_argument('--epochs', type=int, default=100, help='Upper epoch limit.')
    parser.add_argument('--batch_size', type=int, default=40, help='Batch size.')
    parser.add_argument('--device', type=int, default=0, help='Training device.')
    parser.add_argument('--start_subject', type=int, default=1, help='Start subject.')
    parser.add_argument('--dataset', type=str, default='DEAP', help='DEAP and HCI dataset.')

    # 解析命令行参数
    args = parser.parse_args()

    return args


def main():
    args = get_args()
    
    if args.dataset == 'DEAP':
        for subject_index in range(args.start_subject, 33):
            eeg_channels = 32
            emg_channels = 2
            gsr_channels = 1
            time_len = 768
            hidden_dim = 32
            num_classes = 1
            model = NeuroFDAN(
                eeg_channels=eeg_channels,
                ecg_channels=emg_channels,
                gsr_channels=gsr_channels,
                input_len=time_len,
                hidden_dim=hidden_dim,
                num_classes=num_classes
            )

            device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=args.lr,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=0.01,
                amsgrad=False,
            )

            # scheduler = OneCycleLR(optimizer, max_lr=0.1, total_steps=(31*40*10*args.epochs)//args.batch_size, pct_start=0.3)
            # scheduler = OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=31*40*10//args.batch_size, epochs=args.epochs, pct_start=0.3, anneal_strategy='cos')
            #scheduler = ExponentialLR(optimizer, gnibuamma=0.98)  # 每次调用使学习率乘以 0.95

            # 尝试不使用scheduler

            # 时间戳作为events路径名
            TIMESTAMP = "{0:%Y-%m-%d---%H-%M-%S/}".format(datetime.now())          
            
            mt = DEAPModelTrainer(
                model,
                root_dir= args.root_dir,
                optimizer= optimizer,
                scheduler= None,
                test_index= subject_index,
                n_epoch= args.epochs,
                batch_size= args.batch_size,
                label= args.label,
                Trainer_name= TIMESTAMP,
                device_id= args.device
            )
            mt.run()

            del model, mt
            gc.collect()
            torch.cuda.empty_cache()
            
    elif args.dataset == 'HCI':
        for subject_index in range(args.start_subject, 25):
            eeg_channels = 32
            ecg_channels = 3
            gsr_channels = 1
            time_len = 768
            hidden_dim = 32
            num_classes = 1
            model = NeuroFDAN(
                eeg_channels=eeg_channels,
                ecg_channels=ecg_channels,
                gsr_channels=gsr_channels,
                input_len=time_len,
                hidden_dim=hidden_dim,
                num_classes=num_classes
            )

            device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=args.lr,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=0.01,
                amsgrad=False,
            )

            # scheduler = OneCycleLR(optimizer, max_lr=0.1, total_steps=(31*40*10*args.epochs)//args.batch_size, pct_start=0.3)
            # scheduler = OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=31*40*10//args.batch_size, epochs=args.epochs, pct_start=0.3, anneal_strategy='cos')
            #scheduler = ExponentialLR(optimizer, gamma=0.98)  # 每次调用使学习率乘以 0.95

            # 时间戳作为events路径名
            TIMESTAMP = "{0:%Y-%m-%d---%H-%M-%S/}".format(datetime.now())

            mt = HCIModelTrainer(
                model,
                root_dir= args.root_dir,
                optimizer= optimizer,
                scheduler= None,
                test_index= subject_index,
                n_epoch= args.epochs,
                batch_size= args.batch_size,
                label= args.label,
                Trainer_name= TIMESTAMP,
                device_id= args.device
            )
            mt.run()

            del model, mt
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()