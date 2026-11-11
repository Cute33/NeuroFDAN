"""服务器端NeuroFDAN训练入口；默认完整LOSO，显式支持单fold接口检查。"""
import argparse
import logging
import math
import os


def get_args(description='NeuroFDAN第二阶段LOSO训练', argv=None):
    parser = argparse.ArgumentParser(description=description, add_help=False,
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-h', '--help', action='help', help='显示中文参数说明并退出')
    parser.add_argument('--dataset', default='DEAP', help='DEAP或HCI/MAHNOB_HCI')
    parser.add_argument('--label', '--task', dest='label', required=True, help='arousal或valence，不区分大小写')
    parser.add_argument('--root_dir', default=os.environ.get('NEURO_ROOT', '/home/fss'), help='包含Dataset子目录的数据根目录')
    parser.add_argument('--output_dir', default='outputs/phase2', help='第二阶段结果目录，不与旧实验结果混用')
    parser.add_argument('--run_name', default=None, help='实验名称；默认由前门模式、路径消融和参考维度生成')
    parser.add_argument('--seed', type=int, choices=[42], default=42, help='固定随机种子，初始化模型及数据之前设置')
    parser.add_argument('--device', default='auto', help='auto、cpu、cuda:0或旧版GPU数字编号')
    parser.add_argument('--lr', type=float, default=.0001, help='Adam初始学习率')
    parser.add_argument('--weight_decay', type=float, default=.01,
                        help='Adam耦合L2系数；默认沿用0.01，零仅用于显式单变量对照，不切换AdamW')
    parser.add_argument('--epochs', type=int, default=100, help='每个fold的训练轮数')
    parser.add_argument('--batch_size', type=int, default=40, help='训练/测试batch大小，训练末尾不足batch明确丢弃')
    parser.add_argument('--workers', type=int, default=0, help='DataLoader进程数，内存有限时保持零')
    parser.add_argument('--sampling_mode', choices=['grouped', 'window'], default='grouped',
                        help='优化窗口排列：默认按trial整组打乱；window逐窗口打乱，不改变样本选择或参考库')
    parser.add_argument('--cpu_threads', type=int, default=0, help='CPU计算线程数；零保留PyTorch默认，接口检查可设1')
    parser.add_argument('--start_subject', type=int, default=1, help='完整LOSO起始被试，包含该编号')
    parser.add_argument('--end_subject', type=int, default=None, help='LOSO末尾被试，默认数据集最后一名')
    parser.add_argument('--held_out_subjects', '--held_out_subject', type=int, nargs='+', default=None, help='显式指定一个或多个held-out被试，覆盖起止范围')
    parser.add_argument('--no_balance', action='store_true', help='显式关闭既有trial标签平衡；参考总体始终不做标签平衡')
    parser.add_argument('--fd_mode', choices=['full', 'none'], default='full', help='完整前门概率或Phase 1事实路径对照')
    parser.add_argument('--no_cns2ans', action='store_true', help='估计器消融：切断CNS中介到ANS电流的路径')
    parser.add_argument('--mediator_dim', type=int, default=64, help='CNS和ANS潜在中介维度')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Phase 1中介及结果模型隐藏维度，不改变卷积通道数')
    parser.add_argument('--dropout', type=float, default=.3, help='结果模型dropout')
    parser.add_argument('--ref_dim', type=int, default=0, help='零完整保留当前骨干特征；正整数为额外投影维度')
    parser.add_argument('--S_train', type=int, default=4, help='训练中介采样次数')
    parser.add_argument('--K_train', type=int, default=16, help='训练经验参考采样次数')
    parser.add_argument('--S_eval', type=int, default=16, help='测试中介采样次数')
    parser.add_argument('--K_eval', type=int, default=64, help='测试经验参考采样次数')
    parser.add_argument('--N_ref', type=int, default=2048, help='从全部未平衡训练窗口抽取的参考库容量')
    parser.add_argument('--reference_chunk_size', type=int, default=4, help='结果模型每次计算的K分块大小，零不分块，概率积分公式不变')
    parser.add_argument('--beta', type=float, default=.001, help='标准正态KL正则系数')
    parser.add_argument('--beta_warmup_epochs', type=int, default=10, help='KL系数线性warmup轮数')
    parser.add_argument('--lambda_obs', type=float, default=1., help='事实分类CE权重，不用观察标签监督干预概率')
    parser.add_argument('--fire_weight', type=float, default=None, help='发放率正则权重，默认沿用HCI为1、DEAP为0')
    parser.add_argument('--time_steps', type=int, default=4, help='Phase 1独立中介LIF的驱动步数，非原始窗口采样点数')
    parser.add_argument('--logvar_min', type=float, default=-6., help='中介log方差下界')
    parser.add_argument('--logvar_max', type=float, default=2., help='中介log方差上界')
    parser.add_argument('--resume', default=None, help='单fold加载本项目checkpoint继续训练，支持CPU/CUDA映射')
    parser.add_argument('--max_train_batches', type=int, default=0, help='仅用于服务器接口检查；零执行完整训练epoch')
    parser.add_argument('--max_test_batches', type=int, default=0, help='仅用于服务器接口检查；零完整测试')
    parser.add_argument('--diagnostic_level', choices=['basic', 'full'], default='basic',
                        help='默认逐轮完整训练/测试矩阵与概率统计；full增加中介分布和裁剪前梯度；不改变训练目标')
    parser.add_argument('--log_detail_interval', type=int, default=10,
                        help='终端简洁诊断摘要间隔；每轮仍打印两组紧凑矩阵，零关闭周期摘要；不影响逐轮诊断保存')
    args = parser.parse_args(argv)
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error('weight_decay必须为有限非负数')
    if args.batch_size < 2 or args.epochs < 1 or args.workers < 0 or args.cpu_threads < 0 or args.lr <= 0:
        parser.error('batch至少2、epochs至少1、workers非负且学习率为正')
    if min(args.max_train_batches, args.max_test_batches) < 0 or args.lambda_obs < 0 or (args.fire_weight is not None and args.fire_weight < 0):
        parser.error('接口检查batch上限及损失权重不得为负')
    if args.log_detail_interval < 0:
        parser.error('终端诊断摘要间隔不得为负')
    from Datasets import canonical_dataset, canonical_label
    try:
        args.dataset, args.label = canonical_dataset(args.dataset), canonical_label(args.label)
        from ModelTrain import frontdoor_config
        frontdoor_config(args)
    except ValueError as error:
        parser.error(str(error))
    return args


def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    if args.cpu_threads:
        import torch
        torch.set_num_threads(args.cpu_threads)
    from ModelTrain import run_loso
    run_loso(args)


if __name__ == '__main__':
    main()
