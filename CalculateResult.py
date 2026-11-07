from venv import logger
import numpy as np
import argparse

global logger
def get_args(description='NeuroFDAN on Emotion Recognition'):
    parser = argparse.ArgumentParser(description=description)
    
    # 添加文件路径参数
    parser.add_argument('--label', type=str, required=True, help="Emotion recognition label, optional: 'arousal' and 'valence'.")
    parser.add_argument('--dataset', type=str, required=True, help="Dataset, optional: 'DEAP' and 'HCI'.")

    # 解析命令行参数
    args = parser.parse_args()

    return args


def main():
    args = get_args()

    file_path = f'/home/zkx/NeuroFDAN/NeuroFDAN-{args.dataset}_{args.label}_cm.npy'
    # shape: (32, 2, 2)
    confusion_matrices = np.load(file_path)
    print(confusion_matrices)

    # 初始化存储结果的数组
    accuracy_list = []
    sensitivity_list = []
    specificity_list = []
    f1_score_list = []

    for cm in confusion_matrices:
        TP = cm[1, 1]
        TN = cm[0, 0]
        FP = cm[1, 0]
        FN = cm[0, 1]

        
        TP = cm[0, 0]
        TN = cm[1, 1]
        FP = cm[0, 1]
        FN = cm[1, 0]
        

        # 计算准确率
        accuracy = (TP + TN) / (TP + TN + FP + FN)

        # 计算敏感度（召回率）
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0

        # 计算特异性
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

        # 计算F1指数
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

        # 将结果添加到列表中
        accuracy_list.append(accuracy)
        sensitivity_list.append(sensitivity)
        specificity_list.append(specificity)
        f1_score_list.append(f1_score)

    # 转换为 numpy 数组以便进一步处理
    accuracy_list = np.array(accuracy_list)
    sensitivity_list = np.array(sensitivity_list)
    specificity_list = np.array(specificity_list)
    f1_score_list = np.array(f1_score_list)

    # 打印每个指标的平均值
    print(f"平均准确率: {np.mean(accuracy_list):.4f}")
    print(f"平均敏感度: {np.mean(sensitivity_list):.4f}")
    print(f"平均特异性: {np.mean(specificity_list):.4f}")
    print(f"平均F1指数: {np.mean(f1_score_list):.4f}")
    print(f"准确率标准差: {np.std(accuracy_list):.4f}")
    print(f"敏感度标准差: {np.std(sensitivity_list):.4f}")
    print(f"特异性标准差: {np.std(specificity_list):.4f}")
    print(f"F1指数标准差: {np.std(f1_score_list):.4f}")

if __name__ == "__main__":
    main()