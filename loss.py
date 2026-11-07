import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalMultiLoss(nn.Module):
    """
    主任务、后门、前门三路损失，可加权。
    """
    def __init__(self, weights=None, loss_type='ce'):
        """
        weights: dict, 各分支损失的权重，如 {'main':1.0, 'backdoor':1.0, 'frontdoor':1.0}
        loss_type: 'ce' (CrossEntropy) 或 'bce' (BCEWithLogits)
        """
        super().__init__()
        self.weights = weights if weights is not None else {
            'main': 1.0,
            'backdoor': 1.0,
            'frontdoor': 1.0
        }
        self.class_weights = torch.tensor([0.6, 1.5])
        self.pos_weights = torch.tensor([0.5])

        self.loss_type = loss_type
        if loss_type == 'ce':
            self.criterion = nn.CrossEntropyLoss(weights=self.class_weights)
        elif loss_type == 'bce':
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weights)
        else:
            raise ValueError("Unknown loss_type")

    def forward(self,
                main_preds, main_labels,
                cns, ans,
                backdoor_preds=None, backdoor_labels=None,
                frontdoor_preds=None, frontdoor_labels=None):
        """
        main_preds: [B, num_classes], main_labels: [B]
        backdoor_preds: [B, num_classes], backdoor_labels: [B]
        frontdoor_preds: [B, num_classes], frontdoor_labels: [B]
        """
        loss_dict = {}
        total_loss = 0.0
        
        if self.loss_type == 'bce':
            # BCE要求标签是float，且shape为[batch_size, 1]
            if main_labels.dim() == 1:
                main_labels = main_labels.unsqueeze(1)
            if main_labels.dtype != torch.float32:
                main_labels = main_labels.float()
            if backdoor_labels is not None and backdoor_labels.dim() > 1:
                backdoor_labels = backdoor_labels.unsqueeze(1)
            if backdoor_labels is not None and backdoor_labels.dtype != torch.float32:
                backdoor_labels = backdoor_labels.float()
            if frontdoor_labels is not None and frontdoor_labels.dim() > 1:
                frontdoor_labels = frontdoor_labels.unsqueeze(1)
            if frontdoor_labels is not None and frontdoor_labels.dtype != torch.float32:
                frontdoor_labels = frontdoor_labels.float()
        elif self.loss_type == 'ce':
            # CE要求标签是long，且shape为[batch_size]
            if main_labels.dim() > 1:
                main_labels = main_labels.squeeze()
            if main_labels.dtype != torch.long:
                main_labels = main_labels.long()
            if backdoor_labels is not None and backdoor_labels.dim() > 1:
                backdoor_labels = backdoor_labels.squeeze()
            if backdoor_labels is not None and backdoor_labels.dtype != torch.long:
                backdoor_labels = backdoor_labels.long()
            if frontdoor_labels is not None and frontdoor_labels.dim() > 1:
                frontdoor_labels = frontdoor_labels.squeeze()
            if frontdoor_labels is not None and frontdoor_labels.dtype != torch.long:
                frontdoor_labels = frontdoor_labels.long()        
                
        # 主任务损失
        if main_preds is not None and main_labels is not None:
            loss_main = self.criterion(main_preds, main_labels)
            loss_dict['main'] = loss_main
            total_loss += self.weights['main'] * loss_main

        # 后门分支损失
        if backdoor_preds is not None and backdoor_labels is not None:
            loss_backdoor = self.criterion(backdoor_preds, backdoor_labels)
            loss_dict['backdoor'] = loss_backdoor
            total_loss += self.weights['backdoor'] * loss_backdoor

        # 前门分支损失
        if frontdoor_preds is not None and frontdoor_labels is not None:
            loss_frontdoor = self.criterion(frontdoor_preds, frontdoor_labels)
            loss_dict['frontdoor'] = loss_frontdoor
            total_loss += self.weights['frontdoor'] * loss_frontdoor
        
        total_loss += 0.5 * cns + 0.2 * ans
        loss_dict['cns'] = cns
        loss_dict['ans'] = ans

        return total_loss, loss_dict
    
def calculate_accuracy(logits, labels, loss_type='ce'):
    """
    logits: 模型输出, shape 取决于 loss_type
    labels: 真实标签, shape 可能与 logits 不同
    loss_type: 'bce' 或 'ce'
    """
    # 兼容不同 loss_type
    if loss_type == 'bce':
        # 预测
        predictions = (torch.sigmoid(logits).view(-1) > 0.5).float()
        # 标签shape对齐
        labels = labels.float().view(-1)
    elif loss_type == 'ce':
        # 预测
        predictions = torch.argmax(logits, dim=1)
        labels = labels.long().view(-1)
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")

    # 计算准确率
    correct = (predictions == labels).sum().item()
    accuracy = correct / len(labels)
    return accuracy, correct

def update_confusion_matrix(logits, labels, current_confusion_matrix, loss_type='ce'):
    """
    logits: 模型输出
    labels: 真实标签
    current_confusion_matrix: 当前累计的混淆矩阵 (torch.Tensor, shape=[num_classes, num_classes])
    loss_type: 'bce' 或 'ce'
    """
    # 预测和标签处理
    if loss_type == 'bce':
        predictions = (torch.sigmoid(logits).view(-1) > 0.5).long()
        labels = labels.float().view(-1).long()
        num_classes = 2
    elif loss_type == 'ce':
        predictions = torch.argmax(logits, dim=1)
        labels = labels.long().view(-1)
        num_classes = logits.shape[1]
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")

    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for t, p in zip(predictions, labels):
        confusion_matrix[t.long().cpu(), p.long().cpu()] += 1

    updated_confusion_matrix = current_confusion_matrix + confusion_matrix

    return updated_confusion_matrix
