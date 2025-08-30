"""
攻击评估指标计算工具模块
包含攻击效果的4个评估指标：acc, top-1 acc, top-5 acc, F1 score
"""

import torch
import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support
import torch.nn.functional as F

def calculate_attack_metrics(predictions, targets, num_classes=None):
    """
    计算攻击的4个评估指标
    
    Args:
        predictions (torch.Tensor): 模型预测的logits或概率 [N, num_classes]
        targets (torch.Tensor): 真实标签 [N]
        num_classes (int): 类别数量，如果为None则自动推断
        
    Returns:
        dict: 包含4个攻击评估指标的字典
            - acc: 标准准确率
            - top1_acc: Top-1准确率（与acc相同）
            - top5_acc: Top-5准确率
            - f1_score: F1分数（macro平均）
    """
    if num_classes is None:
        num_classes = predictions.size(1)
    
    # 确保输入是tensor
    if isinstance(predictions, np.ndarray):
        predictions = torch.from_numpy(predictions)
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)
    
    # 移到CPU进行计算
    predictions = predictions.cpu()
    targets = targets.cpu()
    
    batch_size = targets.size(0)
    
    # 1. 计算准确率 (acc) 和 Top-1准确率 (top1_acc)
    _, pred_top1 = predictions.topk(1, 1, True, True)
    pred_top1 = pred_top1.t()
    correct_top1 = pred_top1.eq(targets.view(1, -1).expand_as(pred_top1))
    
    acc = correct_top1.view(-1).float().sum(0, keepdim=True).mul_(100.0 / batch_size).item()
    top1_acc = acc  # Top-1准确率与标准准确率相同
    
    # 2. 计算Top-5准确率
    if num_classes >= 5:
        k = min(5, num_classes)
        _, pred_topk = predictions.topk(k, 1, True, True)
        pred_topk = pred_topk.t()
        correct_topk = pred_topk.eq(targets.view(1, -1).expand_as(pred_topk))
        top5_acc = correct_topk.view(-1).float().sum(0, keepdim=True).mul_(100.0 / batch_size).item()
    else:
        # 如果类别数少于5，Top-5准确率等于Top-1准确率
        top5_acc = top1_acc
    
    # 3. 计算F1分数
    # 获取预测的类别
    _, predicted_labels = torch.max(predictions, 1)
    
    # 转换为numpy进行sklearn计算
    predicted_labels_np = predicted_labels.numpy()
    targets_np = targets.numpy()
    
    # 计算macro F1分数
    if num_classes > 2:
        f1_macro = f1_score(targets_np, predicted_labels_np, average='macro') * 100
    else:
        # 二分类情况
        f1_macro = f1_score(targets_np, predicted_labels_np, average='binary') * 100
    
    return {
        'acc': acc,
        'top1_acc': top1_acc,
        'top5_acc': top5_acc,
        'f1_score': f1_macro
    }

def evaluate_attack_model(model, test_loader, device, num_classes=None):
    """
    使用测试数据加载器评估攻击模型
    
    Args:
        model: 攻击模型
        test_loader: 测试数据加载器
        device: 计算设备
        num_classes: 类别数量
        
    Returns:
        dict: 攻击评估指标
    """
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_data in test_loader:
            # 处理不同的数据格式
            if len(batch_data) == 3:
                batch_X, _, batch_y = batch_data
            else:
                batch_X, batch_y = batch_data
            
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # 获取模型预测
            predictions = model(batch_X)
            
            all_predictions.append(predictions.cpu())
            all_targets.append(batch_y.cpu())
    
    # 合并所有批次的结果
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # 计算攻击指标
    metrics = calculate_attack_metrics(all_predictions, all_targets, num_classes)
    
    return metrics

def print_attack_metrics(metrics, algorithm_name, dataset_name):
    """
    打印攻击评估指标
    
    Args:
        metrics (dict): 攻击评估指标
        algorithm_name (str): 算法名称
        dataset_name (str): 数据集名称
    """
    print(f"\n=== {algorithm_name} 攻击评估指标 ({dataset_name}) ===")
    print(f"准确率 (ACC):     {metrics['acc']:.2f}%")
    print(f"Top-1准确率:      {metrics['top1_acc']:.2f}%")
    print(f"Top-5准确率:      {metrics['top5_acc']:.2f}%")
    print(f"F1分数 (Macro):   {metrics['f1_score']:.2f}%")
    print("=" * 50)

def format_metrics_for_csv(metrics):
    """
    将攻击指标格式化为CSV保存格式
    
    Args:
        metrics (dict): 攻击评估指标
        
    Returns:
        dict: 格式化后的指标字典
    """
    return {
        'attack_acc': f"{metrics['acc']:.2f}",
        'attack_top1': f"{metrics['top1_acc']:.2f}",
        'attack_top5': f"{metrics['top5_acc']:.2f}",
        'attack_f1': f"{metrics['f1_score']:.2f}"
    }
