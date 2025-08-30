
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import json
import sys
import os
import numpy as np
import csv
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

import models.architectures as arch
from utils.data_loader import load_bcw, load_cifar10, load_cinic10, create_dataloader
from utils.attack_metrics import calculate_attack_metrics, print_attack_metrics
from utils.results_saver import save_attack_results

# --- 日志记录类 ---
class Logger(object):
    def __init__(self, filename="default.log", stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass

# --- 模型与数据加载器的映射 ---
MODEL_MAP = {
    "FCNN_Bottom": arch.FCNN_Bottom,
    "ResNet18_Bottom": arch.ResNet18_Bottom,
}

DATA_LOADER_MAP = {
    "bcw": load_bcw,
    "cifar10": load_cifar10,
    "cinic10": load_cinic10
}

# ============================== Direct LIA Training and Attack ==============================
def train_and_attack(args, X_a_train, X_b_train, y_train):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- 开始在 {args.dataset} 上进行直接LIA攻击训练 ---")
    
    with open('config.json', 'r') as f:
        config = json.load(f)[args.dataset]
    params = config['params']
    model_config = config['vanilla']

    # 创建一个包含原始索引的数据集
    train_indices = torch.arange(len(y_train))
    train_dataset = TensorDataset(X_a_train, X_b_train, y_train, train_indices)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    num_classes = params['num_classes']

    if config['model_type'] == 'fcnn':
        party_a_model = MODEL_MAP[model_config['bottom_model_party_a']](input_dim=params['party_a_features'], output_dim=num_classes)
        party_b_model = MODEL_MAP[model_config['bottom_model_party_b']](input_dim=params['party_b_features'], output_dim=num_classes)
    else: # resnet
        party_a_model = MODEL_MAP[model_config['bottom_model_party_a']](output_dim=num_classes)
        party_b_model = MODEL_MAP[model_config['bottom_model_party_b']](output_dim=num_classes)

    party_a_model, party_b_model = party_a_model.to(device), party_b_model.to(device)
    optimizers = [optim.Adam(m.parameters(), lr=args.lr) for m in [party_a_model, party_b_model]]
    criterion = nn.CrossEntropyLoss()

    inferred_labels = torch.full((len(y_train),), -1, dtype=torch.long)

    print(f"--- 开始VFL训练并同步进行梯度分析 ---")
    for epoch in range(args.epochs):
        for i, (batch_X_a, batch_X_b, batch_y, batch_indices) in enumerate(train_loader):
            batch_X_a, batch_X_b, batch_y = batch_X_a.to(device), batch_X_b.to(device), batch_y.to(device)
            
            for opt in optimizers: opt.zero_grad()

            logits_a = party_a_model(batch_X_a)
            logits_b = party_b_model(batch_X_b)

            grad_a = [None]
            def hook(grad): grad_a[0] = grad
            logits_a.register_hook(hook)

            total_logits = logits_a + logits_b
            loss = criterion(total_logits, batch_y)
            loss.backward()
            for opt in optimizers: opt.step()

            if grad_a[0] is not None:
                inferred_batch_labels = torch.argmin(grad_a[0], dim=1)
                for j, orig_idx in enumerate(batch_indices):
                    inferred_labels[orig_idx.item()] = inferred_batch_labels[j].item()

    print("--- 训练和攻击结束 ---")
    
    # 计算攻击指标
    valid_inferences = inferred_labels != -1
    if valid_inferences.sum().item() > 0:
        # 为计算攻击指标，构造虚拟的predictions
        valid_targets = y_train[valid_inferences]
        valid_predictions = inferred_labels[valid_inferences]
        
        # 获取类别数量
        with open('config.json', 'r') as f:
            config = json.load(f)[args.dataset]
        num_classes = config['params']['num_classes']
        
        # 构造one-hot样式的predictions用于计算top-k指标
        predictions_onehot = torch.zeros(len(valid_predictions), num_classes)
        predictions_onehot[range(len(valid_predictions)), valid_predictions] = 1.0
        
        # 计算攻击指标
        attack_metrics = calculate_attack_metrics(predictions_onehot, valid_targets, num_classes)
        
        print_attack_metrics(attack_metrics, 'Direct_LIA', args.dataset)
    else:
        # 如果没有有效推断，所有指标都为0
        attack_metrics = {'acc': 0.0, 'top1_acc': 0.0, 'top5_acc': 0.0, 'f1_score': 0.0}
        print("警告：没有有效的标签推断，所有攻击指标为0")
    
    return (party_a_model, party_b_model), attack_metrics

# ============================== Main Task Performance Evaluation ==============================
def test_direct_vfl(args, models, X_a_test, X_b_test, y_test):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("--- 评估直接攻击场景下的VFL主任务性能 ---")
    party_a_model, party_b_model = models
    party_a_model.eval()
    party_b_model.eval()

    test_loader = create_dataloader(X_a_test, X_b_test, y_test, batch_size=args.batch_size, shuffle=False)

    correct, total = 0, 0
    with torch.no_grad():
        for batch_X_a, batch_X_b, batch_y in test_loader:
            batch_X_a, batch_X_b, batch_y = batch_X_a.to(device), batch_X_b.to(device), batch_y.to(device)
            logits_a = party_a_model(batch_X_a)
            logits_b = party_b_model(batch_X_b)
            total_logits = logits_a + logits_b
            _, predicted = torch.max(total_logits.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

    main_accuracy = 100 * correct / total if total > 0 else 0
    print(f'VFL Main Accuracy on {args.dataset} test set: {main_accuracy:.2f} %')
    return main_accuracy

# ============================== Save Results ==============================
# 注意：现在只保存攻击指标，不再保存主任务准确率
# 使用utils/results_saver.py中的save_attack_results函数

# ============================== Main Execution ==============================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Direct Label Inference Attack on a simplified VFL.")
    parser.add_argument('--dataset', type=str, required=True, choices=['bcw', 'cifar10', 'cinic10'], help='Dataset for the attack.')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs (1 is often enough for this attack).')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    
    args = parser.parse_args()

    log_dir = 'result'
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f'direct_lia_{args.dataset}_results.txt')
    sys.stdout = Logger(log_file_path, sys.stdout)

    print(f"========== 开始在 {args.dataset} 上进行直接LIA攻击 ==========")

    data_loader = DATA_LOADER_MAP[args.dataset]
    (X_a_train, X_b_train, y_train), (X_a_test, X_b_test, y_test) = data_loader()

    vfl_models, attack_metrics = train_and_attack(args, X_a_train, X_b_train, y_train)

    # 保存攻击评估结果（不再测试主任务准确率）
    save_attack_results('Direct_LIA', args.dataset, attack_metrics)

    print(f"========== 直接LIA攻击流程结束 ==========")
