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

import models.architectures as arch
from utils.data_loader import load_bcw, load_cifar10, load_cinic10, create_dataloader

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

    train_loader = create_dataloader(X_a_train, X_b_train, y_train, batch_size=args.batch_size)
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
    sample_indices_map = {i: idx for idx, i in enumerate(train_loader.dataset.indices)}

    print(f"--- 开始VFL训练并同步进行梯度分析 ---")
    for epoch in range(args.epochs):
        for i, (batch_X_a, batch_X_b, batch_y) in enumerate(train_loader):
            original_indices = train_loader.dataset.indices[i*args.batch_size : i*args.batch_size+len(batch_X_a)]
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
                for j, orig_idx in enumerate(original_indices):
                    inferred_labels[sample_indices_map[orig_idx]] = inferred_batch_labels[j].item()

    print("--- 训练和攻击结束 ---")
    
    valid_inferences = inferred_labels != -1
    correct_total = (inferred_labels[valid_inferences] == y_train[valid_inferences]).sum().item()
    total_inferred = valid_inferences.sum().item()
    attack_accuracy = 100 * correct_total / total_inferred if total_inferred > 0 else 0
    
    print(f'直接标签推断攻击在 {args.dataset} 训练集上的准确率: {attack_accuracy:.2f} %')
    return (party_a_model, party_b_model), attack_accuracy

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

    main_task_accuracy = 100 * correct / total
    print(f'VFL主任务在 {args.dataset} 测试集上的准确率: {main_task_accuracy:.2f} %')
    return main_task_accuracy

# ============================== Save Results ==============================
def save_results(args, main_task_accuracy, attack_accuracy):
    results_file = os.path.join('result', 'results.csv')
    should_write_header = not os.path.isfile(results_file)
    if not should_write_header:
        try:
            existing_df = pd.read_csv(results_file)
            record_exists = ((existing_df['algorithm'] == 'Direct_LIA') &
                           (existing_df['dataset'] == args.dataset) &
                           (existing_df['main_task_accuracy'] == f'{main_task_accuracy:.2f}') &
                           (existing_df['attack_accuracy'] == f'{attack_accuracy:.2f}')).any()
            if record_exists:
                print("结果已存在于CSV文件中，跳过重复记录。")
                return
        except (pd.errors.EmptyDataError, FileNotFoundError):
            pass

    with open(results_file, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['algorithm', 'dataset', 'main_task_accuracy', 'attack_accuracy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if should_write_header:
            writer.writeheader()
        writer.writerow({
            'algorithm': 'Direct_LIA',
            'dataset': args.dataset,
            'main_task_accuracy': f'{main_task_accuracy:.2f}',
            'attack_accuracy': f'{attack_accuracy:.2f}'
        })
    print(f"结果已保存到 {results_file}")

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

    # 1. 加载数据
    data_loader = DATA_LOADER_MAP[args.dataset]
    (X_a_train, X_b_train, y_train), (X_a_test, X_b_test, y_test) = data_loader()

    # 2. 训练并实施攻击
    vfl_models, attack_accuracy = train_and_attack(args, X_a_train, X_b_train, y_train)

    # 3. 评估主任务性能
    main_task_accuracy = test_direct_vfl(args, vfl_models, X_a_test, X_b_test, y_test)

    # 4. 保存结果
    save_results(args, main_task_accuracy, attack_accuracy)

    print(f"========== 直接LIA攻击流程结束 ==========")