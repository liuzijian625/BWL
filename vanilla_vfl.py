import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import json
import sys
import os
import csv
import pandas as pd

import models.architectures as arch
from utils.data_loader import load_bcw, load_cifar10, load_cinic10, create_dataloader

# --- 日志记录类 (同时输出到屏幕和文件) ---
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
    "FCNN_Top_3": arch.FCNN_Top_3,
    "FCNN_Top_4": arch.FCNN_Top_4,
}

DATA_LOADER_MAP = {
    "bcw": load_bcw,
    "cifar10": load_cifar10,
    "cinic10": load_cinic10
}

# --- 训练函数 (原版 VFL) ---
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    with open('config.json', 'r') as f:
        config = json.load(f)[args.dataset]
    params = config['params']
    model_config = config['vanilla']

    data_loader = DATA_LOADER_MAP[args.dataset]
    (X_a_train, X_b_train, y_train), (X_a_test, X_b_test, y_test) = data_loader()
    
    train_loader = create_dataloader(X_a_train, X_b_train, y_train, batch_size=args.batch_size)

    embedding_dim = params['embedding_dim']
    num_classes = params['num_classes']

    if config['model_type'] == 'fcnn':
        party_a_model = MODEL_MAP[model_config['bottom_model_party_a']](input_dim=params['party_a_features'], output_dim=embedding_dim)
        party_b_model = MODEL_MAP[model_config['bottom_model_party_b']](input_dim=params['party_b_features'], output_dim=embedding_dim)
    else: # resnet
        party_a_model = MODEL_MAP[model_config['bottom_model_party_a']](output_dim=embedding_dim)
        party_b_model = MODEL_MAP[model_config['bottom_model_party_b']](output_dim=embedding_dim)

    top_model = MODEL_MAP[model_config['top_model']](input_dim=embedding_dim * 2, output_dim=num_classes)
    
    # 将所有模型移动到GPU
    party_a_model = party_a_model.to(device)
    party_b_model = party_b_model.to(device)
    top_model = top_model.to(device)

    optimizers = [optim.Adam(m.parameters(), lr=args.lr) for m in [party_a_model, party_b_model, top_model]]
    criterion = nn.CrossEntropyLoss()

    print(f"--- 开始在 {args.dataset} 数据集上进行原版VFL训练 (共 {args.epochs} 个周期) ---")
    
    total_batches = len(train_loader)
    print(f"每个周期包含 {total_batches} 个批次")

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        
        for i, (batch_X_a, batch_X_b, batch_y) in enumerate(train_loader):
            # 将数据移动到GPU
            batch_X_a = batch_X_a.to(device)
            batch_X_b = batch_X_b.to(device)
            batch_y = batch_y.to(device)
            
            for opt in optimizers: opt.zero_grad()

            E_a = party_a_model(batch_X_a)
            E_b = party_b_model(batch_X_b)
            E_fused = torch.cat((E_a, E_b), dim=1)
            prediction = top_model(E_fused)
            
            loss = criterion(prediction, batch_y)
            loss.backward()
            for opt in optimizers: opt.step()
            
            # 累积损失
            epoch_loss += loss.item()
            
            # 细粒度输出
            if (i + 1) % args.print_freq == 0 or (i + 1) == total_batches:
                avg_loss = epoch_loss / (i + 1)
                print(f'周期 [{epoch+1}/{args.epochs}], 批次 [{i+1}/{total_batches}], '
                      f'损失: {loss.item():.4f}, 平均损失: {avg_loss:.4f}')

        # 周期结束输出
        avg_loss = epoch_loss / total_batches
        print(f'=== 周期 [{epoch+1}/{args.epochs}] 完成 === 平均损失: {avg_loss:.4f}')

    print("--- 训练结束 ---")
    
    models = (party_a_model, party_b_model, top_model)
    test_data = (X_a_test, X_b_test, y_test)
    return models, test_data, config

# --- 测试函数 (原版 VFL) ---
def test(args, models, test_data, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("--- 开始测试 ---")
    party_a_model, party_b_model, top_model = models
    X_a_test, X_b_test, y_test = test_data

    test_loader = create_dataloader(X_a_test, X_b_test, y_test, batch_size=args.batch_size, shuffle=False)

    for m in models: m.eval()

    correct, total = 0, 0
    with torch.no_grad():
        for batch_X_a, batch_X_b, batch_y in test_loader:
            # 将数据移动到GPU
            batch_X_a = batch_X_a.to(device)
            batch_X_b = batch_X_b.to(device)
            batch_y = batch_y.to(device)
            E_a = party_a_model(batch_X_a)
            E_b = party_b_model(batch_X_b)
            E_fused = torch.cat((E_a, E_b), dim=1)
            prediction = top_model(E_fused)
            _, predicted = torch.max(prediction.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

    print(f'原版VFL模型在 {args.dataset} 测试集上的准确率: {100 * correct / total:.2f} %')

    # 将结果保存到CSV文件（避免重复记录）
    results_file = os.path.join('result', 'results.csv')
    global_acc = f'{100 * correct / total:.2f}'
    
    # 检查是否已存在相同的记录
    should_write = True
    if os.path.isfile(results_file):
        try:
            existing_df = pd.read_csv(results_file)
            # 检查是否已有相同的算法、数据集和全局准确率的记录
            duplicate = existing_df[
                (existing_df['algorithm'] == 'Vanilla_VFL') &
                (existing_df['dataset'] == args.dataset) &
                (existing_df['global_accuracy'] == global_acc) &
                (existing_df['local_accuracy'] == 'N/A')
            ]
            if not duplicate.empty:
                print(f"结果已存在于CSV文件中，跳过重复记录")
                should_write = False
        except (pd.errors.EmptyDataError, FileNotFoundError):
            should_write = True
    
    if should_write:
        file_exists = os.path.isfile(results_file)
        with open(results_file, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['algorithm', 'dataset', 'global_accuracy', 'local_accuracy']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                'algorithm': 'Vanilla_VFL',
                'dataset': args.dataset,
                'global_accuracy': global_acc,
                'local_accuracy': 'N/A'
            })
            print(f"结果已保存到 {results_file}")

# --- 主程序执行 ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Vanilla VFL training and testing.")
    parser.add_argument('--dataset', type=str, required=True, choices=['bcw', 'cifar10', 'cinic10'], help='Dataset to use for training and testing.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--print_freq', type=int, default=10, help='每多少个batch输出一次训练进度.')
    
    args = parser.parse_args()

    # 设置日志
    log_dir = 'result'
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f'vanilla_{args.dataset}_results.txt')
    sys.stdout = Logger(log_file_path, sys.stdout)

    trained_models, test_data, config = train(args)
    test(args, trained_models, test_data, config)