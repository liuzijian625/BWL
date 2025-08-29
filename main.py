
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import json
import numpy as np
import sys
import os
import csv
import pandas as pd

import models.architectures as arch
from losses.boundary_wandering_loss import boundary_wandering_loss
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
    "FCNN_Shadow": arch.FCNN_Shadow,
    "FCNN_Private": arch.FCNN_Private,
    "ResNet18_Bottom": arch.ResNet18_Bottom,
    "ResNet18_Shadow": arch.ResNet18_Shadow,
    "ResNet18_Private": arch.ResNet18_Private,
    "FCNN_Top_3": arch.FCNN_Top_3,
    "FCNN_Top_4": arch.FCNN_Top_4,
    "LocalHead": arch.LocalHead
}

DATA_LOADER_MAP = {
    "bcw": load_bcw,
    "cifar10": load_cifar10,
    "cinic10": load_cinic10
}

# --- 训练函数 ---
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    with open('config.json', 'r') as f:
        config = json.load(f)[args.dataset]
    params = config['params']
    model_config = config['bwl']

    data_loader = DATA_LOADER_MAP[args.dataset]
    (X_a_train, X_b_train, y_train), (X_a_test, X_b_test, y_test) = data_loader()
    train_loader = create_dataloader(X_a_train, X_b_train, y_train, batch_size=args.batch_size)

    if config['model_type'] == 'fcnn':
        total_b_features = params['party_b_features']
        ratio = params['public_feature_ratio']
        num_public = int(total_b_features * ratio)
        indices = np.arange(total_b_features)
        np.random.shuffle(indices)
        public_indices, private_indices = indices[:num_public], indices[num_public:]
        num_public_features = len(public_indices)
        num_private_features = len(private_indices)
    else:
        public_indices, private_indices = None, None

    embedding_dim = params['embedding_dim']
    num_classes = params['num_classes']

    if config['model_type'] == 'fcnn':
        party_a_model = MODEL_MAP[model_config['bottom_model_party_a']](input_dim=params['party_a_features'], output_dim=embedding_dim)
        shadow_model = MODEL_MAP[model_config['bottom_model_shadow']](input_dim=num_public_features, output_dim=embedding_dim)
        private_model = MODEL_MAP[model_config['bottom_model_private']](input_dim=num_private_features, output_dim=embedding_dim)
    else:
        party_a_model = MODEL_MAP[model_config['bottom_model_party_a']](output_dim=embedding_dim)
        shadow_model = MODEL_MAP[model_config['bottom_model_shadow']](output_dim=embedding_dim)
        private_model = MODEL_MAP[model_config['bottom_model_private']](output_dim=embedding_dim)

    top_model = MODEL_MAP[model_config['top_model']](input_dim=embedding_dim * 2, output_dim=num_classes)
    local_head = MODEL_MAP['LocalHead'](input_dim=embedding_dim * 2, output_dim=num_classes)
    
    # 将所有模型移动到GPU
    party_a_model = party_a_model.to(device)
    shadow_model = shadow_model.to(device)
    private_model = private_model.to(device)
    top_model = top_model.to(device)
    local_head = local_head.to(device)

    optimizers = [optim.Adam(m.parameters(), lr=args.lr) for m in [party_a_model, shadow_model, private_model, top_model, local_head]]
    criterion = nn.CrossEntropyLoss()

    print(f"--- 开始在 {args.dataset} 数据集上进行BWL训练 (共 {args.epochs} 个周期) ---")
    
    total_batches = len(train_loader)
    print(f"每个周期包含 {total_batches} 个批次")

    for epoch in range(args.epochs):
        epoch_pred_loss = 0.0
        epoch_bw_loss = 0.0
        
        for i, (batch_X_a, batch_X_b, batch_y) in enumerate(train_loader):
            # 将数据移动到GPU
            batch_X_a = batch_X_a.to(device)
            batch_X_b = batch_X_b.to(device)
            batch_y = batch_y.to(device)
            
            for opt in optimizers: opt.zero_grad()

            if config['model_type'] == 'fcnn':
                batch_X_b_public = batch_X_b[:, public_indices]
                batch_X_b_private = batch_X_b[:, private_indices]
            else:
                batch_X_b_public = batch_X_b[:, :, :16, :]
                batch_X_b_private = batch_X_b[:, :, 16:, :]

            E_a = party_a_model(batch_X_a)
            E_shadow = shadow_model(batch_X_b_public)
            E_private = private_model(batch_X_b_private)
            E_fused_top = torch.cat((E_a, E_shadow), dim=1)
            prediction = top_model(E_fused_top)
            
            pred_loss = criterion(prediction, batch_y)
            bw_loss = boundary_wandering_loss(E_shadow, batch_y)
            total_loss = pred_loss + args.alpha * bw_loss

            total_loss.backward(retain_graph=True)
            optimizers[0].step(); optimizers[3].step(); optimizers[1].step()

            optimizers[2].zero_grad(); optimizers[4].zero_grad()
            E_fused_local = torch.cat((E_shadow.detach(), E_private), dim=1)
            local_prediction = local_head(E_fused_local)
            local_loss = criterion(local_prediction, batch_y)
            local_loss.backward()
            optimizers[2].step(); optimizers[4].step()
            
            # 累积损失
            epoch_pred_loss += pred_loss.item()
            epoch_bw_loss += bw_loss.item()
            
            # 细粒度输出
            if (i + 1) % args.print_freq == 0 or (i + 1) == total_batches:
                avg_pred_loss = epoch_pred_loss / (i + 1)
                avg_bw_loss = epoch_bw_loss / (i + 1)
                print(f'周期 [{epoch+1}/{args.epochs}], 批次 [{i+1}/{total_batches}], '
                      f'预测损失: {pred_loss.item():.4f}, 边界徘徊损失: {bw_loss.item():.4f}, '
                      f'平均预测损失: {avg_pred_loss:.4f}, 平均边界徘徊损失: {avg_bw_loss:.4f}')

        # 周期结束输出
        avg_pred_loss = epoch_pred_loss / total_batches
        avg_bw_loss = epoch_bw_loss / total_batches
        print(f'=== 周期 [{epoch+1}/{args.epochs}] 完成 === '
              f'平均预测损失: {avg_pred_loss:.4f}, 平均边界徘徊损失: {avg_bw_loss:.4f}')

    print("--- 训练结束 ---")
    
    models = (party_a_model, shadow_model, private_model, top_model, local_head)
    test_data = (X_a_test, X_b_test, y_test)
    split_indices = (public_indices, private_indices)
    return models, test_data, split_indices, config

# --- 测试函数 ---
def test(args, models, test_data, split_indices, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("--- 开始测试 ---")
    party_a_model, shadow_model, private_model, top_model, local_head = models
    X_a_test, X_b_test, y_test = test_data
    public_indices, private_indices = split_indices

    test_loader = create_dataloader(X_a_test, X_b_test, y_test, batch_size=args.batch_size, shuffle=False)

    for m in models: m.eval()

    correct_global, correct_local, total = 0, 0, 0
    with torch.no_grad():
        for batch_X_a, batch_X_b, batch_y in test_loader:
            # 将数据移动到GPU
            batch_X_a = batch_X_a.to(device)
            batch_X_b = batch_X_b.to(device)
            batch_y = batch_y.to(device)
            if config['model_type'] == 'fcnn':
                batch_X_b_public = batch_X_b[:, public_indices]
                batch_X_b_private = batch_X_b[:, private_indices]
            else:
                batch_X_b_public = batch_X_b[:, :, :16, :]
                batch_X_b_private = batch_X_b[:, :, 16:, :]

            E_a = party_a_model(batch_X_a)
            E_shadow = shadow_model(batch_X_b_public)
            E_private = private_model(batch_X_b_private)

            E_fused_top = torch.cat((E_a, E_shadow), dim=1)
            global_prediction = top_model(E_fused_top)
            _, predicted_global = torch.max(global_prediction.data, 1)
            
            E_fused_local = torch.cat((E_shadow, E_private), dim=1)
            local_prediction = local_head(E_fused_local)
            _, predicted_local = torch.max(local_prediction.data, 1)

            total += batch_y.size(0)
            correct_global += (predicted_global == batch_y).sum().item()
            correct_local += (predicted_local == batch_y).sum().item()

    print(f'BWL全局模型在 {args.dataset} 测试集上的准确率: {100 * correct_global / total:.2f} %')
    print(f'BWL防御方本地模型在 {args.dataset} 测试集上的准确率: {100 * correct_local / total:.2f} %')

    # 将结果保存到CSV文件（避免重复记录）
    results_file = os.path.join('result', 'results.csv')
    global_acc = f'{100 * correct_global / total:.2f}'
    local_acc = f'{100 * correct_local / total:.2f}'
    
    # 检查是否已存在相同的记录
    should_write = True
    if os.path.isfile(results_file):
        try:
            existing_df = pd.read_csv(results_file)
            # 检查是否已有相同的算法、数据集、全局准确率和本地准确率的记录
            duplicate = existing_df[
                (existing_df['algorithm'] == 'BWL') &
                (existing_df['dataset'] == args.dataset) &
                (existing_df['global_accuracy'] == global_acc) &
                (existing_df['local_accuracy'] == local_acc)
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
                'algorithm': 'BWL',
                'dataset': args.dataset,
                'global_accuracy': global_acc,
                'local_accuracy': local_acc
            })
            print(f"结果已保存到 {results_file}")

# --- 主程序执行 ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="运行BWL算法，训练并立即测试.")
    parser.add_argument('--dataset', type=str, required=True, choices=['bcw', 'cifar10', 'cinic10'], help='用于训练和测试的数据集.')
    parser.add_argument('--epochs', type=int, default=10, help='训练周期数.')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率.')
    parser.add_argument('--batch_size', type=int, default=64, help='批处理大小.')
    parser.add_argument('--alpha', type=float, default=0.5, help='边界徘徊损失的权重.')
    parser.add_argument('--print_freq', type=int, default=10, help='每多少个batch输出一次训练进度.')
    
    args = parser.parse_args()

    # 设置日志
    log_dir = 'result'
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f'bwl_{args.dataset}_results.txt')
    sys.stdout = Logger(log_file_path, sys.stdout)
    
    # 训练并立即测试
    trained_models, test_data, split_indices, config = train(args)
    test(args, trained_models, test_data, split_indices, config)
