"""
基线版本：没有任何BWL组件的标准垂直联邦学习
用于消融实验的对照组，验证BWL机制的整体贡献
"""

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
from utils.data_loader import load_bcw, load_cifar10, load_cinic10, create_dataloader
from utils.feature_partition import partition_features

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
    "FCNN_Shadow": arch.FCNN_Shadow,
    "FCNN_Private": arch.FCNN_Private,
    "ResNet18_Bottom": arch.ResNet18_Bottom,
    "ResNet18_Shadow": arch.ResNet18_Shadow,
    "ResNet18_Private": arch.ResNet18_Private,
    "FCNN_Top_3": arch.FCNN_Top_3,
    "FCNN_Top_4": arch.FCNN_Top_4,
    "FCNN_Main_Top_3": arch.FCNN_Main_Top_3,
    "FCNN_Main_Top_4": arch.FCNN_Main_Top_4,
    "LocalHead": arch.LocalHead
}

DATA_LOADER_MAP = {
    "bcw": load_bcw,
    "cifar10": load_cifar10,
    "cinic10": load_cinic10
}

# --- 训练函数 ---
def train(args):
    """
    基线训练：没有任何BWL机制的标准垂直联邦学习
    - 没有边界漫游损失
    - 没有影子模型架构
    - 没有特征划分（或使用简单的随机划分）
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    with open('config.json', 'r') as f:
        config = json.load(f)[args.dataset]
    params = config['params']
    model_config = config['bwl']  # 借用BWL配置中的模型结构

    data_loader = DATA_LOADER_MAP[args.dataset]
    (X_a_train, X_b_train, y_train), (X_a_test, X_b_test, y_test) = data_loader()
    train_loader = create_dataloader(X_a_train, X_b_train, y_train, batch_size=args.batch_size)

    print(f"基线实验配置：无BWL机制，纯标准VFL")

    if config['model_type'] == 'fcnn':
        # 对于基线版本，可以选择使用特征划分或不使用
        if args.use_partition:
            # 获取特征划分配置
            feature_partition_config = config.get('feature_partition', {})
            partition_method = args.partition_method if args.partition_method else feature_partition_config.get('method', 'random')
            private_ratio = args.private_ratio if args.private_ratio else feature_partition_config.get('private_ratio', 0.3)
            random_state = feature_partition_config.get('random_state', 42)
            shap_model_type = feature_partition_config.get('shap_model_type', 'xgboost')
            
            print(f"使用{partition_method}方法进行特征划分")
            public_indices, private_indices = partition_features(
                X_b_train, y_train,
                method=partition_method,
                private_ratio=private_ratio,
                random_state=random_state,
                model_type=shap_model_type
            )
            num_public_features = len(public_indices)
            num_private_features = len(private_indices)
            
            print(f"特征划分结果：公开特征 {num_public_features} 个，私有特征 {num_private_features} 个")
        else:
            # 不使用特征划分，Party B的所有特征作为一个整体
            print("不使用特征划分，Party B所有特征作为整体处理")
            public_indices, private_indices = None, None
            num_public_features = params['party_b_features']
            num_private_features = 0
    else:
        public_indices, private_indices = None, None

    embedding_dim = params['embedding_dim']
    num_classes = params['num_classes']

    # 简化模型架构：标准VFL架构
    if config['model_type'] == 'fcnn':
        party_a_model = MODEL_MAP[model_config['bottom_model_party_a']](input_dim=params['party_a_features'], output_dim=embedding_dim)
        
        if args.use_partition:
            # 使用分离的公开和私有模型
            public_model = MODEL_MAP[model_config['bottom_model_public']](input_dim=num_public_features, output_dim=embedding_dim)
            private_model = MODEL_MAP[model_config['bottom_model_private']](input_dim=num_private_features, output_dim=embedding_dim)
            top_model = MODEL_MAP[model_config['main_top_model']](input_dim=embedding_dim * 3, output_dim=num_classes)
        else:
            # 使用单一的Party B模型
            party_b_model = MODEL_MAP[model_config['bottom_model_public']](input_dim=num_public_features, output_dim=embedding_dim)
            top_model = MODEL_MAP[model_config['shadow_top_model']](input_dim=embedding_dim * 2, output_dim=num_classes)
    else:
        party_a_model = MODEL_MAP[model_config['bottom_model_party_a']](output_dim=embedding_dim)
        
        if args.use_partition:
            public_model = MODEL_MAP[model_config['bottom_model_public']](output_dim=embedding_dim)
            private_model = MODEL_MAP[model_config['bottom_model_private']](output_dim=embedding_dim)
            top_model = MODEL_MAP[model_config['main_top_model']](input_dim=embedding_dim * 3, output_dim=num_classes)
        else:
            party_b_model = MODEL_MAP[model_config['bottom_model_public']](output_dim=embedding_dim)
            top_model = MODEL_MAP[model_config['shadow_top_model']](input_dim=embedding_dim * 2, output_dim=num_classes)
    
    # 将模型移到设备上
    party_a_model = party_a_model.to(device)
    top_model = top_model.to(device)
    
    if args.use_partition:
        public_model = public_model.to(device)
        private_model = private_model.to(device)
        models = [party_a_model, public_model, private_model, top_model]
    else:
        party_b_model = party_b_model.to(device)
        models = [party_a_model, party_b_model, top_model]

    # 优化器
    optimizers = [optim.Adam(model.parameters(), lr=args.lr) for model in models]
    criterion = nn.CrossEntropyLoss()

    print(f"--- 开始基线训练 (无BWL机制) (共 {args.epochs} 个周期) ---")
    
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        
        for i, (batch_X_a, batch_X_b, batch_y) in enumerate(train_loader):
            batch_X_a, batch_X_b, batch_y = batch_X_a.to(device), batch_X_b.to(device), batch_y.to(device)
            
            # 清零梯度
            for opt in optimizers: 
                opt.zero_grad()

            # 前向传播
            E_a = party_a_model(batch_X_a)
            
            if args.use_partition:
                # 使用特征划分
                if config['model_type'] == 'fcnn':
                    batch_X_b_public, batch_X_b_private = batch_X_b[:, public_indices], batch_X_b[:, private_indices]
                else:
                    batch_X_b_public, batch_X_b_private = batch_X_b[:, :, :16, :], batch_X_b[:, :, 16:, :]
                
                E_public = public_model(batch_X_b_public)
                E_private = private_model(batch_X_b_private)
                E_fused = torch.cat((E_a, E_public, E_private), dim=1)
            else:
                # 不使用特征划分
                E_b = party_b_model(batch_X_b)
                E_fused = torch.cat((E_a, E_b), dim=1)
            
            prediction = top_model(E_fused)
            
            # 只计算标准交叉熵损失（无BWL损失）
            loss = criterion(prediction, batch_y)
            
            # 反向传播和优化
            loss.backward()
            for opt in optimizers:
                opt.step()
            
            epoch_loss += loss.item()
            
            if (i + 1) % args.print_freq == 0:
                print(f'周期 [{epoch+1}/{args.epochs}], 批次 [{i+1}/{len(train_loader)}], 损失: {loss.item():.4f}')
            
        # 周期结束统计
        avg_loss = epoch_loss / len(train_loader)
        print(f'=== 周期 [{epoch+1}/{args.epochs}] 完成，平均损失: {avg_loss:.4f} ===')

    print("--- 基线训练结束 ---")
    
    if args.use_partition:
        trained_models = (party_a_model, public_model, private_model, top_model)
        split_indices = (public_indices, private_indices)
    else:
        trained_models = (party_a_model, party_b_model, top_model)
        split_indices = (None, None)
    
    test_data = (X_a_test, X_b_test, y_test)
    return trained_models, test_data, split_indices, config

# --- 测试函数 ---
def test(args, models, test_data, split_indices, config):
    """测试基线模型的性能"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("--- 开始基线测试 ---")
    
    X_a_test, X_b_test, y_test = test_data
    public_indices, private_indices = split_indices

    test_loader = create_dataloader(X_a_test, X_b_test, y_test, batch_size=args.batch_size, shuffle=False)

    # 设置评估模式
    for m in models: 
        m.eval()

    correct, total = 0, 0
    with torch.no_grad():
        for batch_X_a, batch_X_b, batch_y in test_loader:
            batch_X_a, batch_X_b, batch_y = batch_X_a.to(device), batch_X_b.to(device), batch_y.to(device)
            
            # 前向传播
            E_a = models[0](batch_X_a)  # party_a_model
            
            if args.use_partition:
                # 使用特征划分
                party_b_public_model, party_b_private_model, top_model = models[1], models[2], models[3]
                
                if config['model_type'] == 'fcnn':
                    batch_X_b_public, batch_X_b_private = batch_X_b[:, public_indices], batch_X_b[:, private_indices]
                else:
                    batch_X_b_public, batch_X_b_private = batch_X_b[:, :, :16, :], batch_X_b[:, :, 16:, :]

                E_public = party_b_public_model(batch_X_b_public)
                E_private = party_b_private_model(batch_X_b_private)
                E_fused = torch.cat((E_a, E_public, E_private), dim=1)
            else:
                # 不使用特征划分
                party_b_model, top_model = models[1], models[2]
                E_b = party_b_model(batch_X_b)
                E_fused = torch.cat((E_a, E_b), dim=1)
            
            prediction = top_model(E_fused)
            _, predicted = torch.max(prediction.data, 1)

            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

    accuracy = 100 * correct / total if total > 0 else 0
    partition_suffix = "_with_partition" if args.use_partition else "_no_partition"
    print(f'基线准确率在 {args.dataset} 测试集上: {accuracy:.2f} %')

    # 保存结果
    results_file = os.path.join('result', 'results.csv')
    fieldnames = ['algorithm', 'dataset', 'main_accuracy', 'shadow_accuracy', 'attack_accuracy']
    new_record = {
        'algorithm': f'Baseline{partition_suffix}',
        'dataset': args.dataset,
        'main_accuracy': f'{accuracy:.2f}',
        'shadow_accuracy': 'N/A',
        'attack_accuracy': 'N/A'
    }

    # 更新结果文件
    if not os.path.isfile(results_file):
        with open(results_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(new_record)
    else:
        df = pd.read_csv(results_file)
        existing_index = df[(df['algorithm'] == f'Baseline{partition_suffix}') & (df['dataset'] == args.dataset)].index
        if not existing_index.empty:
            df.loc[existing_index, list(new_record.keys())] = list(new_record.values())
        else:
            new_df = pd.DataFrame([new_record])
            df = pd.concat([df, new_df], ignore_index=True)
        df.to_csv(results_file, index=False)
        
    print(f"基线结果已更新/保存到 {results_file}")

# --- 主程序执行 ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="运行基线实验：没有任何BWL机制的标准VFL")
    parser.add_argument('--dataset', type=str, required=True, choices=['bcw', 'cifar10', 'cinic10'], help='用于训练和测试的数据集.')
    parser.add_argument('--epochs', type=int, default=10, help='训练周期数.')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率.')
    parser.add_argument('--batch_size', type=int, default=64, help='批处理大小.')
    parser.add_argument('--print_freq', type=int, default=10, help='每多少个batch输出一次训练进度.')
    parser.add_argument('--use_partition', action='store_true', help='是否使用特征划分（默认不使用）')
    parser.add_argument('--partition_method', type=str, default=None, choices=['shap', 'mutual_info', 'random'], help='特征划分方法，仅在--use_partition时有效')
    parser.add_argument('--private_ratio', type=float, default=None, help='私有特征比例，仅在--use_partition时有效')
    
    args = parser.parse_args()

    log_dir = 'result'
    os.makedirs(log_dir, exist_ok=True)
    partition_suffix = "_with_partition" if args.use_partition else "_no_partition"
    log_file_path = os.path.join(log_dir, f'baseline{partition_suffix}_{args.dataset}_results.txt')
    sys.stdout = Logger(log_file_path, sys.stdout)
    
    print(f"=== 基线实验：无BWL机制的标准VFL ===")
    print(f"特征划分: {'启用' if args.use_partition else '禁用'}")
    
    trained_models, test_data, split_indices, config = train(args)
    test(args, trained_models, test_data, split_indices, config)
