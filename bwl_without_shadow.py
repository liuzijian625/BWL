"""
BWL 消融实验版本：只有边界漫游损失，没有影子模型双轨架构
这个版本用于验证边界漫游损失的独立贡献，移除了影子模型的复杂性
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
from losses.boundary_wandering_loss import boundary_wandering_loss
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
    简化的BWL训练：只有边界漫游损失，没有影子模型架构
    在单一模型上同时应用交叉熵损失和边界漫游损失
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    with open('config.json', 'r') as f:
        config = json.load(f)[args.dataset]
    params = config['params']
    model_config = config['bwl']

    data_loader = DATA_LOADER_MAP[args.dataset]
    (X_a_train, X_b_train, y_train), (X_a_test, X_b_test, y_test) = data_loader()
    train_loader = create_dataloader(X_a_train, X_b_train, y_train, batch_size=args.batch_size)

    # 获取特征划分配置（优先使用命令行参数）
    feature_partition_config = config.get('feature_partition', {})
    partition_method = args.partition_method if args.partition_method else feature_partition_config.get('method', 'random')
    private_ratio = args.private_ratio if args.private_ratio else feature_partition_config.get('private_ratio', 0.3)
    random_state = feature_partition_config.get('random_state', 42)
    shap_model_type = feature_partition_config.get('shap_model_type', 'xgboost')
    
    print(f"消融实验配置：方法={partition_method}, 私有比例={private_ratio}, BWL权重={args.alpha}")

    if config['model_type'] == 'fcnn':
        # 使用特征划分接口
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
        public_indices, private_indices = None, None

    embedding_dim = params['embedding_dim']
    num_classes = params['num_classes']

    # 简化模型架构：只有底层模型和一个顶层模型
    if config['model_type'] == 'fcnn':
        party_a_model = MODEL_MAP[model_config['bottom_model_party_a']](input_dim=params['party_a_features'], output_dim=embedding_dim)
        public_model = MODEL_MAP[model_config['bottom_model_public']](input_dim=num_public_features, output_dim=embedding_dim)
        private_model = MODEL_MAP[model_config['bottom_model_private']](input_dim=num_private_features, output_dim=embedding_dim)
    else:
        party_a_model = MODEL_MAP[model_config['bottom_model_party_a']](output_dim=embedding_dim)
        public_model = MODEL_MAP[model_config['bottom_model_public']](output_dim=embedding_dim)
        private_model = MODEL_MAP[model_config['bottom_model_private']](output_dim=embedding_dim)

    # 使用主顶层模型作为唯一的顶层模型
    top_model = MODEL_MAP[model_config['main_top_model']](input_dim=embedding_dim * 3, output_dim=num_classes)
    
    # 将所有模型移到设备上
    party_a_model = party_a_model.to(device)
    public_model = public_model.to(device)
    private_model = private_model.to(device)
    top_model = top_model.to(device)

    # 优化器：所有模型统一优化
    optimizers = [
        optim.Adam(party_a_model.parameters(), lr=args.lr),
        optim.Adam(public_model.parameters(), lr=args.lr),
        optim.Adam(private_model.parameters(), lr=args.lr),
        optim.Adam(top_model.parameters(), lr=args.lr)
    ]
    criterion = nn.CrossEntropyLoss()

    print(f"--- 开始BWL消融实验训练 (只有边界漫游损失，无影子模型) (共 {args.epochs} 个周期) ---")
    
    for epoch in range(args.epochs):
        epoch_pred_loss = 0.0
        epoch_bw_loss = 0.0
        epoch_total_loss = 0.0
        
        for i, (batch_X_a, batch_X_b, batch_y) in enumerate(train_loader):
            batch_X_a, batch_X_b, batch_y = batch_X_a.to(device), batch_X_b.to(device), batch_y.to(device)
            
            # 清零梯度
            for opt in optimizers: 
                opt.zero_grad()

            # 特征处理
            if config['model_type'] == 'fcnn':
                batch_X_b_public, batch_X_b_private = batch_X_b[:, public_indices], batch_X_b[:, private_indices]
            else:
                batch_X_b_public, batch_X_b_private = batch_X_b[:, :, :16, :], batch_X_b[:, :, 16:, :]

            # 前向传播
            E_a = party_a_model(batch_X_a)
            E_public = public_model(batch_X_b_public)
            E_private = private_model(batch_X_b_private)
            
            # 融合所有嵌入
            E_fused = torch.cat((E_a, E_public, E_private), dim=1)
            prediction = top_model(E_fused)
            
            # 计算损失
            pred_loss = criterion(prediction, batch_y)
            bw_loss = boundary_wandering_loss(E_public, batch_y)  # 只对公开嵌入应用BWL
            
            # 总损失：预测损失 + α * 边界漫游损失
            total_loss = pred_loss + args.alpha * bw_loss
            
            # 反向传播和优化
            total_loss.backward()
            for opt in optimizers:
                opt.step()
            
            # 记录损失
            epoch_pred_loss += pred_loss.item()
            epoch_bw_loss += bw_loss.item()
            epoch_total_loss += total_loss.item()
            
            if (i + 1) % args.print_freq == 0:
                print(f'周期 [{epoch+1}/{args.epochs}], 批次 [{i+1}/{len(train_loader)}]')
                print(f'  预测损失: {pred_loss.item():.4f}, BWL损失: {bw_loss.item():.4f}, 总损失: {total_loss.item():.4f}')
            
        # 周期结束统计
        avg_pred_loss = epoch_pred_loss / len(train_loader)
        avg_bw_loss = epoch_bw_loss / len(train_loader)
        avg_total_loss = epoch_total_loss / len(train_loader)
        
        print(f'=== 周期 [{epoch+1}/{args.epochs}] 完成 ===')
        print(f'    平均预测损失: {avg_pred_loss:.4f}')
        print(f'    平均BWL损失: {avg_bw_loss:.4f}')
        print(f'    平均总损失: {avg_total_loss:.4f}')

    print("--- 消融实验训练结束 ---")
    
    models = (party_a_model, public_model, private_model, top_model)
    test_data = (X_a_test, X_b_test, y_test)
    split_indices = (public_indices, private_indices)
    return models, test_data, split_indices, config

# --- 测试函数 ---
def test(args, models, test_data, split_indices, config):
    """测试简化BWL模型的性能"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("--- 开始消融实验测试 ---")
    
    party_a_model, public_model, private_model, top_model = models
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
            
            # 特征处理
            if config['model_type'] == 'fcnn':
                batch_X_b_public, batch_X_b_private = batch_X_b[:, public_indices], batch_X_b[:, private_indices]
            else:
                batch_X_b_public, batch_X_b_private = batch_X_b[:, :, :16, :], batch_X_b[:, :, 16:, :]

            # 前向传播
            E_a = party_a_model(batch_X_a)
            E_public = public_model(batch_X_b_public)
            E_private = private_model(batch_X_b_private)

            # 融合和预测
            E_fused = torch.cat((E_a, E_public, E_private), dim=1)
            prediction = top_model(E_fused)
            _, predicted = torch.max(prediction.data, 1)

            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

    accuracy = 100 * correct / total if total > 0 else 0
    print(f'BWL消融实验准确率在 {args.dataset} 测试集上: {accuracy:.2f} %')

    # 保存结果
    results_file = os.path.join('result', 'results.csv')
    fieldnames = ['algorithm', 'dataset', 'main_accuracy', 'shadow_accuracy', 'attack_accuracy']
    new_record = {
        'algorithm': 'BWL_No_Shadow',
        'dataset': args.dataset,
        'main_accuracy': f'{accuracy:.2f}',
        'shadow_accuracy': 'N/A',  # 没有影子模型
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
        existing_index = df[(df['algorithm'] == 'BWL_No_Shadow') & (df['dataset'] == args.dataset)].index
        if not existing_index.empty:
            df.loc[existing_index, list(new_record.keys())] = list(new_record.values())
        else:
            new_df = pd.DataFrame([new_record])
            df = pd.concat([df, new_df], ignore_index=True)
        df.to_csv(results_file, index=False)
        
    print(f"消融实验结果已更新/保存到 {results_file}")

# --- 主程序执行 ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="运行BWL消融实验：只有边界漫游损失，没有影子模型架构")
    parser.add_argument('--dataset', type=str, required=True, choices=['bcw', 'cifar10', 'cinic10'], help='用于训练和测试的数据集.')
    parser.add_argument('--epochs', type=int, default=10, help='训练周期数.')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率.')
    parser.add_argument('--batch_size', type=int, default=64, help='批处理大小.')
    parser.add_argument('--alpha', type=float, default=1, help='边界徘徊损失的权重.')
    parser.add_argument('--print_freq', type=int, default=10, help='每多少个batch输出一次训练进度.')
    parser.add_argument('--partition_method', type=str, default=None, choices=['shap', 'mutual_info', 'random'], help='特征划分方法，可选: shap, mutual_info, random. 如不指定则使用配置文件设置.')
    parser.add_argument('--private_ratio', type=float, default=None, help='私有特征比例，如不指定则使用配置文件设置.')
    
    args = parser.parse_args()

    log_dir = 'result'
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f'bwl_no_shadow_{args.dataset}_results.txt')
    sys.stdout = Logger(log_file_path, sys.stdout)
    
    print(f"=== BWL消融实验：只有边界漫游损失，α={args.alpha} ===")
    
    trained_models, test_data, split_indices, config = train(args)
    test(args, trained_models, test_data, split_indices, config)
