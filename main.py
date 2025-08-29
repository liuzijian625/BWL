
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
    
    print(f"特征划分配置：方法={partition_method}, 私有比例={private_ratio}")

    if config['model_type'] == 'fcnn':
        # 使用新的特征划分接口
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
        print(f"公开特征索引：{public_indices[:10]}..." if len(public_indices) > 10 else f"公开特征索引：{public_indices}")
        print(f"私有特征索引：{private_indices[:10]}..." if len(private_indices) > 10 else f"私有特征索引：{private_indices}")
    else:
        public_indices, private_indices = None, None

    embedding_dim = params['embedding_dim']
    num_classes = params['num_classes']

    if config['model_type'] == 'fcnn':
        party_a_model = MODEL_MAP[model_config['bottom_model_party_a']](input_dim=params['party_a_features'], output_dim=embedding_dim)
        public_model = MODEL_MAP[model_config['bottom_model_public']](input_dim=num_public_features, output_dim=embedding_dim)
        private_model = MODEL_MAP[model_config['bottom_model_private']](input_dim=num_private_features, output_dim=embedding_dim)
    else:
        party_a_model = MODEL_MAP[model_config['bottom_model_party_a']](output_dim=embedding_dim)
        public_model = MODEL_MAP[model_config['bottom_model_public']](output_dim=embedding_dim)
        private_model = MODEL_MAP[model_config['bottom_model_private']](output_dim=embedding_dim)

    shadow_top_model = MODEL_MAP[model_config['shadow_top_model']](input_dim=embedding_dim * 2, output_dim=num_classes)
    main_top_model = MODEL_MAP[model_config['main_top_model']](input_dim=embedding_dim * 3, output_dim=num_classes)
    
    party_a_model, public_model, private_model, shadow_top_model, main_top_model =party_a_model.to(device), public_model.to(device), private_model.to(device), shadow_top_model.to(device), main_top_model.to(device)

    optimizers = [
        optim.Adam(party_a_model.parameters(), lr=args.lr),
        optim.Adam(public_model.parameters(), lr=args.lr),
        optim.Adam(private_model.parameters(), lr=args.lr),
        optim.Adam(shadow_top_model.parameters(), lr=args.lr),
        optim.Adam(main_top_model.parameters(), lr=args.lr)
    ]
    criterion = nn.CrossEntropyLoss()

    print(f"--- 开始在 {args.dataset} 数据集上进行BWL训练 (共 {args.epochs} 个周期) ---")
    
    for epoch in range(args.epochs):
        for i, (batch_X_a, batch_X_b, batch_y) in enumerate(train_loader):
            batch_X_a, batch_X_b, batch_y = batch_X_a.to(device), batch_X_b.to(device), batch_y.to(device)
            
            for opt in optimizers: opt.zero_grad()

            if config['model_type'] == 'fcnn':
                batch_X_b_public, batch_X_b_private = batch_X_b[:, public_indices], batch_X_b[:, private_indices]
            else:
                batch_X_b_public, batch_X_b_private = batch_X_b[:, :, :16, :], batch_X_b[:, :, 16:, :]

            E_a = party_a_model(batch_X_a)
            E_public = public_model(batch_X_b_public)
            E_private = private_model(batch_X_b_private)
            
            E_fused_shadow = torch.cat((E_a, E_public), dim=1)
            shadow_prediction = shadow_top_model(E_fused_shadow)
            pred_loss = criterion(shadow_prediction, batch_y)
            bw_loss = boundary_wandering_loss(E_public, batch_y)
            shadow_loss = pred_loss + args.alpha * bw_loss
            shadow_loss.backward(retain_graph=True)
            
            optimizers[0].step()
            optimizers[1].step()
            optimizers[3].step()

            for p in list(private_model.parameters()) + list(main_top_model.parameters()): p.grad = None

            E_fused_main = torch.cat((E_a.detach(), E_public.detach(), E_private), dim=1)
            main_prediction = main_top_model(E_fused_main)
            main_loss = criterion(main_prediction, batch_y)
            main_loss.backward()
            optimizers[2].step()
            optimizers[4].step()
            
        print(f'=== 周期 [{epoch+1}/{args.epochs}] 完成 ===')

    print("--- 训练结束 ---")
    
    models = (party_a_model, public_model, private_model, shadow_top_model, main_top_model)
    test_data = (X_a_test, X_b_test, y_test)
    split_indices = (public_indices, private_indices)
    return models, test_data, split_indices, config

# --- 测试函数 ---
def test(args, models, test_data, split_indices, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("--- 开始测试 ---")
    party_a_model, public_model, private_model, shadow_top_model, main_top_model = models
    X_a_test, X_b_test, y_test = test_data
    public_indices, private_indices = split_indices

    test_loader = create_dataloader(X_a_test, X_b_test, y_test, batch_size=args.batch_size, shuffle=False)

    for m in models: m.eval()

    correct_shadow, correct_main, total = 0, 0, 0
    with torch.no_grad():
        for batch_X_a, batch_X_b, batch_y in test_loader:
            batch_X_a, batch_X_b, batch_y = batch_X_a.to(device), batch_X_b.to(device), batch_y.to(device)
            
            if config['model_type'] == 'fcnn':
                batch_X_b_public, batch_X_b_private = batch_X_b[:, public_indices], batch_X_b[:, private_indices]
            else:
                batch_X_b_public, batch_X_b_private = batch_X_b[:, :, :16, :], batch_X_b[:, :, 16:, :]

            E_a = party_a_model(batch_X_a)
            E_public = public_model(batch_X_b_public)
            E_private = private_model(batch_X_b_private)

            E_fused_shadow = torch.cat((E_a, E_public), dim=1)
            shadow_prediction = shadow_top_model(E_fused_shadow)
            _, predicted_shadow = torch.max(shadow_prediction.data, 1)
            
            E_fused_main = torch.cat((E_a, E_public, E_private), dim=1)
            main_prediction = main_top_model(E_fused_main)
            _, predicted_main = torch.max(main_prediction.data, 1)

            total += batch_y.size(0)
            correct_shadow += (predicted_shadow == batch_y).sum().item()
            correct_main += (predicted_main == batch_y).sum().item()

    shadow_acc_val = 100 * correct_shadow / total if total > 0 else 0
    main_acc_val = 100 * correct_main / total if total > 0 else 0
    print(f'BWL 影子准确率 (Shadow Accuracy) 在 {args.dataset} 测试集上: {shadow_acc_val:.2f} %')
    print(f'BWL 真实准确率 (Main Accuracy) 在 {args.dataset} 测试集上: {main_acc_val:.2f} %')

    # 保存或更新结果
    results_file = os.path.join('result', 'results.csv')
    fieldnames = ['algorithm', 'dataset', 'main_accuracy', 'shadow_accuracy', 'attack_accuracy']
    new_record = {
        'algorithm': 'BWL',
        'dataset': args.dataset,
        'main_accuracy': f'{main_acc_val:.2f}',
        'shadow_accuracy': f'{shadow_acc_val:.2f}',
        'attack_accuracy': 'N/A'
    }

    if not os.path.isfile(results_file):
        with open(results_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(new_record)
    else:
        df = pd.read_csv(results_file)
        existing_index = df[(df['algorithm'] == 'BWL') & (df['dataset'] == args.dataset)].index
        if not existing_index.empty:
            df.loc[existing_index, list(new_record.keys())] = list(new_record.values())
        else:
            new_df = pd.DataFrame([new_record])
            df = pd.concat([df, new_df], ignore_index=True)
        df.to_csv(results_file, index=False)
        
    print(f"结果已更新/保存到 {results_file}")

# --- 主程序执行 ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="运行BWL算法，训练并立即测试.")
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
    log_file_path = os.path.join(log_dir, f'bwl_{args.dataset}_results.txt')
    sys.stdout = Logger(log_file_path, sys.stdout)
    
    trained_models, test_data, split_indices, config = train(args)
    test(args, trained_models, test_data, split_indices, config)
