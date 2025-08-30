
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
from sklearn.model_selection import train_test_split

import models.architectures as arch
from losses.boundary_wandering_loss import boundary_wandering_loss
from utils.data_loader import load_bcw, load_cifar10, load_cinic10, create_dataloader
from utils.feature_partition import partition_features
from utils.attack_metrics import evaluate_attack_model, print_attack_metrics
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

# ============================== Malicious Optimizer (from active_lia.py) ==============================
class MaliciousOptimizer:
    def __init__(self, optimizer, eta=0.2, gamma=0.1):
        self.optimizer = optimizer
        self.eta = eta
        self.gamma = gamma
        self.velocities = [torch.zeros_like(p.data) for group in self.optimizer.param_groups for p in group['params']]
        self.scaling_factors = [torch.ones_like(p.data) for group in self.optimizer.param_groups for p in group['params']]

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        param_idx = 0
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad.data
                velocity = self.velocities[param_idx]
                scaling_factor = self.scaling_factors[param_idx]
                sign_match = torch.sign(grad) == torch.sign(velocity)
                scaling_factor[sign_match] *= (1 + self.eta)
                scaling_factor[~sign_match] *= (1 - self.gamma)
                scaling_factor.clamp_(0.1, 10.0)
                p.grad.data.mul_(scaling_factor)
                self.velocities[param_idx] = grad
                self.scaling_factors[param_idx] = scaling_factor
                param_idx += 1
        self.optimizer.step()

# ============================== Phase 1: BWL VFL Training with Active Attacker ==============================
def train_bwl_active(args, X_a_train, X_b_train, y_train):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Phase 1: 在VFL数据上进行带主动攻击的BWL训练 ---")
    
    with open('config.json', 'r') as f:
        config = json.load(f)[args.dataset]
    params = config['params']
    model_config = config['bwl']

    train_loader = create_dataloader(X_a_train, X_b_train, y_train, batch_size=args.batch_size)

    # 获取特征划分配置
    feature_partition_config = config.get('feature_partition', {})
    partition_method = feature_partition_config.get('method', 'random')
    private_ratio = feature_partition_config.get('private_ratio', 0.3)
    random_state = feature_partition_config.get('random_state', 42)
    shap_model_type = feature_partition_config.get('shap_model_type', 'xgboost')

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
    
    models_on_device = [m.to(device) for m in [party_a_model, public_model, private_model, shadow_top_model, main_top_model]]
    party_a_model, public_model, private_model, shadow_top_model, main_top_model = models_on_device

    # 使用恶意优化器攻击Party A
    optimizer_a = MaliciousOptimizer(optim.Adam(party_a_model.parameters(), lr=args.lr))
    optimizers = [
        optimizer_a,
        optim.Adam(public_model.parameters(), lr=args.lr),
        optim.Adam(private_model.parameters(), lr=args.lr),
        optim.Adam(shadow_top_model.parameters(), lr=args.lr),
        optim.Adam(main_top_model.parameters(), lr=args.lr)
    ]
    criterion = nn.CrossEntropyLoss()

    print(f"--- 开始在 {args.dataset} 数据集上进行BWL(主动攻击)训练 (共 {args.epochs} 个周期) ---")
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
            
            optimizers[0].step() # Malicious optimizer for Party A
            optimizers[1].step()
            optimizers[3].step()

            for p in list(private_model.parameters()) + list(main_top_model.parameters()): p.grad = None

            E_fused_main = torch.cat((E_a.detach(), E_public.detach(), E_private), dim=1)
            main_prediction = main_top_model(E_fused_main)
            main_loss = criterion(main_prediction, batch_y)
            main_loss.backward()
            optimizers[2].step()
            optimizers[4].step()
        print(f'=== BWL(主动攻击)训练周期 [{epoch+1}/{args.epochs}] 完成 ===')

    print("--- BWL(主动攻击)训练结束 ---")
    return models_on_device, (public_indices, private_indices), config

# ============================== BWL Performance Evaluation ==============================
def test_bwl(args, models, test_data, split_indices, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("--- 评估BWL模型性能 ---")
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

    shadow_acc = 100 * correct_shadow / total if total > 0 else 0
    main_acc = 100 * correct_main / total if total > 0 else 0
    print(f'BWL 影子准确率 (Shadow Accuracy) 在 {args.dataset} 测试集上: {shadow_acc:.2f} %')
    print(f'BWL 真实准确率 (Main Accuracy) 在 {args.dataset} 测试集上: {main_acc:.2f} %')
    return main_acc, shadow_acc

# ============================== Attack Logic (from active_lia.py) ==============================
class AttackModel(nn.Module):
    def __init__(self, feature_extractor, num_classes, embedding_dim):
        super(AttackModel, self).__init__()
        self.feature_extractor = feature_extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.classifier_head = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        return self.classifier_head(self.feature_extractor(x))

def train_attack_model(args, party_a_model, X_a_aux, y_aux):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Phase 2: 在辅助数据上训练攻击模型 ---")
    with open('config.json', 'r') as f:
        config = json.load(f)[args.dataset]
    params = config['params']
    aux_loader = create_dataloader(X_a_aux, torch.zeros(len(X_a_aux), 0), y_aux, batch_size=args.batch_size)
    attack_model = AttackModel(
        feature_extractor=party_a_model.to(device),
        num_classes=params['num_classes'],
        embedding_dim=params['embedding_dim']
    ).to(device)
    optimizer = optim.Adam(attack_model.classifier_head.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(args.attack_epochs):
        for batch_X_a, _, batch_y in aux_loader:
            batch_X_a, batch_y = batch_X_a.to(device), batch_y.to(device)
            optimizer.zero_grad()
            prediction = attack_model(batch_X_a)
            loss = criterion(prediction, batch_y)
            loss.backward()
            optimizer.step()
        print(f'攻击模型训练周期 [{epoch+1}/{args.attack_epochs}] 完成, 损失: {loss.item():.4f}')
    print("--- 攻击模型训练结束 ---")
    return attack_model.to('cpu')

def perform_inference(args, attack_model, X_a_test, y_test):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Phase 3: 在测试集上进行标签推断 ---")
    
    # 准备测试数据加载器
    test_loader = create_dataloader(X_a_test, torch.zeros(len(X_a_test), 0), y_test, batch_size=args.batch_size, shuffle=False)
    
    # 获取类别数量
    with open('config.json', 'r') as f:
        config = json.load(f)[args.dataset]
    num_classes = config['params']['num_classes']
    
    # 使用新的评估指标评估攻击模型
    attack_metrics = evaluate_attack_model(attack_model, test_loader, device, num_classes)
    
    # 打印攻击评估指标
    print_attack_metrics(attack_metrics, 'Active_LIA_on_BWL', args.dataset)
    
    return attack_metrics

# ============================== Save Results ==============================
# 注意：现在只保存攻击指标，不再保存主任务准确率
# 使用utils/results_saver.py中的save_attack_results函数

# ============================== Main Execution ==============================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Active Label Inference Attack on BWL-defended VFL.")
    parser.add_argument('--dataset', type=str, required=True, choices=['bcw', 'cifar10', 'cinic10'], help='Dataset for the attack.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of VFL training epochs.')
    parser.add_argument('--attack_epochs', type=int, default=10, help='Number of attack model training epochs.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--alpha', type=float, default=1, help='边界徘徊损失的权重.')
    parser.add_argument('--aux_data_ratio', type=float, default=0.1, help='Ratio of training data to use as auxiliary data.')
    
    args = parser.parse_args()

    log_dir = 'result'
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f'active_lia_on_bwl_{args.dataset}_results.txt')
    sys.stdout = Logger(log_file_path, sys.stdout)

    print(f"========== 开始在 {args.dataset} 上进行 Active LIA on BWL ==========")
    
    data_loader = DATA_LOADER_MAP[args.dataset]
    (X_a_train_full, X_b_train_full, y_train_full), (X_a_test, X_b_test, y_test) = data_loader()

    indices = np.arange(len(X_a_train_full))
    train_indices, aux_indices = train_test_split(indices, test_size=args.aux_data_ratio, random_state=42)
    
    X_a_vfl, y_vfl = X_a_train_full[train_indices], y_train_full[train_indices]
    X_b_vfl = X_b_train_full[train_indices]
    X_a_aux, y_aux = X_a_train_full[aux_indices], y_train_full[aux_indices]

    print(f"数据切分完成: VFL训练数据 {len(y_vfl)} 条, 攻击者辅助数据 {len(y_aux)} 条。")

    # 1. 训练BWL模型
    bwl_models, split_indices, config = train_bwl_active(args, X_a_vfl, X_b_vfl, y_vfl)
    
    # 2. 提取攻击者模型并进行攻击（不再评估主任务性能）
    party_a_model_cpu = bwl_models[0].to('cpu')
    attack_model = train_attack_model(args, party_a_model_cpu, X_a_aux, y_aux)
    attack_metrics = perform_inference(args, attack_model, X_a_test, y_test)

    # 3. 保存攻击评估结果
    save_attack_results('Active_LIA_on_BWL', args.dataset, attack_metrics)

    print(f"========== Active LIA on BWL 流程结束 ==========")
