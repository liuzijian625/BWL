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
from sklearn.model_selection import train_test_split

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
    "FCNN_Top_3": arch.FCNN_Top_3,
    "FCNN_Top_4": arch.FCNN_Top_4,
}

DATA_LOADER_MAP = {
    "bcw": load_bcw,
    "cifar10": load_cifar10,
    "cinic10": load_cinic10
}

# ============================== Phase 1: Standard VFL Training ==============================
def train_vfl(args, X_a_vfl, X_b_vfl, y_vfl):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Phase 1: 在VFL数据上进行标准VFL训练 ---")
    
    with open('config.json', 'r') as f:
        config = json.load(f)[args.dataset]
    params = config['params']
    model_config = config['vanilla']

    vfl_loader = create_dataloader(X_a_vfl, X_b_vfl, y_vfl, batch_size=args.batch_size)

    embedding_dim = params['embedding_dim']
    num_classes = params['num_classes']

    if config['model_type'] == 'fcnn':
        party_a_model = MODEL_MAP[model_config['bottom_model_party_a']](input_dim=params['party_a_features'], output_dim=embedding_dim)
        party_b_model = MODEL_MAP[model_config['bottom_model_party_b']](input_dim=params['party_b_features'], output_dim=embedding_dim)
    else: # resnet
        party_a_model = MODEL_MAP[model_config['bottom_model_party_a']](output_dim=embedding_dim)
        party_b_model = MODEL_MAP[model_config['bottom_model_party_b']](output_dim=embedding_dim)

    top_model = MODEL_MAP[model_config['top_model']](input_dim=embedding_dim * 2, output_dim=num_classes)
    
    party_a_model, party_b_model, top_model = party_a_model.to(device), party_b_model.to(device), top_model.to(device)

    optimizers = [optim.Adam(m.parameters(), lr=args.lr) for m in [party_a_model, party_b_model, top_model]]
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        for i, (batch_X_a, batch_X_b, batch_y) in enumerate(vfl_loader):
            batch_X_a, batch_X_b, batch_y = batch_X_a.to(device), batch_X_b.to(device), batch_y.to(device)
            
            for opt in optimizers: opt.zero_grad()

            E_a = party_a_model(batch_X_a)
            E_b = party_b_model(batch_X_b)
            E_fused = torch.cat((E_a, E_b), dim=1)
            prediction = top_model(E_fused)
            
            loss = criterion(prediction, batch_y)
            loss.backward()
            for opt in optimizers: opt.step()
            
        print(f'VFL训练周期 [{epoch+1}/{args.epochs}] 完成, 损失: {loss.item():.4f}')

    print("--- VFL训练结束 ---")
    return party_a_model.to('cpu'), party_b_model, top_model

# ============================== Main Task Performance Evaluation ==============================
def test_vfl(args, vfl_models, X_a_test, X_b_test, y_test):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("--- 评估VFL主任务性能 ---")
    party_a_model, party_b_model, top_model = vfl_models
    
    party_a_model = party_a_model.to(device)
    party_a_model.eval()
    party_b_model.eval()
    top_model.eval()

    test_loader = create_dataloader(X_a_test, X_b_test, y_test, batch_size=args.batch_size, shuffle=False)

    correct, total = 0, 0
    with torch.no_grad():
        for batch_X_a, batch_X_b, batch_y in test_loader:
            batch_X_a, batch_X_b, batch_y = batch_X_a.to(device), batch_X_b.to(device), batch_y.to(device)
            E_a = party_a_model(batch_X_a)
            E_b = party_b_model(batch_X_b)
            E_fused = torch.cat((E_a, E_b), dim=1)
            prediction = top_model(E_fused)
            _, predicted = torch.max(prediction.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

    main_accuracy = 100 * correct / total if total > 0 else 0
    print(f'VFL Main Accuracy on {args.dataset} test set: {main_accuracy:.2f} %')
    return main_accuracy

# ============================== Phase 2 & 3: Attack Model Training ==============================
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
    print(f"--- Phase 2 & 3: 在辅助数据上训练攻击模型 ---")

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

# ============================== Phase 4: Inference ==============================
def perform_inference(args, attack_model, X_a_test, y_test):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Phase 4: 在测试集上进行标签推断 ---")
    
    attack_model = attack_model.to(device)
    attack_model.eval()

    test_loader = create_dataloader(X_a_test, torch.zeros(len(X_a_test), 0), y_test, batch_size=args.batch_size, shuffle=False)
    
    correct, total = 0, 0
    with torch.no_grad():
        for batch_X_a, _, batch_y in test_loader:
            batch_X_a, batch_y = batch_X_a.to(device), batch_y.to(device)
            prediction = attack_model(batch_X_a)
            _, predicted = torch.max(prediction.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

    attack_accuracy = 100 * correct / total if total > 0 else 0
    print(f'Passive LIA Attack Accuracy on {args.dataset} test set: {attack_accuracy:.2f} %')
    return attack_accuracy

# ============================== Save Results ==============================
def save_results(args, main_accuracy, attack_accuracy):
    results_file = os.path.join('result', 'results.csv')
    fieldnames = ['algorithm', 'dataset', 'main_accuracy', 'shadow_accuracy', 'attack_accuracy']
    new_record = {
        'algorithm': 'Passive_LIA',
        'dataset': args.dataset,
        'main_accuracy': f'{main_accuracy:.2f}',
        'shadow_accuracy': 'N/A',
        'attack_accuracy': f'{attack_accuracy:.2f}'
    }

    if not os.path.isfile(results_file):
        with open(results_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(new_record)
    else:
        df = pd.read_csv(results_file)
        existing_index = df[(df['algorithm'] == 'Passive_LIA') & (df['dataset'] == args.dataset)].index
        if not existing_index.empty:
            df.loc[existing_index, list(new_record.keys())] = list(new_record.values())
        else:
            new_df = pd.DataFrame([new_record])
            df = pd.concat([df, new_df], ignore_index=True)
        df.to_csv(results_file, index=False)
        
    print(f"结果已更新/保存到 {results_file}")

# ============================== Main Execution ==============================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Passive Label Inference Attack on Vanilla VFL.")
    parser.add_argument('--dataset', type=str, required=True, choices=['bcw', 'cifar10', 'cinic10'], help='Dataset for the attack.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of VFL training epochs.')
    parser.add_argument('--attack_epochs', type=int, default=10, help='Number of attack model training epochs.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--aux_data_ratio', type=float, default=0.1, help='Ratio of training data to use as auxiliary data for the attacker.')
    
    args = parser.parse_args()

    log_dir = 'result'
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f'passive_lia_{args.dataset}_results.txt')
    sys.stdout = Logger(log_file_path, sys.stdout)

    print(f"========== 开始在 {args.dataset} 上进行被动LIA攻击 ==========")
    
    data_loader = DATA_LOADER_MAP[args.dataset]
    (X_a_train_full, X_b_train_full, y_train_full), (X_a_test, X_b_test, y_test) = data_loader()

    indices = np.arange(len(X_a_train_full))
    train_indices, aux_indices = train_test_split(indices, test_size=args.aux_data_ratio, random_state=42)
    
    X_a_vfl, y_vfl = X_a_train_full[train_indices], y_train_full[train_indices]
    X_b_vfl = X_b_train_full[train_indices]
    X_a_aux, y_aux = X_a_train_full[aux_indices], y_train_full[aux_indices]

    print(f"数据切分完成: VFL训练数据 {len(y_vfl)} 条, 攻击者辅助数据 {len(y_aux)} 条。")

    party_a_model_cpu, party_b_model, top_model = train_vfl(args, X_a_vfl, X_b_vfl, y_vfl)

    main_accuracy = test_vfl(args, (party_a_model_cpu, party_b_model, top_model), X_a_test, X_b_test, y_test)

    attack_model = train_attack_model(args, party_a_model_cpu, X_a_aux, y_aux)

    attack_accuracy = perform_inference(args, attack_model, X_a_test, y_test)

    save_results(args, main_accuracy, attack_accuracy)

    print(f"========== 被动LIA攻击流程结束 ==========")