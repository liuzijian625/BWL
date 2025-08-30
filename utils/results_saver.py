"""
结果保存工具模块
用于统一保存攻击评估结果到CSV文件
"""

import os
import csv
import pandas as pd
from .attack_metrics import format_metrics_for_csv

def save_attack_results(algorithm_name, dataset_name, attack_metrics, results_file='result/results.csv'):
    """
    保存攻击评估结果到CSV文件
    
    Args:
        algorithm_name (str): 算法名称
        dataset_name (str): 数据集名称  
        attack_metrics (dict): 攻击评估指标字典
        results_file (str): 结果文件路径
    """
    # 确保结果目录存在
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    # 格式化攻击指标
    formatted_metrics = format_metrics_for_csv(attack_metrics)
    
    # 创建新记录
    fieldnames = ['algorithm', 'dataset', 'attack_acc', 'attack_top1', 'attack_top5', 'attack_f1']
    new_record = {
        'algorithm': algorithm_name,
        'dataset': dataset_name,
        **formatted_metrics
    }
    
    # 保存或更新结果
    if not os.path.isfile(results_file):
        # 创建新文件
        with open(results_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(new_record)
    else:
        # 更新现有文件
        df = pd.read_csv(results_file)
        existing_index = df[(df['algorithm'] == algorithm_name) & (df['dataset'] == dataset_name)].index
        
        if not existing_index.empty:
            # 更新现有记录
            df.loc[existing_index, list(new_record.keys())] = list(new_record.values())
        else:
            # 添加新记录
            new_df = pd.DataFrame([new_record])
            df = pd.concat([df, new_df], ignore_index=True)
        
        df.to_csv(results_file, index=False)
    
    print(f"攻击评估结果已保存到 {results_file}")
    print(f"算法: {algorithm_name}, 数据集: {dataset_name}")
    print(f"攻击准确率: {formatted_metrics['attack_acc']}%")

def load_attack_results(results_file='result/results.csv'):
    """
    加载攻击评估结果
    
    Args:
        results_file (str): 结果文件路径
        
    Returns:
        pd.DataFrame: 结果数据框
    """
    if os.path.exists(results_file):
        return pd.read_csv(results_file)
    else:
        # 返回空的DataFrame，包含正确的列名
        return pd.DataFrame(columns=['algorithm', 'dataset', 'attack_acc', 'attack_top1', 'attack_top5', 'attack_f1'])

def print_results_summary(results_file='result/results.csv'):
    """
    打印结果摘要
    
    Args:
        results_file (str): 结果文件路径
    """
    df = load_attack_results(results_file)
    
    if df.empty:
        print("没有找到任何结果数据")
        return
    
    print("\n=== 攻击评估结果摘要 ===")
    print(df.to_string(index=False))
    print("=" * 60)
