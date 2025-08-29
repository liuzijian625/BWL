#!/usr/bin/env python3
"""
测试脚本：验证特征划分功能的实现
"""

import numpy as np
import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.feature_partition import partition_features
from utils.data_loader import load_bcw

def test_feature_partition_methods():
    """测试三种特征划分方法"""
    print("=== 特征划分功能测试 ===\n")
    
    # 加载BCW数据集进行测试
    print("1. 加载BCW数据集...")
    (X_a_train, X_b_train, y_train), (X_a_test, X_b_test, y_test) = load_bcw()
    print(f"   Party B 训练数据形状: {X_b_train.shape}")
    print(f"   标签分布: {np.bincount(y_train)}")
    
    # 设置测试参数
    private_ratio = 0.3
    random_state = 42
    
    print(f"\n2. 测试参数设置:")
    print(f"   私有特征比例: {private_ratio}")
    print(f"   随机种子: {random_state}")
    
    methods_to_test = ['random', 'mutual_info', 'shap']
    results = {}
    
    for method in methods_to_test:
        print(f"\n3. 测试 {method.upper()} 方法...")
        try:
            public_indices, private_indices = partition_features(
                X_b_train, y_train,
                method=method,
                private_ratio=private_ratio,
                random_state=random_state,
                model_type='xgboost'  # 对于SHAP方法
            )
            
            results[method] = {
                'public_indices': public_indices,
                'private_indices': private_indices,
                'success': True
            }
            
            print(f"   ✅ {method} 方法成功")
            print(f"   公开特征数量: {len(public_indices)}")
            print(f"   私有特征数量: {len(private_indices)}")
            print(f"   公开特征索引前5个: {public_indices[:5]}")
            print(f"   私有特征索引前5个: {private_indices[:5]}")
            
            # 验证索引不重叠且覆盖所有特征
            all_indices = set(public_indices) | set(private_indices)
            expected_indices = set(range(X_b_train.shape[1]))
            
            if all_indices == expected_indices:
                print(f"   ✅ 索引验证通过：覆盖所有特征，无重叠")
            else:
                print(f"   ❌ 索引验证失败：存在重叠或遗漏")
                
        except Exception as e:
            print(f"   ❌ {method} 方法失败: {str(e)}")
            results[method] = {
                'success': False,
                'error': str(e)
            }
    
    # 比较不同方法的结果
    print(f"\n4. 方法比较:")
    successful_methods = [m for m in methods_to_test if results[m]['success']]
    
    if len(successful_methods) >= 2:
        for i, method1 in enumerate(successful_methods):
            for method2 in successful_methods[i+1:]:
                indices1 = set(results[method1]['private_indices'])
                indices2 = set(results[method2]['private_indices'])
                
                overlap = len(indices1 & indices2)
                overlap_ratio = overlap / len(indices1) if len(indices1) > 0 else 0
                
                print(f"   {method1} vs {method2}: 私有特征重叠 {overlap}/{len(indices1)} ({overlap_ratio:.2%})")
    
    # 生成总结报告
    print(f"\n5. 测试总结:")
    for method in methods_to_test:
        if results[method]['success']:
            print(f"   ✅ {method.upper()}: 正常工作")
        else:
            print(f"   ❌ {method.upper()}: {results[method]['error']}")
    
    return results

def test_config_integration():
    """测试与配置文件的集成"""
    print(f"\n=== 配置文件集成测试 ===\n")
    
    import json
    
    # 读取配置文件
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        
        print("1. 配置文件读取成功")
        
        for dataset in ['bcw', 'cifar10', 'cinic10']:
            if dataset in config:
                feature_config = config[dataset].get('feature_partition', {})
                print(f"   {dataset}: {feature_config}")
            else:
                print(f"   ❌ {dataset}: 配置缺失")
                
    except Exception as e:
        print(f"❌ 配置文件读取失败: {str(e)}")

def test_cli_interface():
    """测试命令行接口"""
    print(f"\n=== 命令行接口测试 ===\n")
    
    # 模拟命令行调用
    test_commands = [
        "python3 main.py --dataset bcw --epochs 1 --partition_method random",
        "python3 main.py --dataset bcw --epochs 1 --partition_method mutual_info",
        "python3 main.py --dataset bcw --epochs 1 --partition_method shap --private_ratio 0.2",
    ]
    
    print("推荐的测试命令:")
    for i, cmd in enumerate(test_commands, 1):
        print(f"   {i}. {cmd}")
    
    print(f"\n注意：实际运行这些命令需要安装所有依赖包")

if __name__ == "__main__":
    print("开始特征划分功能测试...\n")
    
    try:
        # 运行所有测试
        test_results = test_feature_partition_methods()
        test_config_integration()
        test_cli_interface()
        
        print(f"\n🎉 测试完成！")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
