#!/usr/bin/env python3
"""
测试新的攻击评估指标系统
"""

import torch
import numpy as np
import sys
import os

# 添加当前目录到路径
sys.path.append('.')

from utils.attack_metrics import calculate_attack_metrics, print_attack_metrics
from utils.results_saver import save_attack_results, load_attack_results, print_results_summary

def test_attack_metrics():
    """测试攻击评估指标计算"""
    print("=== 测试攻击评估指标计算 ===")
    
    # 创建模拟数据
    batch_size = 100
    num_classes = 3
    
    # 模拟预测概率 (logits)
    predictions = torch.randn(batch_size, num_classes)
    # 添加一些噪声，使得预测不是完全随机的
    predictions[:50, 0] += 2.0  # 前50个样本更可能是类别0
    predictions[50:80, 1] += 2.0  # 接下来30个样本更可能是类别1  
    predictions[80:, 2] += 2.0  # 最后20个样本更可能是类别2
    
    # 创建对应的真实标签
    targets = torch.zeros(batch_size, dtype=torch.long)
    targets[:50] = 0
    targets[50:80] = 1
    targets[80:] = 2
    
    # 计算攻击指标
    metrics = calculate_attack_metrics(predictions, targets, num_classes)
    
    # 打印结果
    print_attack_metrics(metrics, 'Test_Attack', 'test_dataset')
    
    # 验证指标的合理性
    assert 0 <= metrics['acc'] <= 100, "准确率应该在0-100之间"
    assert 0 <= metrics['top1_acc'] <= 100, "Top-1准确率应该在0-100之间"
    assert 0 <= metrics['top5_acc'] <= 100, "Top-5准确率应该在0-100之间"
    assert 0 <= metrics['f1_score'] <= 100, "F1分数应该在0-100之间"
    assert metrics['top5_acc'] >= metrics['top1_acc'], "Top-5准确率应该大于等于Top-1准确率"
    
    print("✅ 攻击评估指标计算测试通过!")
    return metrics

def test_results_saver():
    """测试结果保存和加载"""
    print("\n=== 测试结果保存和加载 ===")
    
    # 创建测试指标
    test_metrics = {
        'acc': 85.5,
        'top1_acc': 85.5,
        'top5_acc': 95.0,
        'f1_score': 82.3
    }
    
    # 保存测试结果
    test_results_file = 'result/test_results.csv'
    save_attack_results('Test_Algorithm', 'test_dataset', test_metrics, test_results_file)
    
    # 加载并验证结果
    df = load_attack_results(test_results_file)
    assert len(df) > 0, "应该能加载到结果数据"
    
    # 验证保存的数据
    test_row = df[(df['algorithm'] == 'Test_Algorithm') & (df['dataset'] == 'test_dataset')]
    assert len(test_row) == 1, "应该找到一条测试记录"
    
    saved_acc = float(test_row['attack_acc'].values[0])
    assert abs(saved_acc - test_metrics['acc']) < 0.1, "保存的准确率应该与原始值接近"
    
    print("✅ 结果保存和加载测试通过!")
    
    # 清理测试文件
    if os.path.exists(test_results_file):
        os.remove(test_results_file)

def test_edge_cases():
    """测试边界情况"""
    print("\n=== 测试边界情况 ===")
    
    # 测试完美预测
    perfect_predictions = torch.tensor([[10.0, 0.0], [0.0, 10.0], [10.0, 0.0]])
    perfect_targets = torch.tensor([0, 1, 0])
    perfect_metrics = calculate_attack_metrics(perfect_predictions, perfect_targets, 2)
    
    assert perfect_metrics['acc'] == 100.0, "完美预测的准确率应该是100%"
    assert perfect_metrics['f1_score'] == 100.0, "完美预测的F1分数应该是100%"
    
    # 测试随机预测
    random_predictions = torch.randn(50, 5)  # 5类随机预测
    random_targets = torch.randint(0, 5, (50,))
    random_metrics = calculate_attack_metrics(random_predictions, random_targets, 5)
    
    # 随机预测的准确率应该大约在20%左右（1/5）
    assert 5 <= random_metrics['acc'] <= 40, f"随机预测准确率异常: {random_metrics['acc']}"
    
    print("✅ 边界情况测试通过!")

def test_compatibility_with_existing_code():
    """测试与现有代码的兼容性"""
    print("\n=== 测试与现有代码兼容性 ===")
    
    try:
        # 测试导入是否成功
        from utils.attack_metrics import evaluate_attack_model
        from utils.results_saver import save_attack_results
        print("✅ 所有模块导入成功!")
        
        # 测试results.csv格式
        with open('result/results.csv', 'r') as f:
            header = f.readline().strip()
            expected_header = "algorithm,dataset,attack_acc,attack_top1,attack_top5,attack_f1"
            assert header == expected_header, f"CSV格式不正确: {header}"
        print("✅ Results.csv格式正确!")
        
    except Exception as e:
        print(f"❌ 兼容性测试失败: {e}")
        raise

def main():
    """主测试函数"""
    print("开始测试新的攻击评估指标系统...\n")
    
    try:
        # 执行所有测试
        test_attack_metrics()
        test_results_saver()
        test_edge_cases()
        test_compatibility_with_existing_code()
        
        print("\n🎉 所有测试通过! 新的评估系统工作正常。")
        print("\n📋 总结:")
        print("✅ 攻击评估指标计算正确")
        print("✅ 结果保存和加载功能正常")
        print("✅ 边界情况处理正确")
        print("✅ 与现有代码兼容")
        
        print("\n🚀 现在可以运行攻击脚本测试新的评估指标:")
        print("   python3 passive_lia.py --dataset bcw --epochs 1")
        print("   python3 active_lia.py --dataset bcw --epochs 1")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
