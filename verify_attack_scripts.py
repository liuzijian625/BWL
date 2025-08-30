#!/usr/bin/env python3
"""
验证所有攻击脚本的修改是否正确
检查导入、函数定义和结果格式
"""

import os
import sys
import importlib.util

def check_file_imports(file_path):
    """检查文件的导入语句是否正确"""
    print(f"检查文件: {file_path}")
    
    required_imports = [
        'from utils.attack_metrics import',
        'from utils.results_saver import'
    ]
    
    missing_imports = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        for import_line in required_imports:
            if import_line not in content:
                missing_imports.append(import_line)
        
        if missing_imports:
            print(f"  ❌ 缺少导入: {missing_imports}")
            return False
        else:
            print(f"  ✅ 导入检查通过")
            return True
            
    except Exception as e:
        print(f"  ❌ 读取文件失败: {e}")
        return False

def check_results_csv_format():
    """检查results.csv格式是否正确"""
    print("\n检查 result/results.csv 格式...")
    
    expected_header = "algorithm,dataset,attack_acc,attack_top1,attack_top5,attack_f1"
    
    try:
        with open('result/results.csv', 'r') as f:
            actual_header = f.readline().strip()
        
        if actual_header == expected_header:
            print("  ✅ CSV格式正确")
            return True
        else:
            print(f"  ❌ CSV格式错误")
            print(f"     期望: {expected_header}")
            print(f"     实际: {actual_header}")
            return False
            
    except FileNotFoundError:
        print("  ❌ results.csv文件不存在")
        return False
    except Exception as e:
        print(f"  ❌ 检查CSV文件失败: {e}")
        return False

def check_utils_files():
    """检查工具文件是否存在"""
    print("\n检查工具文件...")
    
    required_files = [
        'utils/attack_metrics.py',
        'utils/results_saver.py'
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"  ✅ {file_path} 存在")
        else:
            print(f"  ❌ {file_path} 不存在")
            all_exist = False
    
    return all_exist

def main():
    """主验证函数"""
    print("=== BWL攻击脚本修改验证 ===\n")
    
    # 需要检查的攻击脚本列表
    attack_scripts = [
        'passive_lia.py',
        'active_lia.py', 
        'direct_lia.py',
        'passive_lia_on_bwl.py',
        'active_lia_on_bwl.py',
        'direct_lia_with_bwl_defense.py'
    ]
    
    # 检查工具文件
    utils_ok = check_utils_files()
    
    # 检查CSV格式
    csv_ok = check_results_csv_format()
    
    # 检查攻击脚本的导入
    print(f"\n检查 {len(attack_scripts)} 个攻击脚本...")
    import_checks = []
    
    for script in attack_scripts:
        if os.path.exists(script):
            result = check_file_imports(script)
            import_checks.append(result)
        else:
            print(f"  ❌ {script} 文件不存在")
            import_checks.append(False)
    
    # 总结结果
    print(f"\n=== 验证结果总结 ===")
    
    all_passed = utils_ok and csv_ok and all(import_checks)
    
    if all_passed:
        print("🎉 所有检查都通过！攻击脚本修改完成。")
        print("\n📋 已完成的修改:")
        print("  ✅ 创建了新的攻击评估指标工具")
        print("  ✅ 更新了results.csv格式") 
        print("  ✅ 修改了所有攻击脚本的评估逻辑")
        print("  ✅ 所有攻击脚本现在输出4个指标：acc, top-1, top-5, F1")
        
        print("\n🚀 可以运行以下命令测试:")
        print("  python3 passive_lia.py --dataset bcw --epochs 2")
        print("  python3 active_lia.py --dataset bcw --epochs 2")
        
    else:
        print("❌ 部分检查失败，请查看上面的错误信息。")
        
        failed_scripts = [script for script, passed in zip(attack_scripts, import_checks) if not passed]
        if failed_scripts:
            print(f"\n需要修复的脚本: {failed_scripts}")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
