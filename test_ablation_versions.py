#!/usr/bin/env python3
"""
测试消融实验版本的功能验证脚本
快速验证各个消融版本是否能正常工作
"""

import subprocess
import sys
import os
import time

def run_command_with_timeout(cmd, timeout=60):
    """运行命令并设置超时"""
    try:
        print(f"运行命令: {cmd}")
        process = subprocess.Popen(
            cmd, shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        stdout, stderr = process.communicate(timeout=timeout)
        
        if process.returncode == 0:
            print(f"✅ 命令成功执行")
            return True, stdout, stderr
        else:
            print(f"❌ 命令执行失败，返回码: {process.returncode}")
            return False, stdout, stderr
            
    except subprocess.TimeoutExpired:
        process.kill()
        print(f"❌ 命令超时 ({timeout}秒)")
        return False, "", "Timeout"
    except Exception as e:
        print(f"❌ 命令执行异常: {str(e)}")
        return False, "", str(e)

def test_ablation_versions():
    """测试各个消融实验版本"""
    print("=== BWL消融实验版本测试 ===\n")
    
    # 测试参数
    test_args = "--dataset bcw --epochs 1 --batch_size 32"
    timeout = 120  # 2分钟超时
    
    test_cases = [
        {
            "name": "完整BWL版本",
            "script": "main.py",
            "args": f"{test_args} --alpha 1.0"
        },
        {
            "name": "BWL消融版本（无影子模型）",
            "script": "bwl_without_shadow.py", 
            "args": f"{test_args} --alpha 1.0"
        },
        {
            "name": "基线版本（有特征划分）",
            "script": "baseline_no_bwl.py",
            "args": f"{test_args} --use_partition"
        },
        {
            "name": "基线版本（无特征划分）",
            "script": "baseline_no_bwl.py",
            "args": test_args
        }
    ]
    
    results = {}
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"{i}. 测试 {test_case['name']}")
        print("-" * 50)
        
        cmd = f"cd /root/project/sys/BWL && python3 {test_case['script']} {test_case['args']}"
        
        start_time = time.time()
        success, stdout, stderr = run_command_with_timeout(cmd, timeout)
        end_time = time.time()
        
        results[test_case['name']] = {
            'success': success,
            'runtime': end_time - start_time,
            'stdout_lines': len(stdout.split('\n')) if stdout else 0,
            'stderr_lines': len(stderr.split('\n')) if stderr else 0
        }
        
        if success:
            print(f"   运行时间: {end_time - start_time:.1f}秒")
            print(f"   输出行数: {len(stdout.split('\n'))}")
        else:
            print(f"   失败原因: {stderr[:200]}...")
        
        print()
    
    # 结果总结
    print("=== 测试结果总结 ===")
    success_count = sum(1 for r in results.values() if r['success'])
    total_count = len(results)
    
    print(f"成功: {success_count}/{total_count}")
    print()
    
    for name, result in results.items():
        status = "✅ 成功" if result['success'] else "❌ 失败"
        runtime = f"{result['runtime']:.1f}s" if result['success'] else "N/A"
        print(f"  {name}: {status} ({runtime})")
    
    if success_count == total_count:
        print(f"\n🎉 所有消融实验版本测试通过！")
        print("可以运行完整的消融实验:")
        print("  bash run_ablation_quick.sh    # 快速测试")
        print("  bash run_ablation_study.sh    # 完整实验")
    else:
        print(f"\n⚠️  有 {total_count - success_count} 个版本测试失败，请检查代码")
    
    return results

def check_dependencies():
    """检查依赖包是否安装"""
    print("=== 检查依赖包 ===\n")
    
    required_packages = ['torch', 'numpy', 'pandas', 'sklearn', 'shap', 'xgboost']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}: 已安装")
        except ImportError:
            print(f"❌ {package}: 未安装")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  缺少依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False
    else:
        print(f"\n✅ 所有依赖包已安装")
        return True

def check_files():
    """检查必要文件是否存在"""
    print("=== 检查必要文件 ===\n")
    
    required_files = [
        'main.py',
        'bwl_without_shadow.py', 
        'baseline_no_bwl.py',
        'config.json',
        'utils/feature_partition.py',
        'losses/boundary_wandering_loss.py',
        'models/architectures.py'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        full_path = f'/root/project/sys/BWL/{file_path}'
        if os.path.exists(full_path):
            print(f"✅ {file_path}: 存在")
        else:
            print(f"❌ {file_path}: 不存在")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n⚠️  缺少文件: {', '.join(missing_files)}")
        return False
    else:
        print(f"\n✅ 所有必要文件存在")
        return True

if __name__ == "__main__":
    print("开始BWL消融实验版本测试...\n")
    
    # 检查环境
    deps_ok = check_dependencies()
    files_ok = check_files()
    
    if not (deps_ok and files_ok):
        print("❌ 环境检查失败，请修复后重试")
        sys.exit(1)
    
    # 运行测试
    print()
    test_results = test_ablation_versions()
    
    print(f"\n测试完成！")
