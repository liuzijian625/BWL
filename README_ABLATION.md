# BWL 消融实验指南

本文档介绍如何运行BWL（边界漫游学习）的消融实验，用于验证不同组件的独立贡献。

## 📊 消融实验版本

### 1. **完整BWL版本** (`main.py`)
- ✅ 影子模型双轨架构
- ✅ 边界漫游损失
- ✅ 特征划分机制
- **用途**: 完整的BWL防御机制

### 2. **BWL消融版本** (`bwl_without_shadow.py`)
- ❌ 影子模型双轨架构
- ✅ 边界漫游损失
- ✅ 特征划分机制
- **用途**: 验证影子模型架构的贡献

### 3. **基线版本（有特征划分）** (`baseline_no_bwl.py --use_partition`)
- ❌ 影子模型双轨架构
- ❌ 边界漫游损失
- ✅ 特征划分机制
- **用途**: 验证BWL损失的贡献

### 4. **纯基线版本** (`baseline_no_bwl.py`)
- ❌ 影子模型双轨架构
- ❌ 边界漫游损失
- ❌ 特征划分机制
- **用途**: 标准VFL基线对照

## 🚀 快速开始

### 方法一：快速测试（推荐）
```bash
# 在BCW数据集上快速验证所有版本（训练5轮）
bash run_ablation_quick.sh
```

### 方法二：完整实验
```bash
# 在所有数据集上运行完整消融实验（训练20轮）
bash run_ablation_study.sh
```

### 方法三：单独运行特定版本

#### 完整BWL
```bash
python3 main.py --dataset bcw --epochs 20 --alpha 1.0
```

#### BWL消融版本（无影子模型）
```bash
python3 bwl_without_shadow.py --dataset bcw --epochs 20 --alpha 1.0
```

#### 基线版本（有特征划分）
```bash
python3 baseline_no_bwl.py --dataset bcw --epochs 20 --use_partition
```

#### 纯基线版本（无特征划分）
```bash
python3 baseline_no_bwl.py --dataset bcw --epochs 20
```

## 🧪 测试验证

运行测试脚本验证所有版本是否正常工作：
```bash
python3 test_ablation_versions.py
```

## 📈 结果分析

### 查看结果
消融实验结果保存在 `result/results.csv` 中，包含以下算法：

| 算法名称 | 说明 |
|---------|------|
| `BWL` | 完整的边界漫游学习 |
| `BWL_No_Shadow` | 只有边界漫游损失，无影子模型 |
| `Baseline_with_partition` | 有特征划分，无BWL机制 |
| `Baseline_no_partition` | 无特征划分，无BWL机制 |

### 预期分析目标

1. **影子模型架构的贡献**：
   - 比较 `BWL` vs `BWL_No_Shadow`
   - 验证双轨架构是否提升了防御效果

2. **边界漫游损失的贡献**：
   - 比较 `BWL_No_Shadow` vs `Baseline_with_partition`
   - 验证BWL损失是否有效

3. **特征划分的贡献**：
   - 比较 `Baseline_with_partition` vs `Baseline_no_partition`
   - 验证智能特征划分的效果

4. **整体BWL机制的贡献**：
   - 比较 `BWL` vs `Baseline_no_partition`
   - 验证完整BWL相比标准VFL的提升

## ⚙️ 参数配置

### 通用参数
- `--dataset`: 数据集选择 (`bcw`, `cifar10`, `cinic10`)
- `--epochs`: 训练轮数
- `--lr`: 学习率
- `--batch_size`: 批处理大小

### BWL特定参数
- `--alpha`: 边界漫游损失权重（仅BWL相关版本）
- `--partition_method`: 特征划分方法 (`shap`, `mutual_info`, `random`)
- `--private_ratio`: 私有特征比例

### 基线版本参数
- `--use_partition`: 是否启用特征划分

## 📝 示例输出

运行 `bash run_ablation_quick.sh` 的示例输出：
```
=== 快速消融实验结果摘要 ===
Algorithm,Dataset,Main_Accuracy,Shadow_Accuracy,Attack_Accuracy
---
BWL,bcw,96.49,95.61,N/A
BWL_No_Shadow,bcw,95.26,N/A,N/A
Baseline_with_partition,bcw,96.12,N/A,N/A
Baseline_no_partition,bcw,96.49,N/A,N/A
```

## 🔬 实验建议

### 快速验证
1. 先运行 `python3 test_ablation_versions.py` 确保环境正常
2. 运行 `bash run_ablation_quick.sh` 快速测试
3. 检查结果是否合理

### 完整实验
1. 确认快速测试通过后，运行 `bash run_ablation_study.sh`
2. 在多个数据集上对比结果
3. 分析不同alpha值的影响

### 深入分析
1. 尝试不同的特征划分方法：
   ```bash
   python3 main.py --dataset bcw --partition_method shap --private_ratio 0.2
   python3 main.py --dataset bcw --partition_method mutual_info --private_ratio 0.3
   ```

2. 测试不同的BWL损失权重：
   ```bash
   python3 bwl_without_shadow.py --dataset bcw --alpha 0.5
   python3 bwl_without_shadow.py --dataset bcw --alpha 2.0
   ```

## 🐛 故障排除

### 常见问题

1. **依赖包缺失**：
   ```bash
   pip install -r requirements.txt
   ```

2. **GPU内存不足**：
   - 减少 `--batch_size`
   - 或设置 `export CUDA_VISIBLE_DEVICES=""` 使用CPU

3. **权限问题**：
   ```bash
   chmod +x run_ablation_*.sh
   ```

4. **结果文件权限**：
   ```bash
   chmod 755 result/
   ```

## 📊 结果说明

消融实验的目标是回答以下问题：

1. **影子模型架构是否必要？** → 比较完整BWL vs BWL_No_Shadow
2. **边界漫游损失是否有效？** → 比较BWL_No_Shadow vs Baseline_with_partition  
3. **特征划分是否重要？** → 比较Baseline_with_partition vs Baseline_no_partition
4. **BWL整体提升如何？** → 比较BWL vs Baseline_no_partition

通过这些比较，可以量化每个组件的独立贡献，为BWL机制的有效性提供科学证据。
