#!/bin/bash

# BWL 消融实验脚本
# 运行不同版本的BWL来验证各组件的贡献

# 设置使用的GPU
export CUDA_VISIBLE_DEVICES=4

echo "=================================================="
echo "开始BWL消融实验"
echo "使用GPU: $CUDA_VISIBLE_DEVICES"
echo "=================================================="

# 设置实验参数
EPOCHS=20
LR=0.001
BATCH_SIZE=64
ALPHA=1.0

# 选择要测试的数据集
DATASETS=("bcw" "cifar10" "cinic10")

for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo "================================================================"
    echo "数据集: $DATASET"
    echo "================================================================"
    
    echo ""
    echo "--- 1. 完整BWL（影子模型 + 边界漫游损失）---"
    python3 main.py --dataset $DATASET --epochs $EPOCHS --lr $LR --batch_size $BATCH_SIZE --alpha $ALPHA
    echo "完整BWL实验完成"
    
    echo ""
    echo "--- 2. BWL消融版本（只有边界漫游损失，无影子模型）---"
    python3 bwl_without_shadow.py --dataset $DATASET --epochs $EPOCHS --lr $LR --batch_size $BATCH_SIZE --alpha $ALPHA
    echo "BWL消融版本实验完成"
    
    echo ""
    echo "--- 3. 基线版本（有特征划分，无BWL机制）---"
    python3 baseline_no_bwl.py --dataset $DATASET --epochs $EPOCHS --lr $LR --batch_size $BATCH_SIZE --use_partition
    echo "基线版本（有特征划分）实验完成"
    
    echo ""
    echo "--- 4. 纯基线版本（无特征划分，无BWL机制）---"
    python3 baseline_no_bwl.py --dataset $DATASET --epochs $EPOCHS --lr $LR --batch_size $BATCH_SIZE
    echo "纯基线版本实验完成"
    
    echo ""
    echo "$DATASET 数据集的所有消融实验完成"
    echo "================================================================"
done

echo ""
echo "=================================================="
echo "所有消融实验已完成"
echo "结果已保存到 result/results.csv"
echo "=================================================="

# 显示结果摘要
echo ""
echo "=== 消融实验结果摘要 ==="
if [ -f "result/results.csv" ]; then
    echo "从 result/results.csv 读取结果:"
    echo ""
    
    # 显示表头
    head -1 result/results.csv
    echo "---"
    
    # 显示BWL相关的结果
    grep -E "(BWL|Baseline)" result/results.csv | sort
else
    echo "结果文件不存在"
fi
