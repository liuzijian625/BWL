#!/bin/bash
# 该脚本用于批量运行所有针对原版VFL的标签推断攻击

# 设置在4号显卡上运行
export CUDA_VISIBLE_DEVICES=4

set -e

echo "Running all experiments on GPU: $CUDA_VISIBLE_DEVICES"

echo "===================================================="
echo "Running Passive Label Inference Attacks (LIA)"
echo "===================================================="

python passive_lia.py --dataset bcw
python passive_lia.py --dataset cifar10
python passive_lia.py --dataset cinic10

echo ""
echo "===================================================="
echo "Running Active Label Inference Attacks (LIA)"
echo "===================================================="

python active_lia.py --dataset bcw
python active_lia.py --dataset cifar10
python active_lia.py --dataset cinic10

echo ""
echo "===================================================="
echo "Running Direct Label Inference Attacks (LIA)"
echo "===================================================="

python direct_lia.py --dataset bcw
python direct_lia.py --dataset cifar10
python direct_lia.py --dataset cinic10

echo ""
echo "===================================================="
echo "All original attack scripts have been executed."
echo "Check results in the 'result/' directory."
echo "===================================================="