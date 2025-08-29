#!/bin/bash
# 该脚本用于批量运行所有针对BWL防御的评估攻击

# 设置在4号显卡上运行
export CUDA_VISIBLE_DEVICES=4

set -e

#echo "Running all BWL evaluation experiments on GPU: $CUDA_VISIBLE_DEVICES"
#
#echo "===================================================="
#echo "Running Passive Label Inference Attacks on BWL"
#echo "===================================================="
#
#python passive_lia_on_bwl.py --dataset bcw
#python passive_lia_on_bwl.py --dataset cifar10
#python passive_lia_on_bwl.py --dataset cinic10
#
#echo ""
#echo "===================================================="
#echo "Running Active Label Inference Attacks on BWL"
#echo "===================================================="
#
#python active_lia_on_bwl.py --dataset bcw
#python active_lia_on_bwl.py --dataset cifar10
#python active_lia_on_bwl.py --dataset cinic10
#
#echo ""
#echo "===================================================="
#echo "Running Direct LIA with BWL-style Defense"
#echo "===================================================="

python direct_lia_with_bwl_defense.py --dataset bcw --alpha 0.5 --beta 0.5
python direct_lia_with_bwl_defense.py --dataset cifar10 --alpha 0.5 --beta 0.5
python direct_lia_with_bwl_defense.py --dataset cinic10 --alpha 0.5 --beta 0.5

echo ""
echo "===================================================="
echo "All BWL evaluation scripts have been executed."
echo "Check results in the 'result/' directory."
echo "===================================================="