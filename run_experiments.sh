#!/bin/bash

# 设置使用的GPU
export CUDA_VISIBLE_DEVICES=4

echo "=================================================="
echo "开始所有实验"
echo "使用GPU: $CUDA_VISIBLE_DEVICES"
echo "=================================================="

# --- BCW 数据集 ---
echo ""
echo "--- 运行 BCW 数据集 ---"

echo "在 BCW 上运行 BWL 算法..."
python3 main.py --dataset bcw --epochs 20

echo "在 BCW 上运行 原版VFL 算法..."
python3 vanilla_vfl.py --dataset bcw --epochs 20

echo "--- BCW 数据集运行完毕 ---"


## --- CIFAR-10 数据集 ---
#echo ""
#echo "--- 运行 CIFAR-10 数据集 ---"
#
#echo "在 CIFAR-10 上运行 BWL 算法..."
#python3 main.py --dataset cifar10 --epochs 30
#
#echo "在 CIFAR-10 上运行 原版VFL 算法..."
#python3 vanilla_vfl.py --dataset cifar10 --epochs 30
#
#echo "--- CIFAR-10 数据集运行完毕 ---"
#
#
## --- CINIC-10 数据集 ---
#echo ""
#echo "--- 运行 CINIC-10 数据集 ---"
#
#echo "在 CINIC-10 上运行 BWL 算法..."
#python3 main.py --dataset cinic10 --epochs 30
#
#echo "在 CINIC-10 上运行 原版VFL 算法..."
#python3 vanilla_vfl.py --dataset cinic10 --epochs 30
#
#echo "--- CINIC-10 数据集运行完毕 ---"
#
#echo ""
#echo "=================================================="
#echo "所有实验已结束."
#echo "=================================================="
