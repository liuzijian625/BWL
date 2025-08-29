import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import shap
import xgboost as xgb
import warnings

def partition_features_shap(X_features, y_labels, private_ratio=0.3, model_type='xgboost', random_state=42):
    """
    基于SHAP值的模型驱动特征划分
    
    Args:
        X_features (np.ndarray): 特征矩阵 (n_samples, n_features)
        y_labels (np.ndarray): 标签向量 (n_samples,)
        private_ratio (float): 私有特征比例
        model_type (str): 临时模型类型 ('xgboost', 'random_forest')
        random_state (int): 随机种子
        
    Returns:
        tuple: (public_indices, private_indices)
    """
    print(f"正在使用SHAP值进行特征划分，私有特征比例: {private_ratio}")
    
    num_features = X_features.shape[1]
    num_private_features = int(num_features * private_ratio)
    
    # 如果只有少数特征，直接随机划分
    if num_features <= 4:
        print("特征数量过少，回退到随机划分")
        return partition_features_random(X_features, y_labels, private_ratio, random_state)
    
    try:
        # 训练数据分割
        X_train, X_test, y_train, y_test = train_test_split(
            X_features, y_labels, test_size=0.3, random_state=random_state, stratify=y_labels
        )
        
        # 训练临时模型
        if model_type == 'xgboost':
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=random_state,
                eval_metric='logloss'
            )
        elif model_type == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=random_state
            )
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        # 训练模型
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train)
        
        # 计算SHAP值
        if model_type == 'xgboost':
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.TreeExplainer(model)
            
        # 使用训练集的子集计算SHAP值（避免内存问题）
        sample_size = min(100, X_train.shape[0])
        sample_indices = np.random.choice(X_train.shape[0], sample_size, replace=False)
        X_sample = X_train[sample_indices]
        
        shap_values = explainer.shap_values(X_sample)
        
        # 处理多分类情况
        if isinstance(shap_values, list):
            # 多分类：计算所有类别的平均绝对SHAP值
            mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
        else:
            # 二分类或回归
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        # 根据SHAP值排序特征
        feature_importance_order = np.argsort(mean_abs_shap)[::-1]  # 从高到低
        
        # 划分特征
        private_indices = feature_importance_order[:num_private_features]
        public_indices = feature_importance_order[num_private_features:]
        
        print(f"SHAP特征划分完成：私有特征{len(private_indices)}个，公开特征{len(public_indices)}个")
        print(f"私有特征重要性: {mean_abs_shap[private_indices][:5]}...")  # 显示前5个
        
        return public_indices, private_indices
        
    except Exception as e:
        print(f"SHAP划分失败，错误: {str(e)}，回退到随机划分")
        return partition_features_random(X_features, y_labels, private_ratio, random_state)


def partition_features_mutual_info(X_features, y_labels, private_ratio=0.3, random_state=42):
    """
    基于互信息的特征划分
    
    Args:
        X_features (np.ndarray): 特征矩阵 (n_samples, n_features)
        y_labels (np.ndarray): 标签向量 (n_samples,)
        private_ratio (float): 私有特征比例
        random_state (int): 随机种子
        
    Returns:
        tuple: (public_indices, private_indices)
    """
    print(f"正在使用互信息进行特征划分，私有特征比例: {private_ratio}")
    
    num_features = X_features.shape[1]
    num_private_features = int(num_features * private_ratio)
    
    try:
        # 计算每个特征与标签的互信息
        mi_scores = mutual_info_classif(
            X_features, y_labels, 
            discrete_features=False,
            random_state=random_state
        )
        
        # 根据互信息值排序特征
        feature_importance_order = np.argsort(mi_scores)[::-1]  # 从高到低
        
        # 划分特征
        private_indices = feature_importance_order[:num_private_features]
        public_indices = feature_importance_order[num_private_features:]
        
        print(f"互信息特征划分完成：私有特征{len(private_indices)}个，公开特征{len(public_indices)}个")
        print(f"私有特征互信息: {mi_scores[private_indices][:5]}...")  # 显示前5个
        
        return public_indices, private_indices
        
    except Exception as e:
        print(f"互信息划分失败，错误: {str(e)}，回退到随机划分")
        return partition_features_random(X_features, y_labels, private_ratio, random_state)


def partition_features_random(X_features, y_labels, private_ratio=0.3, random_state=42):
    """
    随机特征划分
    
    Args:
        X_features (np.ndarray): 特征矩阵 (n_samples, n_features)
        y_labels (np.ndarray): 标签向量 (n_samples,)
        private_ratio (float): 私有特征比例
        random_state (int): 随机种子
        
    Returns:
        tuple: (public_indices, private_indices)
    """
    print(f"正在使用随机方法进行特征划分，私有特征比例: {private_ratio}")
    
    num_features = X_features.shape[1]
    num_private_features = int(num_features * private_ratio)
    
    # 设置随机种子以保证可重复性
    np.random.seed(random_state)
    
    indices = np.arange(num_features)
    np.random.shuffle(indices)
    
    private_indices = indices[:num_private_features]
    public_indices = indices[num_private_features:]
    
    print(f"随机特征划分完成：私有特征{len(private_indices)}个，公开特征{len(public_indices)}个")
    
    return public_indices, private_indices


def partition_features(X_features, y_labels, method='random', private_ratio=0.3, random_state=42, **kwargs):
    """
    统一的特征划分接口
    
    Args:
        X_features (np.ndarray): 特征矩阵 (n_samples, n_features)
        y_labels (np.ndarray): 标签向量 (n_samples,)
        method (str): 划分方法 ('shap', 'mutual_info', 'random')
        private_ratio (float): 私有特征比例
        random_state (int): 随机种子
        **kwargs: 其他方法特定参数
        
    Returns:
        tuple: (public_indices, private_indices)
    """
    print(f"\n=== 开始特征划分 ===")
    print(f"方法: {method}")
    print(f"特征矩阵形状: {X_features.shape}")
    print(f"标签数量: {len(np.unique(y_labels))} 类")
    
    if method == 'shap':
        return partition_features_shap(X_features, y_labels, private_ratio, random_state=random_state, **kwargs)
    elif method == 'mutual_info':
        return partition_features_mutual_info(X_features, y_labels, private_ratio, random_state=random_state)
    elif method == 'random':
        return partition_features_random(X_features, y_labels, private_ratio, random_state=random_state)
    else:
        raise ValueError(f"不支持的特征划分方法: {method}. 支持的方法: 'shap', 'mutual_info', 'random'")


# 保持向后兼容性
def partition_features_legacy(features, private_ratio=0.3):
    """
    保持向后兼容的旧接口
    
    Args:
        features (np.ndarray): 特征矩阵
        private_ratio (float): 私有特征比例
        
    Returns:
        tuple: (public_indices, private_indices)
    """
    warnings.warn("partition_features_legacy已弃用，请使用partition_features", DeprecationWarning)
    
    # 创建虚拟标签进行随机划分
    dummy_labels = np.zeros(features.shape[0])
    return partition_features_random(features, dummy_labels, private_ratio)