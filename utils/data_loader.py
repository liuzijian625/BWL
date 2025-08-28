import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import pickle
import os
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# --- BCW (Breast Cancer Wisconsin) Loader --- #
def load_bcw():
    print("Loading Breast Cancer Wisconsin dataset...")
    # Fetch dataset
    breast_cancer_wisconsin = fetch_ucirepo(id=17)

    # Data (as pandas dataframes)
    X = breast_cancer_wisconsin.data.features
    y = breast_cancer_wisconsin.data.targets

    # Preprocessing
    # 1. Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y.values.ravel())

    # 2. Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 3. Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. VFL Split (15 features for each party)
    X_a_train, X_b_train = X_train[:, :15], X_train[:, 15:]
    X_a_test, X_b_test = X_test[:, :15], X_test[:, 15:]

    # 5. SMP Split for Party B (8 public, 7 private)
    # This split is done on the indices, which will be used in the training loop
    party_b_feature_indices = np.arange(15)
    np.random.shuffle(party_b_feature_indices)
    public_indices = party_b_feature_indices[:8]
    private_indices = party_b_feature_indices[8:]

    # 6. Convert to Tensors
    X_a_train_t = torch.from_numpy(X_a_train).float()
    X_b_train_t = torch.from_numpy(X_b_train).float()
    y_train_t = torch.from_numpy(y_train).long()
    X_a_test_t = torch.from_numpy(X_a_test).float()
    X_b_test_t = torch.from_numpy(X_b_test).float()
    y_test_t = torch.from_numpy(y_test).long()
    
    print("BCW data loaded and preprocessed.")

    return (X_a_train_t, X_b_train_t, y_train_t), \
           (X_a_test_t, X_b_test_t, y_test_t), \
           public_indices, private_indices

# --- Image Loaders (CIFAR-10, CINIC-10) --- #
# (Keeping them for reference, but they are not used for BCW)

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_cifar10(data_dir="/root/dataset/cifar10"):
    # ... (implementation from before)
    pass

def load_cinic10(data_dir="/root/dataset/cinic10/data"):
    # ... (implementation from before)
    pass

# --- Generic DataLoader Creator --- #
def create_dataloader(X_a, X_b, y, batch_size=64, shuffle=True):
    dataset = TensorDataset(X_a, X_b, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader