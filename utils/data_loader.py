import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import pickle
import os
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from PIL import Image
import io

# --- BCW (Breast Cancer Wisconsin) Loader ---
def load_bcw():
    print("加载BCW数据集...")
    breast_cancer_wisconsin = fetch_ucirepo(id=17)
    X = breast_cancer_wisconsin.data.features
    y = breast_cancer_wisconsin.data.targets
    le = LabelEncoder()
    y = le.fit_transform(y.values.ravel())
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_a_train, X_b_train = X_train[:, :15], X_train[:, 15:]
    X_a_test, X_b_test = X_test[:, :15], X_test[:, 15:]
    X_a_train_t = torch.from_numpy(X_a_train).float()
    X_b_train_t = torch.from_numpy(X_b_train).float()
    y_train_t = torch.from_numpy(y_train).long()
    X_a_test_t = torch.from_numpy(X_a_test).float()
    X_b_test_t = torch.from_numpy(X_b_test).float()
    y_test_t = torch.from_numpy(y_test).long()
    print("BCW数据集加载和预处理完成.")
    return (X_a_train_t, X_b_train_t, y_train_t), (X_a_test_t, X_b_test_t, y_test_t)

# --- Image Loaders (CIFAR-10, CINIC-10) ---
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_cifar10(data_dir="/root/dataset/cifar10"):
    print("加载CIFAR-10数据集...")
    train_data, train_labels = [], []
    for i in range(1, 6):
        batch_path = os.path.join(data_dir, f'data_batch_{i}')
        batch_dict = unpickle(batch_path)
        train_data.append(batch_dict[b'data'])
        train_labels.extend(batch_dict[b'labels'])
    X_train = np.concatenate(train_data)
    y_train = np.array(train_labels)

    test_dict = unpickle(os.path.join(data_dir, 'test_batch'))
    X_test, y_test = np.array(test_dict[b'data']), np.array(test_dict[b'labels'])

    X_train = X_train.reshape(50000, 3, 32, 32).astype("float32") / 255.0
    X_test = X_test.reshape(10000, 3, 32, 32).astype("float32") / 255.0

    X_a_train, X_b_train = X_train[:, :, :, :16], X_train[:, :, :, 16:]
    X_a_test, X_b_test = X_test[:, :, :, :16], X_test[:, :, :, 16:]

    print("CIFAR-10数据集加载和预处理完成.")
    return (torch.from_numpy(X_a_train), torch.from_numpy(X_b_train), torch.from_numpy(y_train).long()), \
           (torch.from_numpy(X_a_test), torch.from_numpy(X_b_test), torch.from_numpy(y_test).long())

def load_cinic10(data_dir="/root/dataset/cinic10/data"):
    print("加载CINIC-10数据集...")
    train_df = pd.read_parquet(os.path.join(data_dir, 'train-00000-of-00001.parquet'))
    test_df = pd.read_parquet(os.path.join(data_dir, 'test-00000-of-00001.parquet'))

    def process_df(df):
        images = df['image'].tolist()
        labels = df['label'].tolist()
        
        print(f"处理 {len(images)} 个图像...")
        
        # 检查第一个图像的结构
        if len(images) > 0:
            first_image = images[0]
            print(f"第一个图像类型: {type(first_image)}")
            if isinstance(first_image, dict):
                print(f"图像字典键: {list(first_image.keys())}")
        
        processed_images = []
        processed_labels = []
        
        for i, (img_data, label) in enumerate(zip(images, labels)):
            try:
                if isinstance(img_data, dict) and 'bytes' in img_data:
                    # 从字节数据解码图像
                    img_bytes = img_data['bytes']
                    img = Image.open(io.BytesIO(img_bytes))
                    
                    # 确保图像是RGB格式
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # 调整图像大小确保一致性（CINIC-10应该是32x32）
                    if img.size != (32, 32):
                        img = img.resize((32, 32), Image.LANCZOS)
                    
                    # 转换为numpy数组
                    img_array = np.array(img)
                    
                    # 确保形状是(32, 32, 3)
                    if img_array.shape != (32, 32, 3):
                        if i % 1000 == 0:  # 减少警告输出频率
                            print(f"警告：第 {i} 个图像形状异常: {img_array.shape}，跳过此图像")
                        continue
                        
                    processed_images.append(img_array)
                    processed_labels.append(label)
                    
                elif hasattr(img_data, '__array__') or hasattr(img_data, '__len__'):
                    # 如果是numpy数组或类似的数据
                    img_array = np.array(img_data)
                    if img_array.shape == (32, 32, 3):
                        processed_images.append(img_array)
                        processed_labels.append(label)
                    else:
                        if i % 1000 == 0:  # 减少警告输出频率
                            print(f"警告：第 {i} 个图像形状异常: {img_array.shape}，跳过此图像")
                        continue
                else:
                    if i % 1000 == 0:  # 减少警告输出频率
                        print(f"警告：第 {i} 个图像数据格式未知: {type(img_data)}，跳过此图像")
                    continue
                    
                if i % 10000 == 0:
                    print(f"已处理 {i}/{len(images)} 个图像，成功 {len(processed_images)} 个")
                    
            except Exception as e:
                if i % 1000 == 0:  # 减少错误输出频率
                    print(f"处理第 {i} 个图像时出错: {e}，跳过此图像")
                continue
        
        print(f"成功处理了 {len(processed_images)} 个图像，跳过了 {len(images) - len(processed_images)} 个图像")
        
        if len(processed_images) == 0:
            raise ValueError("没有成功处理任何图像")
        
        # 现在所有图像都应该有相同的形状
        images_np = np.array(processed_images)
        labels_np = np.array(processed_labels)
        
        print(f"处理后图像数组形状: {images_np.shape}")
        print(f"处理后标签数组形状: {labels_np.shape}")
        
        # 确保图像是(N, H, W, C)格式，然后转换为(N, C, H, W)
        if len(images_np.shape) == 4 and images_np.shape[-1] == 3:
            images_np = images_np.transpose((0, 3, 1, 2)).astype("float32") / 255.0
        else:
            raise ValueError(f"期望(N, H, W, C)格式的图像数组，得到: {images_np.shape}")
        
        print(f"最终图像数组形状: {images_np.shape}")
        
        return images_np, labels_np

    X_train, y_train = process_df(train_df)
    X_test, y_test = process_df(test_df)
    print("CINIC-10数据集加载和预处理完成.")

    X_a_train, X_b_train = X_train[:, :, :, :16], X_train[:, :, :, 16:]
    X_a_test, X_b_test = X_test[:, :, :, :16], X_test[:, :, :, 16:]

    return (torch.from_numpy(X_a_train), torch.from_numpy(X_b_train), torch.from_numpy(y_train).long()), \
           (torch.from_numpy(X_a_test), torch.from_numpy(X_b_test), torch.from_numpy(y_test).long())

# --- Generic DataLoader Creator ---
def create_dataloader(X_a, X_b, y, batch_size=64, shuffle=True):
    dataset = TensorDataset(X_a, X_b, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
