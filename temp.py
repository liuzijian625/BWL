import os

def list_directory_structure(path, prefix="", is_last=True):
    """递归列出目录结构"""
    if not os.path.exists(path):
        print(f"错误: 路径 {path} 不存在")
        return
    
    # 获取目录下所有文件和文件夹
    items = os.listdir(path)
    items.sort()  # 排序
    
    for i, item in enumerate(items):
        item_path = os.path.join(path, item)
        is_last_item = (i == len(items) - 1)
        
        # 确定当前项的显示前缀
        if is_last:
            current_prefix = "└── "
            next_prefix = "    "
        else:
            current_prefix = "├── "
            next_prefix = "│   "
        
        print(f"{prefix}{current_prefix}{item}")
        
        # 如果是目录，递归显示其内容
        if os.path.isdir(item_path):
            list_directory_structure(item_path, prefix + next_prefix, is_last_item)

def list_cifar10_files():
    """列出CINIC10数据集文件夹结构（递归）"""
    # 指定CINIC10数据集路径
    cifar10_path = '/root/dataset/BCW'
    
    # 检查目录是否存在
    if os.path.exists(cifar10_path):
        print(f"BCW数据集路径: {cifar10_path}")
        print("=" * 50)
        print("文件夹结构（递归）:")
        print("-" * 30)
        
        # 递归列出目录结构
        list_directory_structure(cifar10_path)
    else:
        print(f"错误: 路径 {cifar10_path} 不存在")

if __name__ == "__main__":
    list_cifar10_files()
