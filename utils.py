import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset


class EdgeImageDataset(Dataset):
    """边缘图和真实图像对数据集"""
    
    def __init__(self, root_dir, mode='train', transform=None):
        """
        参数:
            root_dir: 数据目录，包含边缘图和真实图像
            mode: 'train'或'val'或'test'
            transform: 图像预处理变换
        """
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        
        # 假设数据组织方式：
        # root_dir/
        #   ├── edges/        # 边缘图文件夹
        #   └── real_images/  # 真实图像文件夹
        
        self.edge_dir = os.path.join(root_dir, 'edges')
        self.real_dir = os.path.join(root_dir, 'real_images')
        
        # 获取所有图像文件名（假设边缘图和真实图像有相同的文件名）
        self.image_files = [
            f for f in os.listdir(self.edge_dir) 
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ]
        
        print(f"找到{len(self.image_files)}对边缘图-真实图像数据")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # 获取图像文件名
        img_name = self.image_files[idx]
        
        # 构建边缘图和真实图像的路径
        edge_path = os.path.join(self.edge_dir, img_name)
        real_path = os.path.join(self.real_dir, img_name)
        
        # 读取边缘图（灰度图）
        edge_img = Image.open(edge_path)
        if edge_img.mode != 'L':  # 确保是灰度图
            edge_img = edge_img.convert('L')
        
        # 读取真实图像（彩色图）
        real_img = Image.open(real_path)
        if real_img.mode != 'RGB':  # 确保是RGB图
            real_img = real_img.convert('RGB')
        
        # 应用变换
        if self.transform:
            edge_img = self.transform(edge_img)
            real_img = self.transform(real_img)
        
        return edge_img, real_img


def save_checkpoint(model, optimizer, epoch, filepath):
    """保存模型检查点
    
    参数:
        model: 要保存的模型
        optimizer: 优化器状态
        epoch: 当前轮数
        filepath: 保存路径
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, filepath)
    print(f"检查点已保存到 {filepath}")


def load_checkpoint(model, optimizer, filepath, device):
    """加载模型检查点
    
    参数:
        model: 要加载的模型
        optimizer: 优化器
        filepath: 检查点文件路径
        device: 设备（'cuda'或'cpu'）
    
    返回:
        epoch: 检查点对应的轮数
    """
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"从 {filepath} 加载检查点，轮数: {epoch}")
    return epoch


def save_samples(edges, real_imgs, fake_imgs, filepath, num_samples=4):
    """保存生成样本图像比较
    
    参数:
        edges: 输入的边缘图
        real_imgs: 真实图像
        fake_imgs: 生成的图像
        filepath: 保存路径
        num_samples: 要保存的样本数
    """
    # 将图像从张量转换为numpy数组
    def tensor_to_numpy(img):
        # 确保图像是在CPU上，并且已经去归一化
        img = (img.cpu().detach() * 0.5 + 0.5).clamp(0, 1)
        # 将通道维度移到最后
        return img.permute(0, 2, 3, 1).numpy()
    
    edges_np = tensor_to_numpy(edges[:num_samples])
    real_np = tensor_to_numpy(real_imgs[:num_samples])
    fake_np = tensor_to_numpy(fake_imgs[:num_samples])
    
    # 绘图
    fig, axs = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    
    for i in range(num_samples):
        # 显示边缘图
        if edges_np.shape[-1] == 1:  # 灰度图
            axs[i, 0].imshow(edges_np[i, :, :, 0], cmap='gray')
        else:  # RGB图
            axs[i, 0].imshow(edges_np[i])
        axs[i, 0].set_title('边缘图')
        axs[i, 0].axis('off')
        
        # 显示真实图像
        axs[i, 1].imshow(real_np[i])
        axs[i, 1].set_title('真实图像')
        axs[i, 1].axis('off')
        
        # 显示生成图像
        axs[i, 2].imshow(fake_np[i])
        axs[i, 2].set_title('生成图像')
        axs[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    print(f"样本已保存到 {filepath}")


def create_custom_dataset(edge_dir, real_dir, output_dir, 
                          num_pairs=1000, size=256):
    """创建自定义的边缘图-真实图像对数据集（用于测试）
    
    参数:
        edge_dir: 边缘图保存目录
        real_dir: 真实图像保存目录
        output_dir: 输出目录
        num_pairs: 生成的图像对数量
        size: 图像尺寸
    """
    os.makedirs(edge_dir, exist_ok=True)
    os.makedirs(real_dir, exist_ok=True)
    
    for i in range(num_pairs):
        # 创建随机边缘图（简单示例，实际使用需要更复杂的生成方法）
        edge = np.zeros((size, size), dtype=np.uint8)
        
        # 随机添加一些线条和形状（简化示例）
        num_lines = np.random.randint(5, 15)
        for _ in range(num_lines):
            x1, y1 = np.random.randint(0, size), np.random.randint(0, size)
            x2, y2 = np.random.randint(0, size), np.random.randint(0, size)
            thickness = np.random.randint(1, 4)
            color = 255
            # 画线
            cv2.line(edge, (x1, y1), (x2, y2), color, thickness)
        
        # 创建对应的"假想的"真实图像（彩色）
        real = np.zeros((size, size, 3), dtype=np.uint8)
        
        # 为了简单演示，我们只是基于边缘创建一些颜色区域
        for _ in range(np.random.randint(3, 8)):
            x, y = np.random.randint(0, size), np.random.randint(0, size)
            radius = np.random.randint(20, 80)
            color = (
                np.random.randint(0, 255),
                np.random.randint(0, 255),
                np.random.randint(0, 255)
            )
            cv2.circle(real, (x, y), radius, color, -1)
        
        # 将边缘覆盖到真实图像上
        edge_rgb = np.stack([edge, edge, edge], axis=2)
        real = np.where(edge_rgb > 0, edge_rgb, real)
        
        # 保存图像
        edge_img = Image.fromarray(edge)
        real_img = Image.fromarray(real)
        
        edge_img.save(os.path.join(edge_dir, f"{i:04d}.png"))
        real_img.save(os.path.join(real_dir, f"{i:04d}.png"))
    
    print(f"已创建{num_pairs}对边缘图-真实图像数据集")


if __name__ == "__main__":
    # 为了能够运行create_custom_dataset函数，导入cv2
    import cv2
    
    # 测试创建自定义数据集
    output_dir = "test_dataset"
    edge_dir = os.path.join(output_dir, "edges")
    real_dir = os.path.join(output_dir, "real_images")
    
    create_custom_dataset(edge_dir, real_dir, output_dir, 
                          num_pairs=10, size=256)
    
    # 测试数据集类
    transform = None  # 在此处可定义变换
    dataset = EdgeImageDataset(output_dir, transform=transform)
    edge, real = dataset[0]
    
    print(f"边缘图尺寸: {edge.size if isinstance(edge, Image.Image) else edge.shape}")
    print(f"真实图尺寸: {real.size if isinstance(real, Image.Image) else real.shape}") 