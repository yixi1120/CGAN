import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


class ImageProcessor:
    def __init__(self, size=256):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def load_image(self, image_path):
        """加载并预处理图片"""
        image = Image.open(image_path)
        image_tensor = self.transform(image)
        return image_tensor.unsqueeze(0).to(self.device)  # 添加batch维度并移到GPU

    def save_image(self, tensor, save_path):
        """将输出张量转换回图片并保存"""
        # 将值从[-1,1]转换回[0,1]
        tensor = (tensor + 1) / 2.0
        # 将tensor转换为numpy数组
        image = tensor.squeeze(0).detach().cpu().numpy()
        # 转换通道顺序从CxHxW到HxWxC
        image = np.transpose(image, (1, 2, 0))
        # 确保值在[0,1]范围内
        image = np.clip(image, 0, 1)
        # 转换为0-255的uint8格式
        image = (image * 255).astype(np.uint8)
        # 保存图片
        Image.fromarray(image).save(save_path)

    def load_image_pair(self, condition_image_path, real_image_path):
        """加载条件图像和真实图像，返回包含这两个图像的元组"""
        condition_image = self.load_image(condition_image_path)
        real_image = self.load_image(real_image_path)
        return condition_image, real_image
    