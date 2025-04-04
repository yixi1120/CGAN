import torch
import cv2
import numpy as np
from torchvision import models, transforms
from PIL import Image

class ImageEvaluator:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 加载预训练的FCN模型
        self.fcn_model = models.segmentation.fcn_resnet50().eval().to(self.device)
        
        # 图像预处理转换
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

    def preprocess_image(self, image_path):
        """预处理图像"""
        image = Image.open(image_path)
        return self.transform(image).unsqueeze(0).to(self.device)

    @torch.amp.autocast('cuda')  # 使用混合精度
    def compute_fcn_score(self, real_image_path, fake_image_path):
        """计算FCN语义分割评分"""
        real_img = self.preprocess_image(real_image_path)
        fake_img = self.preprocess_image(fake_image_path)

        with torch.no_grad():
            real_output = self.fcn_model(real_img)['out'].argmax(1)
            fake_output = self.fcn_model(fake_img)['out'].argmax(1)
            
            # 在GPU上计算IoU
            real_mask = (real_output > 0)
            fake_mask = (fake_output > 0)
            intersection = torch.logical_and(real_mask, fake_mask).sum().float()
            union = real_mask.sum().float() + 1e-10
            iou = (intersection / union).item()

        return iou

    def compare_histograms(self, real_image_path, fake_image_path):
        """计算颜色直方图差异"""
        # 使用GPU加速的直方图计算
        real_img = torch.from_numpy(cv2.imread(real_image_path)).to(self.device).float() / 255.0
        fake_img = torch.from_numpy(cv2.imread(fake_image_path)).to(self.device).float() / 255.0
        
        # 在GPU上计算直方图
        real_hist = torch.histc(real_img, bins=256, min=0, max=1)
        fake_hist = torch.histc(fake_img, bins=256, min=0, max=1)
        
        # 归一化
        real_hist = real_hist / (real_hist.sum() + 1e-10)
        fake_hist = fake_hist / (fake_hist.sum() + 1e-10)
        
        # 计算KL散度
        kl_div = torch.sum(real_hist * torch.log((real_hist + 1e-10) / (fake_hist + 1e-10))).item()
        return kl_div

    def evaluate(self, real_image_path, fake_image_path):
        """评估生成图像质量"""
        fcn_score = self.compute_fcn_score(real_image_path, fake_image_path)
        kl_score = self.compare_histograms(real_image_path, fake_image_path)
        
        return {
            'fcn_iou_score': fcn_score,
            'kl_divergence': kl_score
        } 