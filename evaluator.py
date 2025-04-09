import torch
import cv2
import numpy as np
from torchvision import models, transforms
from PIL import Image

class ImageEvaluator:
    def __init__(self):
        # Using GPU is much faster, use CPU if no GPU available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load pre-trained FCN model - resnet50 backbone should be sufficient
        self.fcn_model = models.segmentation.fcn_resnet50().eval().to(self.device)
        
        # Image preprocessing transform - 256 size is suitable
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

    def preprocess_image(self, image_path):
        """Preprocess image"""
        image = Image.open(image_path)
        return self.transform(image).unsqueeze(0).to(self.device)

    @torch.amp.autocast('cuda')  # Using mixed precision for speed
    def compute_fcn_score(self, real_image_path, fake_image_path):
        """Compute FCN semantic segmentation score"""
        real_img = self.preprocess_image(real_image_path)
        fake_img = self.preprocess_image(fake_image_path)

        with torch.no_grad():  # No need for gradient calculation, saves memory
            real_output = self.fcn_model(real_img)['out'].argmax(1)
            fake_output = self.fcn_model(fake_img)['out'].argmax(1)
            
            # Compute IoU on GPU - should be faster
            real_mask = (real_output > 0)
            fake_mask = (fake_output > 0)
            intersection = torch.logical_and(real_mask, fake_mask).sum().float()
            union = real_mask.sum().float() + 1e-10  # Add small value to prevent division by zero
            iou = (intersection / union).item()

        return iou

    def compare_histograms(self, real_image_path, fake_image_path):
        """Calculate color histogram difference"""
        # Using CV2 to read images may be more efficient
        real_img = torch.from_numpy(cv2.imread(real_image_path)).to(self.device).float() / 255.0
        fake_img = torch.from_numpy(cv2.imread(fake_image_path)).to(self.device).float() / 255.0
        
        # Compute histograms on GPU - 256 bins seems sufficient
        real_hist = torch.histc(real_img, bins=256, min=0, max=1)
        fake_hist = torch.histc(fake_img, bins=256, min=0, max=1)
        
        # Normalize - ensure sum of probabilities is 1
        real_hist = real_hist / (real_hist.sum() + 1e-10)
        fake_hist = fake_hist / (fake_hist.sum() + 1e-10)
        
        # Calculate KL divergence - standard formula
        kl_div = torch.sum(real_hist * torch.log((real_hist + 1e-10) / (fake_hist + 1e-10))).item()
        return kl_div

    def evaluate(self, real_image_path, fake_image_path):
        """Evaluate generated image quality"""
        fcn_score = self.compute_fcn_score(real_image_path, fake_image_path)
        kl_score = self.compare_histograms(real_image_path, fake_image_path)
        
        return {
            'fcn_iou_score': fcn_score,
            'kl_divergence': kl_score
        } 