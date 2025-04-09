import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


class ImageProcessor:
    def __init__(self, size=256):
        # GPU will be faster
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # This transform seems to work well
        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def load_image(self, image_path):
        """Load and preprocess image"""
        image = Image.open(image_path)
        image_tensor = self.transform(image)
        return image_tensor.unsqueeze(0).to(self.device)  # Add batch dimension and move to GPU

    def save_image(self, tensor, save_path):
        """Convert output tensor back to image and save"""
        # Convert values from [-1,1] back to [0,1], model output uses tanh activation
        tensor = (tensor + 1) / 2.0
        # Convert tensor to numpy array
        image = tensor.squeeze(0).detach().cpu().numpy()
        # Change channel order from CxHxW to HxWxC
        image = np.transpose(image, (1, 2, 0))
        # Ensure values are in [0,1] range, just in case
        image = np.clip(image, 0, 1)
        # Convert to 0-255 uint8 format
        image = (image * 255).astype(np.uint8)
        # Save image, note the format issue
        Image.fromarray(image).save(save_path)

    def load_image_pair(self, condition_image_path, real_image_path):
        """Load condition image and real image, return tuple containing these two images"""
        condition_image = self.load_image(condition_image_path)
        real_image = self.load_image(real_image_path)
        return condition_image, real_image
    