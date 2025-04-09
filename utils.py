import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset


class EdgeImageDataset(Dataset):
    """Edge image and real image pairs dataset"""
    
    def __init__(self, root_dir, mode='train', transform=None):
        """
        Parameters:
            root_dir: Data directory containing edge images and real images
            mode: 'train' or 'val' or 'test'
            transform: Image preprocessing transformations
        """
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        
        # Assuming data organization:
        # root_dir/
        #   ├── edges/        # Edge images folder
        #   └── real_images/  # Real images folder
        
        self.edge_dir = os.path.join(root_dir, 'edges')
        self.real_dir = os.path.join(root_dir, 'real_images')
        
        # Get all image filenames (assuming edge and real images have the same filename)
        self.image_files = [
            f for f in os.listdir(self.edge_dir) 
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ]
        
        print(f"Found {len(self.image_files)} edge-real image pairs")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Get image filename
        img_name = self.image_files[idx]
        
        # Build paths for edge and real images
        edge_path = os.path.join(self.edge_dir, img_name)
        real_path = os.path.join(self.real_dir, img_name)
        
        # Read edge image (grayscale)
        edge_img = Image.open(edge_path)
        if edge_img.mode != 'L':  # Ensure it's grayscale
            edge_img = edge_img.convert('L')
        
        # Read real image (color)
        real_img = Image.open(real_path)
        if real_img.mode != 'RGB':  # Ensure it's RGB
            real_img = real_img.convert('RGB')
        
        # Apply transformations
        if self.transform:
            edge_img = self.transform(edge_img)
            real_img = self.transform(real_img)
        
        return edge_img, real_img


def save_checkpoint(model, optimizer, epoch, filepath):
    """Save model checkpoint
    
    Parameters:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        filepath: Save path
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(model, optimizer, filepath, device):
    """Load model checkpoint
    
    Parameters:
        model: Model to load
        optimizer: Optimizer
        filepath: Checkpoint file path
        device: Device ('cuda' or 'cpu')
    
    Returns:
        epoch: Checkpoint epoch
    """
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Loaded checkpoint from {filepath}, epoch: {epoch}")
    return epoch


def save_samples(edges, real_imgs, fake_imgs, filepath, num_samples=4):
    """Save generated sample images comparison
    
    Parameters:
        edges: Input edge images
        real_imgs: Real images
        fake_imgs: Generated images
        filepath: Save path
        num_samples: Number of samples to save
    """
    # Convert images from tensors to numpy arrays
    def tensor_to_numpy(img):
        # Ensure images are on CPU and denormalized
        img = (img.cpu().detach() * 0.5 + 0.5).clamp(0, 1)
        # Move channel dimension to the end
        return img.permute(0, 2, 3, 1).numpy()
    
    edges_np = tensor_to_numpy(edges[:num_samples])
    real_np = tensor_to_numpy(real_imgs[:num_samples])
    fake_np = tensor_to_numpy(fake_imgs[:num_samples])
    
    # Create plot
    fig, axs = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    
    for i in range(num_samples):
        # Display edge image
        if edges_np.shape[-1] == 1:  # Grayscale
            axs[i, 0].imshow(edges_np[i, :, :, 0], cmap='gray')
        else:  # RGB
            axs[i, 0].imshow(edges_np[i])
        axs[i, 0].set_title('Edge Image')
        axs[i, 0].axis('off')
        
        # Display real image
        axs[i, 1].imshow(real_np[i])
        axs[i, 1].set_title('Real Image')
        axs[i, 1].axis('off')
        
        # Display generated image
        axs[i, 2].imshow(fake_np[i])
        axs[i, 2].set_title('Generated Image')
        axs[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    print(f"Samples saved to {filepath}")


def create_custom_dataset(edge_dir, real_dir, output_dir, 
                          num_pairs=1000, size=256):
    """Create custom edge-real image pairs dataset (for testing)
    
    Parameters:
        edge_dir: Directory to save edge images
        real_dir: Directory to save real images
        output_dir: Output directory
        num_pairs: Number of image pairs to generate
        size: Image size
    """
    os.makedirs(edge_dir, exist_ok=True)
    os.makedirs(real_dir, exist_ok=True)
    
    for i in range(num_pairs):
        # Create random edge image (simple example, actual use would need more complex generation)
        edge = np.zeros((size, size), dtype=np.uint8)
        
        # Randomly add some lines and shapes (simplified example)
        num_lines = np.random.randint(5, 15)
        for _ in range(num_lines):
            x1, y1 = np.random.randint(0, size), np.random.randint(0, size)
            x2, y2 = np.random.randint(0, size), np.random.randint(0, size)
            thickness = np.random.randint(1, 4)
            color = 255
            # Draw line
            cv2.line(edge, (x1, y1), (x2, y2), color, thickness)
        
        # Create corresponding "imaginary" real image (color)
        real = np.zeros((size, size, 3), dtype=np.uint8)
        
        # For simple demonstration, we just create some color regions based on edges
        for _ in range(np.random.randint(3, 8)):
            x, y = np.random.randint(0, size), np.random.randint(0, size)
            radius = np.random.randint(20, 80)
            color = (
                np.random.randint(0, 255),
                np.random.randint(0, 255),
                np.random.randint(0, 255)
            )
            cv2.circle(real, (x, y), radius, color, -1)
        
        # Overlay edges onto real image
        edge_rgb = np.stack([edge, edge, edge], axis=2)
        real = np.where(edge_rgb > 0, edge_rgb, real)
        
        # Save images
        edge_img = Image.fromarray(edge)
        real_img = Image.fromarray(real)
        
        edge_img.save(os.path.join(edge_dir, f"{i:04d}.png"))
        real_img.save(os.path.join(real_dir, f"{i:04d}.png"))
    
    print(f"Created {num_pairs} edge-real image pairs dataset")


if __name__ == "__main__":
    # Import cv2 to be able to run create_custom_dataset function
    import cv2
    
    # Test creating custom dataset
    output_dir = "test_dataset"
    edge_dir = os.path.join(output_dir, "edges")
    real_dir = os.path.join(output_dir, "real_images")
    
    create_custom_dataset(edge_dir, real_dir, output_dir, 
                          num_pairs=10, size=256)
    
    # Test dataset class
    transform = None  # Define transformations here if needed
    dataset = EdgeImageDataset(output_dir, transform=transform)
    edge, real = dataset[0]
    
    print(f"Edge image size: {edge.size if isinstance(edge, Image.Image) else edge.shape}")
    print(f"Real image size: {real.size if isinstance(real, Image.Image) else real.shape}") 