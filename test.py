import os
# Environment variable settings - Fix annoying OpenMP warning
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import entropy
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import datetime

from simple_cnn_model import SimpleCNNEdge2Image
from utils import EdgeImageDataset, save_samples


def calculate_kl_divergence(p, q):
    """Calculate KL divergence - Measure the difference between two distributions"""
    # Convert to probability distributions
    p = F.softmax(p.view(p.size(0), -1), dim=1).cpu().numpy()
    q = F.softmax(q.view(q.size(0), -1), dim=1).cpu().numpy()
    
    # Calculate average KL divergence
    kl_vals = [entropy(p[i], q[i]) for i in range(p.shape[0])]
    return np.mean(kl_vals)


def calculate_iou(pred, target, num_classes=21):
    """Calculate IoU score
    
    Parameters:
        pred: Predicted class labels [B, H, W]
        target: Target class labels [B, H, W]
        num_classes: Number of classes (FCN-ResNet50 default is 21)
    
    Returns:
        mean_iou: Mean IoU across all classes
    """
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)
    
    # Calculate IoU for each class
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        
        # Calculate intersection
        intersection = (pred_inds & target_inds).sum().float()
        # Calculate union
        union = (pred_inds | target_inds).sum().float()
        
        if union > 0:  # Avoid division by zero
            iou = intersection / union
            ious.append(iou.item())
            
    # Return mean IoU of all valid classes
    return np.mean(ious) if ious else 0.0


def calculate_fcn_score(real_imgs, fake_imgs, fcn_model):
    """Calculate FCN score using IoU
    
    Parameters:
        real_imgs: Real images [B, C, H, W]
        fake_imgs: Generated images [B, C, H, W]
        fcn_model: Pretrained FCN model
    
    Returns:
        fcn_score: IoU score (range 0-1)
    """
    with torch.no_grad():
        # Get FCN predictions
        real_pred = fcn_model(real_imgs)['out']  # [B, num_classes, H, W]
        fake_pred = fcn_model(fake_imgs)['out']  # [B, num_classes, H, W]
        
        # Get class predictions (argmax)
        real_labels = torch.argmax(real_pred, dim=1)  # [B, H, W]
        fake_labels = torch.argmax(fake_pred, dim=1)  # [B, H, W]
        
        # Calculate IoU score
        iou_score = calculate_iou(fake_labels, real_labels)
        
        return iou_score


def make_grid_image(tensor, normalize=False):
    """Process image for TensorBoard display"""
    # Process on CPU
    tensor = tensor.cpu().detach()
    
    # Normalize to 0-1 range
    if normalize:
        tensor = (tensor * 0.5 + 0.5).clamp(0, 1)
    
    # Just take the first image
    return tensor[0]


def test_model(args):
    """Test model performance and calculate quality metrics"""
    # Basic setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create TensorBoard logger
    log_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = os.path.join(args.log_dir, f"test_{log_time}")
    tb_writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs: {log_dir}")

    # Data preprocessing
    img_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load test data
    test_dataset = EdgeImageDataset(
        root_dir=args.test_dir,
        mode='test',
        transform=img_transform
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    # Create model
    model = SimpleCNNEdge2Image(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        ngf=args.ngf
    ).to(device)

    # Load trained model
    if os.path.exists(args.model_path):
        try:
            ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
            # If complete checkpoint
            if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                model.load_state_dict(ckpt['model_state_dict'])
            else:
                # If just state dict
                model.load_state_dict(ckpt)
            print(f"Model loaded successfully: {args.model_path}")
        except Exception as e:
            print(f"Model loading failed: {e}")
            return
    else:
        print(f"Model file doesn't exist: {args.model_path}")
        return

    # Load FCN model
    print("Loading FCN model...")
    fcn_model = torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet50', pretrained=True)
    fcn_model = fcn_model.to(device)
    fcn_model.eval()

    # Initialize test variables
    model.eval()
    total_kl = 0.0
    total_fcn = 0.0
    fcn_scores = []
    kl_divs = []
    
    # Start testing
    print("Starting test...")
    with torch.no_grad():
        for idx, (edge, real) in enumerate(tqdm(test_loader)):
            # Prepare data
            edge = edge.to(device)
            real = real.to(device)
            
            # Generate images
            fake = model(edge)
            
            # Calculate evaluation metrics
            kl_val = calculate_kl_divergence(real, fake)
            fcn_val = calculate_fcn_score(real, fake, fcn_model)
            
            # Accumulate metrics
            total_kl += kl_val
            total_fcn += fcn_val
            kl_divs.append(kl_val)
            fcn_scores.append(fcn_val)
            
            # Record to TensorBoard
            tb_writer.add_scalar('Metrics/KL_Divergence', kl_val, idx)
            tb_writer.add_scalar('Metrics/FCN_Score', fcn_val, idx)
            
            # Save samples and add to TensorBoard
            if idx % args.sample_freq == 0:
                # TensorBoard visualization
                real_img = make_grid_image(real, normalize=True)
                fake_img = make_grid_image(fake, normalize=True)
                edge_img = make_grid_image(edge, normalize=True)
                
                tb_writer.add_image(f'Images/Real_{idx}', real_img, idx)
                tb_writer.add_image(f'Images/Generated_{idx}', fake_img, idx)
                tb_writer.add_image(f'Images/Edge_{idx}', edge_img, idx)
                
                # Create and save comparison image
                def to_numpy(t):
                    return (t.cpu().detach() * 0.5 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).numpy()
                
                # Convert to numpy
                edge_np = to_numpy(edge)
                real_np = to_numpy(real)
                fake_np = to_numpy(fake)
                
                # Create plot
                fig, ax = plt.subplots(1, 3, figsize=(12, 4))
                
                # Plot edge image
                if edge_np.shape[-1] == 1:
                    ax[0].imshow(edge_np[0, :, :, 0], cmap='gray')
                else:
                    ax[0].imshow(edge_np[0])
                ax[0].set_title('Edge Image')
                ax[0].axis('off')
                
                # Plot real and generated images
                ax[1].imshow(real_np[0])
                ax[1].set_title('Real Image')
                ax[1].axis('off')
                
                ax[2].imshow(fake_np[0])
                ax[2].set_title('Generated Image')
                ax[2].axis('off')
                
                plt.tight_layout()
                save_path = os.path.join(args.output_dir, f"test_sample_{idx}.png")
                plt.savefig(save_path)
                plt.close()
                print(f"Sample saved: {save_path}")
    
    # Calculate average scores
    test_count = len(test_loader)
    avg_kl = total_kl / test_count
    avg_fcn = total_fcn / test_count
    
    # Print results
    print("\nTest Results:")
    print(f"Average KL divergence: {avg_kl:.4f}")
    print(f"Average FCN score: {avg_fcn:.4f} (range 0-1, higher is better)")
    
    # Record average scores to TensorBoard
    tb_writer.add_scalar('Metrics/Average_KL_Divergence', avg_kl, 0)
    tb_writer.add_scalar('Metrics/Average_FCN_Score', avg_fcn, 0)
    
    # Plot distribution graphs
    plt.figure(figsize=(12, 5))
    
    # KL divergence distribution
    plt.subplot(1, 2, 1)
    plt.hist(kl_divs, bins=min(20, len(kl_divs)), alpha=0.7, color='skyblue')
    plt.axvline(avg_kl, color='crimson', linestyle='dashed', linewidth=2)
    plt.title(f'KL Divergence Distribution (Avg: {avg_kl:.4f})')
    plt.xlabel('KL Divergence')
    plt.ylabel('Frequency')
    
    # FCN score distribution
    plt.subplot(1, 2, 2)
    plt.hist(fcn_scores, bins=min(20, len(fcn_scores)), alpha=0.7, color='lightgreen')
    plt.axvline(avg_fcn, color='crimson', linestyle='dashed', linewidth=2)
    plt.title(f'FCN Score Distribution (Avg: {avg_fcn:.4f})')
    plt.xlabel('FCN Score')
    plt.ylabel('Frequency')
    
    # Save distribution plot
    plt.tight_layout()
    dist_path = os.path.join(args.output_dir, 'evaluation_metrics.png')
    plt.savefig(dist_path)
    
    # Add to TensorBoard
    tb_writer.add_figure('Metrics/Distributions', plt.gcf(), 0)
    plt.close()
    
    print(f"Evaluation metric distributions saved: {dist_path}")
    print(f"Testing complete! View results: tensorboard --logdir={args.log_dir}")
    
    # Close TensorBoard
    tb_writer.close()


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="CNN edge map generation model test script")
    
    # Data parameters
    parser.add_argument("--test_dir", type=str, default="./processed/test",
                        help="Test data directory")
    parser.add_argument("--output_dir", type=str, default="test_results",
                        help="Results save directory")
    parser.add_argument("--image_size", type=int, default=256,
                        help="Image size")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Data loading thread count")
    
    # Model parameters
    parser.add_argument("--model_path", type=str, 
                        default="checkpoints/final_model.pth",
                        help="Model path")
    parser.add_argument("--in_channels", type=int, default=1,
                        help="Input channel count")
    parser.add_argument("--out_channels", type=int, default=3,
                        help="Output channel count")
    parser.add_argument("--ngf", type=int, default=64,
                        help="Feature map count")
    
    # Test parameters
    parser.add_argument("--sample_freq", type=int, default=10,
                        help="Save sample frequency")
    
    # TensorBoard parameters
    parser.add_argument("--log_dir", type=str, default="runs",
                        help="TensorBoard log directory")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Run test
    test_model(args) 