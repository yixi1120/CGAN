import os
# Environment variable settings - Fix annoying OpenMP warning
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
from scipy.stats import entropy
from torch.utils.tensorboard import SummaryWriter
import datetime

from simple_cnn_model import SimpleCNNEdge2Image
from utils import EdgeImageDataset, save_checkpoint, save_samples


def calculate_kl_divergence(p, q):
    """Calculate KL divergence - Measure the difference between two distributions"""
    # Convert to probability distributions
    p = F.softmax(p.view(p.size(0), -1), dim=1).cpu().numpy()
    q = F.softmax(q.view(q.size(0), -1), dim=1).cpu().numpy()
    
    # Calculate average KL divergence
    kl_vals = [entropy(p[i], q[i]) for i in range(p.shape[0])]
    return np.mean(kl_vals)


def calculate_fcn_score(real_imgs, fake_imgs, fcn_model):
    """Calculate FCN score"""
    with torch.no_grad():
        # Extract features
        real_feats = fcn_model(real_imgs)['out']
        fake_feats = fcn_model(fake_imgs)['out']
        
        # Calculate L1 distance in feature space
        fcn_score = F.l1_loss(fake_feats, real_feats)
    return fcn_score.item()


def evaluate_model(model, val_loader, device, fcn_model=None):
    """Evaluate model performance"""
    model.eval()
    total_kl = 0.0
    total_fcn = 0.0
    batch_count = 0
    
    with torch.no_grad():
        for edge, real in val_loader:
            # Prepare data
            edge = edge.to(device)
            real = real.to(device)
            
            # Generate images
            fake = model(edge)
            
            # Calculate KL divergence
            kl_val = calculate_kl_divergence(real, fake)
            total_kl += kl_val
            
            # If FCN model is available, calculate FCN score
            if fcn_model is not None:
                fcn_val = calculate_fcn_score(real, fake, fcn_model)
                total_fcn += fcn_val
            
            batch_count += 1
    
    # Calculate average scores
    avg_kl = total_kl / batch_count
    metrics = {'kl_divergence': avg_kl}
    
    if fcn_model is not None:
        avg_fcn = total_fcn / batch_count
        metrics['fcn_score'] = avg_fcn
    
    return metrics


def make_grid_image(tensor, normalize=False):
    """Process image for TensorBoard display"""
    # Process on CPU
    tensor = tensor.cpu().detach()
    
    # Normalize to 0-1 range
    if normalize:
        tensor = (tensor * 0.5 + 0.5).clamp(0, 1)
    
    # Just take the first image
    return tensor[0]


def train(args):
    """Train edge-to-image CNN model"""
    # Basic setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create TensorBoard logger
    log_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = os.path.join(args.log_dir, f"train_{log_time}")
    tb_writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs: {log_dir}")

    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)

    # Data preprocessing
    img_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1,1]
    ])

    # Load training data
    train_dataset = EdgeImageDataset(
        root_dir=args.data_dir,
        mode='train',
        transform=img_transform
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    # Load validation data (if available)
    val_dataset = None
    val_loader = None
    if args.val_dir:
        val_dataset = EdgeImageDataset(
            root_dir=args.val_dir,
            mode='val',
            transform=img_transform
        )
        val_loader = DataLoader(
            val_dataset,
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
    
    # Model information
    print(model)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Choose loss function - L1(MAE) or MSE
    criterion = nn.L1Loss() if args.loss == 'l1' else nn.MSELoss()
    print(f"Loss function: {args.loss.upper()}")
    
    # Setup optimizer - Adam
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    print(f"Optimizer: Adam (lr={args.lr})")
    
    # Learning rate scheduler
    scheduler = None
    if args.lr_decay:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=args.lr_decay_epochs, 
            gamma=args.lr_decay_rate
        )
        print(f"Using LR scheduler: decay by {args.lr_decay_rate} every {args.lr_decay_epochs} epochs")

    # Load pretrained FCN (for evaluation)
    fcn_model = None
    if args.fcn_model_path and os.path.exists(args.fcn_model_path):
        print(f"Loading FCN model: {args.fcn_model_path}")
        fcn_model = torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet50', pretrained=True)
        fcn_model = fcn_model.to(device)
        fcn_model.eval()

    # Start training
    print(f"\n{'='*20} TRAINING START {'='*20}")
    print("Using simple CNN approach - training with reconstruction loss only, no discriminator")
    
    step_count = 0
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        # Training mode
        model.train()
        epoch_start = time.time()
        epoch_loss = 0.0
        
        for batch_idx, (edge, real) in enumerate(train_loader):
            # Prepare data
            edge = edge.to(device)
            real = real.to(device)
            
            # Reset gradients
            optimizer.zero_grad()
            
            # Forward pass
            fake = model(edge)
            
            # Calculate loss
            loss = criterion(fake, real)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track loss
            epoch_loss += loss.item()
            
            # Record to TensorBoard
            tb_writer.add_scalar('loss/train', loss.item(), step_count)
            
            # Record learning rate
            cur_lr = scheduler.get_last_lr()[0] if scheduler else args.lr
            tb_writer.add_scalar('learning rate', cur_lr, step_count)
            
            # Print progress
            if batch_idx % 10 == 0:  # Fixed printing frequency
                print(f"Epoch [{epoch+1}/{args.epochs}] | "
                      f"Batch [{batch_idx}/{len(train_loader)}] | "
                      f"Loss: {loss.item():.4f}")
            
            # Save generated samples
            if step_count % args.sample_freq == 0:
                # Save images
                sample_path = os.path.join(args.sample_dir, f"sample_{step_count}.png")
                save_samples(edge, real, fake, sample_path)
                print(f"Sample saved: {sample_path}")
                
                # TensorBoard visualization
                tb_writer.add_image('Edges', make_grid_image(edge, normalize=True), step_count)
                tb_writer.add_image('Real', make_grid_image(real, normalize=True), step_count)
                tb_writer.add_image('Generated', make_grid_image(fake, normalize=True), step_count)

            step_count += 1
        
        # Update learning rate
        if scheduler:
            scheduler.step()
        
        # Calculate average loss
        avg_epoch_loss = epoch_loss / len(train_loader)
        epoch_time = time.time() - epoch_start
        
        # Validation step if validation data is available
        val_metrics = {}
        if val_loader is not None:
            val_metrics = evaluate_model(model, val_loader, device, fcn_model)
            
            # Log validation metrics
            for metric_name, metric_value in val_metrics.items():
                tb_writer.add_scalar(f'Validation/{metric_name}', metric_value, epoch)
        
        # Validation evaluation
        if val_loader:
            print("Executing validation...")
            metrics = evaluate_model(model, val_loader, device, fcn_model)
            
            kl_val = metrics['kl_divergence']
            print(f"KL divergence: {kl_val:.4f}")
            
            if 'fcn_score' in metrics:
                fcn_val = metrics['fcn_score']
                print(f"FCN score: {fcn_val:.4f}")
            
            # Record validation metrics
            tb_writer.add_scalar('Validation/KL divergence', kl_val, epoch)
            if 'fcn_score' in metrics:
                tb_writer.add_scalar('Validation/FCN score', fcn_val, epoch)
            
            # Save best model
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                save_checkpoint(
                    model, 
                    optimizer, 
                    epoch, 
                    os.path.join(args.checkpoint_dir, "best_model.pth")
                )
                print("Best model saved!")
        
        # Save checkpoint every few epochs
        if (epoch + 1) % args.save_freq == 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f"checkpoint_{epoch+1}.pth")
            save_checkpoint(model, optimizer, epoch, ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")
    
    # Save final model
    final_path = os.path.join(args.checkpoint_dir, "final_model.pth")
    save_checkpoint(model, optimizer, args.epochs, final_path)
    print(f"Final model saved: {final_path}")
    
    # Add model graph to TensorBoard
    dummy_input = torch.randn(1, args.in_channels, args.image_size, args.image_size).to(device)
    tb_writer.add_graph(model, dummy_input)
    
    # Close TensorBoard
    tb_writer.close()
    
    print(f"Training completed! Use the following command to view the training process:\ntensorboard --logdir={args.log_dir}")
    print("\nModel summary:")
    print("- Simple edge-to-image CNN")
    print("- Trained with reconstruction loss, no adversarial loss")
    print("- Faster training, simpler implementation, but may lack detail compared to cGAN")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="CNN Edge-to-Image Training Script")
    
    # Data parameters
    parser.add_argument("--data_dir", type=str, default="./processed/train",
                        help="Training data directory")
    parser.add_argument("--val_dir", type=str, default="",
                        help="Validation data directory (optional)")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory to save model checkpoints")
    parser.add_argument("--sample_dir", type=str, default="samples",
                        help="Directory to save generated samples")
    parser.add_argument("--image_size", type=int, default=256,
                        help="Image size")
    parser.add_argument("--in_channels", type=int, default=1,
                        help="Input channels (1 for grayscale edge images)")
    parser.add_argument("--out_channels", type=int, default=3,
                        help="Output channels (3 for RGB images)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Training batch size")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    
    # Model parameters
    parser.add_argument("--ngf", type=int, default=64,
                        help="Number of generator filters")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="Learning rate")
    parser.add_argument("--loss", type=str, default="l1", choices=["l1", "mse"],
                        help="Loss function (l1 or mse)")
    parser.add_argument("--lr_decay", action="store_true",
                        help="Enable learning rate decay")
    parser.add_argument("--lr_decay_epochs", type=int, default=30,
                        help="LR decay step size (epochs)")
    parser.add_argument("--lr_decay_rate", type=float, default=0.5,
                        help="LR decay rate")
    
    # Logging and saving parameters
    parser.add_argument("--log_freq", type=int, default=10,
                        help="Logging frequency (steps)")
    parser.add_argument("--sample_freq", type=int, default=50,
                        help="Sample generation frequency (steps)")
    parser.add_argument("--save_freq", type=int, default=10,
                        help="Checkpoint saving frequency (epochs)")
    parser.add_argument("--log_dir", type=str, default="runs",
                        help="TensorBoard log directory")
    
    # FCN model for evaluation
    parser.add_argument("--fcn_model_path", type=str, default="",
                        help="Path to pretrained FCN model (for evaluation)")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run training
    train(args) 