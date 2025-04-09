import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from src.generator import UNetGenerator
from src.discriminator import Discriminator
from src.utils import ImageProcessor
from src.evaluator import ImageEvaluator
from src.precon import ImagePreprocessor

from torch.utils.tensorboard import SummaryWriter

def load_checkpoint_if_exists(checkpoint_dir):
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
    
    def extract_epoch_number(filename):
        try:
            return int(filename.split("_epoch_")[-1].split(".pt")[0])
        except Exception:
            return 0

    checkpoint_files = sorted(checkpoint_files, key=extract_epoch_number)
    if checkpoint_files:
        latest_checkpoint = os.path.join(checkpoint_dir, checkpoint_files[-1])
        return latest_checkpoint
    else:
        return None

class ImagePairDataset(Dataset):
    def __init__(self, condition_dir, real_dir, transform=None):
        self.condition_dir = condition_dir
        self.real_dir = real_dir
        self.transform = transform
        self.image_files = []
        
        # Get common filenames from both directories
        condition_files = set(os.listdir(condition_dir))
        real_files = set(os.listdir(real_dir))
        common_files = list(condition_files.intersection(real_files))
        
        for file_name in common_files:
            if file_name.lower().endswith((".png", ".jpg", ".jpeg")):
                self.image_files.append(file_name)
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        file_name = self.image_files[idx]
        condition_path = os.path.join(self.condition_dir, file_name)
        real_path = os.path.join(self.real_dir, file_name)
        
        condition_img = Image.open(condition_path).convert("RGB")
        real_img = Image.open(real_path).convert("RGB")
        
        if self.transform:
            condition_img = self.transform(condition_img)
            real_img = self.transform(real_img)
            
        return condition_img, real_img
    
class GANTrainer:
    def __init__(self, condition_dir, real_dir, output_dir, image_size=256, batch_size=4, checkpoint_file=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.checkpoint_file = checkpoint_file  # Save as class attribute
        
        # Create SummaryWriter making sure not to overwrite existing logs
        log_dir = os.path.join(output_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir, purge_step=None)  # purge_step=None ensures not overwriting previous records
    
        self.generator = UNetGenerator(input_channels=3, output_channels=3).to(self.device)
        self.discriminator = Discriminator(input_channels=3).to(self.device)
    
        self.processor = ImageProcessor(size=image_size)
        self.evaluator = ImageEvaluator()
    
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
        self.dataset = ImagePairDataset(condition_dir, real_dir, transform=self.transform)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True
        )
    
        # Modify label size to match discriminator output
        current_batch_size = batch_size
        self.real_label = torch.ones(current_batch_size, 1, 15, 15).to(self.device)
        self.fake_label = torch.zeros(current_batch_size, 1, 15, 15).to(self.device)
        
        self.g_optimizer = Adam(self.generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
        self.d_optimizer = Adam(self.discriminator.parameters(), lr=0.00005, betas=(0.5, 0.999))
        
        self.g_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.g_optimizer, mode='min', factor=0.5, patience=10, verbose=False
        )
        self.d_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.d_optimizer, mode='min', factor=0.5, patience=10, verbose=False
        )
        
        self.criterion = nn.BCEWithLogitsLoss()
        
        self.scaler = torch.amp.GradScaler('cuda')
        
        self.start_epoch = 0
        checkpoint = self.safe_load_checkpoint(checkpoint_file)
        if checkpoint is not None:
            self.generator.load_state_dict(checkpoint['generator_state_dict'])
            self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
            self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
            self.start_epoch = checkpoint.get('epoch', 0)
            print(f"Resuming training from epoch {self.start_epoch}.")
        else:
            print("No valid checkpoint found. Starting from scratch.")
    
    def safe_load_checkpoint(self, checkpoint_file):
        if not checkpoint_file or not os.path.isfile(checkpoint_file):
            print("Checkpoint file not found. Starting from scratch.")
            return None
        try:
            checkpoint = torch.load(checkpoint_file, map_location=self.device, weights_only=False)
            print(f"Successfully loaded checkpoint: {checkpoint_file}")
            return checkpoint
        except Exception as e:
            print(f"Failed to load checkpoint '{checkpoint_file}': {e}")
            return None
    
    def train(self, num_epochs=1000):
        # Restore global step from checkpoint
        global_step = 0
        checkpoint = self.safe_load_checkpoint(self.checkpoint_file)
        if checkpoint is not None and 'global_step' in checkpoint:
            global_step = checkpoint['global_step']
            print(f"Resuming from global step: {global_step}")
            
        torch.backends.cudnn.benchmark = True
        
        for epoch in range(self.start_epoch, num_epochs):
            self.generator.train()
            self.discriminator.train()
            
            for i, (condition_images, real_images) in enumerate(self.dataloader):
                current_batch_size = condition_images.size(0)
                condition_images = condition_images.to(self.device, non_blocking=True)
                real_images = real_images.to(self.device, non_blocking=True)
                
                # Train discriminator
                self.d_optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type='cuda'):
                    d_real = self.discriminator(real_images)
                    d_loss_real = self.criterion(d_real, self.real_label[:current_batch_size])
                    
                    # Generate fake images with generator
                    with torch.no_grad():
                        fake_images = self.generator(condition_images)  # Only using condition images
                    d_fake = self.discriminator(fake_images.detach())
                    d_loss_fake = self.criterion(d_fake, self.fake_label[:current_batch_size])
                    
                    d_loss = (d_loss_real + d_loss_fake) * 0.5
                
                if d_loss.item() > 0.5:
                    d_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=0.5)
                    self.d_optimizer.step()
                
                # Train generator (twice per batch to increase training frequency)
                for _ in range(2):  # Train generator twice
                    self.g_optimizer.zero_grad(set_to_none=True)
                    with torch.amp.autocast(device_type='cuda'):
                        fake_images = self.generator(condition_images)  # Only using condition images
                        g_fake = self.discriminator(fake_images)
                        g_loss = self.criterion(g_fake, self.real_label[:current_batch_size])
                        
                        # Add L1 loss to encourage generator to produce images similar to real ones
                        l1_loss = torch.mean(torch.abs(fake_images - real_images))
                        g_loss = g_loss + 100 * l1_loss  # Weight coefficient can be adjusted
                    
                    g_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=0.5)
                    self.g_optimizer.step()
                
                if i % 100 == 0:
                    print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(self.dataloader)}], '
                          f'd_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}')
                
                if global_step % 10 == 0:
                    self.writer.add_scalar('Loss/Discriminator', d_loss.item(), global_step)
                    self.writer.add_scalar('Loss/Generator', g_loss.item(), global_step)
                
                if global_step % 500 == 0:
                    with torch.no_grad():
                        self.writer.add_images('FakeImages', fake_images[:4], global_step=global_step)
                
                global_step += 1
            
            if (epoch + 1) % 200 == 0:
                print(f"--- Epoch {epoch+1} completed ---")
            
            if (epoch + 1) % 500 == 0:
                self.evaluate_and_save(epoch + 1)
                checkpoint_path = os.path.join(self.output_dir, f"checkpoint_epoch_{epoch+1}.pt")
                self.save_checkpoint(checkpoint_path, epoch + 1, global_step)
                self.g_scheduler.step(g_loss)
                self.d_scheduler.step(d_loss)
        
        final_checkpoint_path = os.path.join(self.output_dir, f"final_checkpoint_epoch_{num_epochs}.pt")
        self.save_checkpoint(final_checkpoint_path, num_epochs, global_step)
        self.writer.close()
    
    def save_checkpoint(self, checkpoint_path, epoch, global_step):
        torch.save({
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'global_step': global_step  # Get global_step from parameter
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def evaluate_and_save(self, epoch):
        """Evaluate model and save generated images"""
        self.generator.eval()
        fcn_scores = []
        kl_divs = []
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        with torch.no_grad():
            for i, batch in enumerate(self.dataloader):  # Use training dataloader
                if i >= 5:
                    break
                
                # Print debug info
                print(f"Batch type: {type(batch)}")
                
                # Correctly unpack the batch - it's a tuple or list containing two tensors
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    condition_images, real_images = batch
                else:
                    print(f"Abnormal batch format: {type(batch)}")
                    continue
                    
                condition_images = condition_images.to(self.device)
                real_images = real_images.to(self.device)
                
                # Generate fake images using condition images
                fake_images = self.generator(condition_images)
                
                for j in range(min(condition_images.size(0), 5)):  # Limit to maximum 5 samples per batch
                    real_path = os.path.join(self.output_dir, f"epoch_{epoch}_real_{i}_{j}.png")
                    fake_path = os.path.join(self.output_dir, f"epoch_{epoch}_fake_{i}_{j}.png")
                    cond_path = os.path.join(self.output_dir, f"epoch_{epoch}_cond_{i}_{j}.png")
                    
                    # Save images
                    self.processor.save_image(real_images[j], real_path)
                    self.processor.save_image(fake_images[j], fake_path)
                    self.processor.save_image(condition_images[j], cond_path)
                    
                    # Evaluate generated images
                    try:
                        scores = self.evaluator.evaluate(real_path, fake_path)
                        fcn_score = scores.get('fcn_iou_score', 0.0)
                        kl_div = scores.get('kl_divergence', 0.0)
                        
                        fcn_scores.append(fcn_score)
                        kl_divs.append(kl_div)
                        
                        # Record metrics for each sample to TensorBoard
                        self.writer.add_scalar(f'Samples/FCN_Score_sample_{i}_{j}', fcn_score, epoch)
                        self.writer.add_scalar(f'Samples/KL_Divergence_sample_{i}_{j}', kl_div, epoch)
                        
                        print(f"Epoch {epoch}, Sample {i}_{j}: FCN Score = {fcn_score:.4f}, KL Divergence = {kl_div:.4f}")
                    except Exception as e:
                        print(f"Error during evaluation: {str(e)}")
        
        # Calculate average scores
        avg_fcn_score = sum(fcn_scores) / len(fcn_scores) if fcn_scores else 0.0
        avg_kl_div = sum(kl_divs) / len(kl_divs) if kl_divs else 0.0
        
        # Record to TensorBoard
        self.writer.add_scalar('Metrics/FCN_Score', avg_fcn_score, epoch)
        self.writer.add_scalar('Metrics/KL_Divergence', avg_kl_div, epoch)
        
        # Add all metrics to histograms to see distributions
        self.writer.add_histogram('Distributions/FCN_Score', torch.tensor(fcn_scores), epoch)
        self.writer.add_histogram('Distributions/KL_Divergence', torch.tensor(kl_divs), epoch)
        
        # Return to training mode
        self.generator.train()

if __name__ == "__main__":
    checkpoint_dir = "./checkpoints"  # Changed to relative path
    checkpoint_file = load_checkpoint_if_exists(checkpoint_dir)
    
    # If no checkpoint found in checkpoints directory, try loading from output/generated_images
    if checkpoint_file is None:
        output_dir = "./output/generated_images"
        if os.path.exists(output_dir):
            checkpoint_file = load_checkpoint_if_exists(output_dir)
        if checkpoint_file is None:
            print("Trying to load final checkpoint: final_checkpoint_epoch_5000.pt")
            final_checkpoint = os.path.join(output_dir, "final_checkpoint_epoch_5000.pt")
            if os.path.exists(final_checkpoint):
                checkpoint_file = final_checkpoint
    
    print(f"Using checkpoint file: {checkpoint_file}")
    
    trainer = GANTrainer(
        condition_dir="./data/processed/train/condition_images",  # Condition images directory
        real_dir="./data/processed/train/real_images",            # Real images directory
        output_dir="./output/generated_images",                   # Output directory
        batch_size=4,
        checkpoint_file=checkpoint_file
    )
    trainer.train(num_epochs=5000)  
