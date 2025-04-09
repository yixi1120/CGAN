import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from generator import UNetGenerator
from evaluator import ImageEvaluator
from utils import ImageProcessor
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
import time  


# Some temporary helper functions
def get_filename(path):
    """Get filename without path and extension"""
    basename = os.path.basename(path)
    filename, _ = os.path.splitext(basename)
    return filename


def is_image_file(filename):
    """Check if file is an image"""
    exts = ['.png', '.jpg', '.jpeg', '.bmp']
    return any(filename.lower().endswith(ext) for ext in exts)


def create_comparison_image(real_img_path, gen_img_path, save_path):
    """Create comparison image by horizontally concatenating real and generated images"""
    real_img = Image.open(real_img_path)
    gen_img = Image.open(gen_img_path)
    
    # Ensure both images have the same dimensions
    width, height = real_img.size
    if gen_img.size != real_img.size:
        gen_img = gen_img.resize((width, height))
    
    # Create new image with twice the width and same height
    comparison = Image.new('RGB', (width * 2, height))
    
    # Paste both images
    comparison.paste(real_img, (0, 0))
    comparison.paste(gen_img, (width, 0))
    
    # Save comparison image
    comparison.save(save_path)


def test_model(generator_path, condition_dir, real_dir, output_dir, image_size=256):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    print(f"Using device: {device}")
    start_time = time.time()  
    
    # Load generator model
    generator = UNetGenerator(input_channels=3, output_channels=3).to(device)
    # Load checkpoint
    checkpoint = torch.load(generator_path, map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()  # Set to evaluation mode
    

    model_size = sum(p.numel() for p in generator.parameters())
    print(f"Model parameter count: {model_size}")
    
    # Initialize evaluator and image processor
    evaluator = ImageEvaluator()
    img_proc = ImageProcessor(size=image_size)  # Renamed to avoid confusion with built-in processor
    
    # Create TensorBoard writer
    tb_dir = os.path.join(output_dir, "tensorboard")
    os.makedirs(tb_dir, exist_ok=True)
    tb_writer = SummaryWriter(tb_dir)
    
    # Image transforms
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    comparison_dir = os.path.join(output_dir, "comparison_images")
    os.makedirs(comparison_dir, exist_ok=True)
    
    results = []
    
    # Get file list
    files = os.listdir(condition_dir)
    
    
    total_files = len(files)
    processed_files = 0
    skipped_files = 0
    
    # Process each image
    for idx, fname in enumerate(files):
        # Skip non-image files
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            skipped_files += 1
            continue
            
        cond_path = os.path.join(condition_dir, fname)
        real_path = os.path.join(real_dir, fname)
        
        # Check if real image exists
        if not os.path.exists(real_path):
            print(f"Skipping {fname}: No corresponding real image found")
            skipped_files += 1
            continue
        
        
        
        pure_name = get_filename(fname)
        
        # Load condition image
        cond_img = Image.open(cond_path).convert("RGB")
        
        
        img_width, img_height = cond_img.size
        
        cond_tensor = transform(cond_img).unsqueeze(0).to(device)
        
        # Generate image (inference phase)
        with torch.no_grad():
            gen_img = generator(cond_tensor)
        
        # Save generated image
        out_path = os.path.join(output_dir, f"gen_{fname}")
        # Convert image from [-1,1] to [0,1] range and save
        save_image((gen_img + 1) / 2, out_path)
        
        # Create and save comparison image
        comparison_path = os.path.join(comparison_dir, f"comparison_{fname}")
        create_comparison_image(real_path, out_path, comparison_path)
        
        # Output to TensorBoard
        tb_writer.add_image(f"Images/Cond_{fname}",
                            (cond_tensor + 1) / 2, idx, dataformats='NCHW')
        tb_writer.add_image(f"Images/Generated_{fname}",
                            (gen_img + 1) / 2, idx, dataformats='NCHW')
        
        # Evaluate generated image
        metrics = evaluator.evaluate(real_path, out_path)
        metrics['filename'] = fname
        metrics['original_size'] = (img_width, img_height)  
        results.append(metrics)
        
        # Record evaluation metrics
        fcn_iou = metrics['fcn_iou_score']
        kl_div = metrics['kl_divergence']
        
        # Add to TensorBoard
        tb_writer.add_scalar(f"Test/FCN_{fname}", fcn_iou, idx)
        tb_writer.add_scalar(f"Test/KL_{fname}", kl_div, idx)
        
        # Print progress
        processed_files += 1
        print(f"Processing: {fname} | FCN={fcn_iou:.4f}, KL={kl_div:.4f}")
        print(f"Comparison image saved to: {comparison_path}")
        
        if idx % 5 == 0:
            elapsed = time.time() - start_time
            print(f"Processed {processed_files}/{total_files} files, time elapsed: {elapsed:.2f} seconds")
    
    # Calculate average metrics
    avg_fcn = np.mean([r['fcn_iou_score'] for r in results])
    avg_kl = np.mean([r['kl_divergence'] for r in results])
    

    min_fcn = min([r['fcn_iou_score'] for r in results]) if results else 0
    max_fcn = max([r['fcn_iou_score'] for r in results]) if results else 0
    std_fcn = np.std([r['fcn_iou_score'] for r in results]) if results else 0
    
    # Record average metrics to TensorBoard
    tb_writer.add_scalar("Test/Average_FCN", avg_fcn, 0)
    tb_writer.add_scalar("Test/Average_KL", avg_kl, 0)
    
    # Add histograms
    tb_writer.add_histogram("Distributions/FCN", 
                         torch.tensor([r['fcn_iou_score'] for r in results]), 0)
    tb_writer.add_histogram("Distributions/KL", 
                         torch.tensor([r['kl_divergence'] for r in results]), 0)
    
    tb_writer.close()
    
    # Print summary
    total_time = time.time() - start_time
    print(f"\nTest complete! FCN_IoU: {avg_fcn:.4f}, KL Divergence: {avg_kl:.4f}")
    print(f"FCN Statistics: Min={min_fcn:.4f}, Max={max_fcn:.4f}, StdDev={std_fcn:.4f}")
    print(f"Processed {processed_files} files, skipped {skipped_files} files, total time: {total_time:.2f} seconds")
    
    
    def summarize_results(results_list):
        """Analyze test results and generate summary report"""
        if not results_list:
            return "No results available for analysis"
            
        summary = {
            "Total samples": len(results_list),
            "Average FCN": np.mean([r['fcn_iou_score'] for r in results_list]),
            "Average KL": np.mean([r['kl_divergence'] for r in results_list]),
        }
        return summary
    
    return results


if __name__ == "__main__":
    # Configure paths
    chkpt_path = "./output/generated_images/final_checkpoint_epoch_5000.pt"
    cond_dir = "./data/processed/test/condition_images"
    real_dir = "./data/processed/test/real_images"
    out_dir = "./output/evaluation_results"
    
    
    options = {
        "save_intermediate": True,
        "verbose_output": True, 
        "use_cuda": torch.cuda.is_available()
    }
    
    # Start testing
    test_model(chkpt_path, cond_dir, real_dir, out_dir)