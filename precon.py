import os
import sys
# Add project root directory to Python path
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from utils import ImageProcessor
import cv2
import numpy as np
import os
import shutil
from utils import ImageProcessor

def load_image(image_path):
    if not os.path.exists(image_path):
        print(f"File {image_path} does not exist!")
        return None

    image = cv2.imread(image_path)
    if image is None:
        print(f"Cannot read image file: {image_path}")
        return None
    
    return image

def resize_with_padding_cv(image, target_size=(720, 720), fill_color=(0, 0, 0)):
    original_height, original_width = image.shape[:2]
    target_width, target_height = target_size

    scale = min(target_width / original_width, target_height / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

    padded_image = np.full((target_height, target_width, 3), fill_color, dtype=np.uint8)
    left = (target_width - new_width) // 2
    top = (target_height - new_height) // 2
    padded_image[top:top+new_height, left:left+new_width] = resized_image

    return padded_image

def normalize_image(image):
    normalized_image = image.astype(np.float32) / 255.0
    return normalized_image

def save_image(image, output_path, is_normalized=False):
    if is_normalized:
        image = (image * 255).astype(np.uint8)
    cv2.imwrite(output_path, image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    print(f"Image saved to: {output_path}")

class ImagePreprocessor:
    def __init__(self, target_size=256):
        self.target_size = target_size
        self.processor = ImageProcessor(size=target_size)

    def process_images_in_folder(self, input_folder, output_folder):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        processed_files = []
        for file_name in os.listdir(input_folder):
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                input_path = os.path.join(input_folder, file_name)
                output_path = os.path.join(output_folder, f"processed_{file_name}")
                
                try:
                    tensor = self.processor.load_image(input_path)
                    self.processor.save_image(tensor, output_path)
                    processed_files.append((input_path, output_path))
                    print(f"Successfully processed image: {file_name}")
                except Exception as e:
                    print(f"Error processing image {file_name}: {str(e)}")
                    
        return processed_files

def split_dataset(raw_condition_dir, raw_real_dir, processed_dir, train_size=800, random_split=False):
    """
    Split the original dataset into training and test sets
    
    Parameters:
        raw_condition_dir: Original condition images directory
        raw_real_dir: Original real images directory
        processed_dir: Directory for processed data
        train_size: Training set size, default 800
        random_split: Whether to split randomly, default False (sequential split)
    """
    # Create directory structure
    train_condition_dir = os.path.join(processed_dir, 'train/condition_images')
    train_real_dir = os.path.join(processed_dir, 'train/real_images')
    test_condition_dir = os.path.join(processed_dir, 'test/condition_images')
    test_real_dir = os.path.join(processed_dir, 'test/real_images')
    
    for directory in [train_condition_dir, train_real_dir, test_condition_dir, test_real_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Get original image file lists
    condition_files = [f for f in os.listdir(raw_condition_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    real_files = [f for f in os.listdir(raw_real_dir) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    # Find common filenames between the two directories
    common_files = list(set(condition_files).intersection(set(real_files)))
    common_files.sort()  # Sort for consistency
    
    print(f"Found {len(common_files)} pairs of image files")
    
    if random_split:
        # Randomly shuffle file order
        import random
        random.shuffle(common_files)
    
    # Split into training and test sets
    train_files = common_files[:train_size]
    test_files = common_files[train_size:]
    
    print(f"Training set: {len(train_files)} image pairs")
    print(f"Test set: {len(test_files)} image pairs")
    
    # Copy training set files
    for filename in train_files:
        # Copy condition images
        src_condition = os.path.join(raw_condition_dir, filename)
        dst_condition = os.path.join(train_condition_dir, filename)
        shutil.copy2(src_condition, dst_condition)
        
        # Copy real images
        src_real = os.path.join(raw_real_dir, filename)
        dst_real = os.path.join(train_real_dir, filename)
        shutil.copy2(src_real, dst_real)
    
    # Copy test set files
    for filename in test_files:
        # Copy condition images
        src_condition = os.path.join(raw_condition_dir, filename)
        dst_condition = os.path.join(test_condition_dir, filename)
        shutil.copy2(src_condition, dst_condition)
        
        # Copy real images
        src_real = os.path.join(raw_real_dir, filename)
        dst_real = os.path.join(test_real_dir, filename)
        shutil.copy2(src_real, dst_real)
    
    print("Dataset split complete!")
    return {
        'train_size': len(train_files),
        'test_size': len(test_files),
        'train_dir': {'condition': train_condition_dir, 'real': train_real_dir},
        'test_dir': {'condition': test_condition_dir, 'real': test_real_dir}
    }

def process_and_split_dataset(raw_condition_dir, raw_real_dir, processed_dir, train_size=800):
    """
    Comprehensive function to process and split the dataset
    """
    # First create a processor
    preprocessor = ImagePreprocessor(target_size=256)
    
    # Split the dataset
    split_info = split_dataset(raw_condition_dir, raw_real_dir, processed_dir, train_size)
    
    print("Dataset splitting and preprocessing complete!")
    return split_info

if __name__ == "__main__":
    # Relative paths
    raw_condition_dir = "./data/raw/condition_images"  # Condition images directory
    raw_real_dir = "./data/raw/real_images"  # Real images directory
    processed_dir = "./data/processed"  # Processed directory
    
    # Process and split the dataset
    split_info = process_and_split_dataset(raw_condition_dir, raw_real_dir, processed_dir, train_size=800)
    
    print(f"Training set size: {split_info['train_size']}")
    print(f"Test set size: {split_info['test_size']}")