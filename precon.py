import os
import sys
# 将项目根目录添加到Python路径
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
        print(f"文件 {image_path} 不存在！")
        return None

    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像文件：{image_path}")
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
    print(f"图像已保存到：{output_path}")

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
                    print(f"成功处理图像：{file_name}")
                except Exception as e:
                    print(f"处理图像 {file_name} 时出错: {str(e)}")
                    
        return processed_files

def split_dataset(raw_condition_dir, raw_real_dir, processed_dir, train_size=800, random_split=False):
    """
    将原始数据集拆分为训练集和测试集
    
    参数:
        raw_condition_dir: 原始条件图像目录
        raw_real_dir: 原始真实图像目录
        processed_dir: 处理后的数据存放目录
        train_size: 训练集大小，默认800
        random_split: 是否随机划分，默认False（按顺序划分）
    """
    # 创建目录结构
    train_condition_dir = os.path.join(processed_dir, 'train/condition_images')
    train_real_dir = os.path.join(processed_dir, 'train/real_images')
    test_condition_dir = os.path.join(processed_dir, 'test/condition_images')
    test_real_dir = os.path.join(processed_dir, 'test/real_images')
    
    for directory in [train_condition_dir, train_real_dir, test_condition_dir, test_real_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # 获取原始图像文件列表
    condition_files = [f for f in os.listdir(raw_condition_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    real_files = [f for f in os.listdir(raw_real_dir) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    # 找出两个目录中共同的文件名
    common_files = list(set(condition_files).intersection(set(real_files)))
    common_files.sort()  # 排序以确保一致性
    
    print(f"找到{len(common_files)}对图像文件")
    
    if random_split:
        # 随机打乱文件顺序
        import random
        random.shuffle(common_files)
    
    # 分割为训练集和测试集
    train_files = common_files[:train_size]
    test_files = common_files[train_size:]
    
    print(f"训练集: {len(train_files)}对图像")
    print(f"测试集: {len(test_files)}对图像")
    
    # 复制训练集文件
    for filename in train_files:
        # 复制条件图像
        src_condition = os.path.join(raw_condition_dir, filename)
        dst_condition = os.path.join(train_condition_dir, filename)
        shutil.copy2(src_condition, dst_condition)
        
        # 复制真实图像
        src_real = os.path.join(raw_real_dir, filename)
        dst_real = os.path.join(train_real_dir, filename)
        shutil.copy2(src_real, dst_real)
    
    # 复制测试集文件
    for filename in test_files:
        # 复制条件图像
        src_condition = os.path.join(raw_condition_dir, filename)
        dst_condition = os.path.join(test_condition_dir, filename)
        shutil.copy2(src_condition, dst_condition)
        
        # 复制真实图像
        src_real = os.path.join(raw_real_dir, filename)
        dst_real = os.path.join(test_real_dir, filename)
        shutil.copy2(src_real, dst_real)
    
    print("数据集划分完成！")
    return {
        'train_size': len(train_files),
        'test_size': len(test_files),
        'train_dir': {'condition': train_condition_dir, 'real': train_real_dir},
        'test_dir': {'condition': test_condition_dir, 'real': test_real_dir}
    }

def process_and_split_dataset(raw_condition_dir, raw_real_dir, processed_dir, train_size=800):
    """
    处理并划分数据集的综合函数
    """
    # 首先创建处理器
    preprocessor = ImagePreprocessor(target_size=256)
    
    # 划分数据集
    split_info = split_dataset(raw_condition_dir, raw_real_dir, processed_dir, train_size)
    
    print("数据集划分和预处理完成！")
    return split_info

if __name__ == "__main__":
    # 相对路径修改
    raw_condition_dir = "./data/raw/condition_images"  # 条件图像目录
    raw_real_dir = "./data/raw/real_images"  # 真实图像目录
    processed_dir = "./data/processed"  # 处理后的目录
    
    # 处理并划分数据集
    split_info = process_and_split_dataset(raw_condition_dir, raw_real_dir, processed_dir, train_size=800)
    
    print(f"训练集大小: {split_info['train_size']}")
    print(f"测试集大小: {split_info['test_size']}")