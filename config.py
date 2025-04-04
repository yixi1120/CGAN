# config.py
import os

# 数据路径配置
RAW_DATA_DIR = './data/raw'  # 原始数据文件夹路径
PROCESSED_DATA_DIR = './data/processed'  # 处理后数据的文件夹路径
CHECKPOINT_DIR = './checkpoints'  # 模型检查点保存的路径
OUTPUT_DIR = './output'  # 生成图像和结果输出的路径

# 训练数据集路径
TRAIN_CONDITION_IMAGES_DIR = os.path.join(PROCESSED_DATA_DIR, 'train/condition_images')
TRAIN_REAL_IMAGES_DIR = os.path.join(PROCESSED_DATA_DIR, 'train/real_images')

# 测试数据集路径
TEST_CONDITION_IMAGES_DIR = os.path.join(PROCESSED_DATA_DIR, 'test/condition_images')
TEST_REAL_IMAGES_DIR = os.path.join(PROCESSED_DATA_DIR, 'test/real_images')

# 训练配置
BATCH_SIZE = 4  # 批量大小
LEARNING_RATE = 0.0001  # 学习率
NUM_EPOCHS = 5000  # 训练的 epoch 数量

# 是否启用混合精度训练
USE_MIXED_PRECISION = True

# 随机种子设置（默认为None，表示不使用固定种子）
# 如果需要可重复的实验结果，请设置一个固定值，如42
RANDOM_SEED = None
