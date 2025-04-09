# config.py
import os

# Data path configuration
RAW_DATA_DIR = './data/raw'  # Original data folder path
PROCESSED_DATA_DIR = './data/processed'  # Processed data folder path
CHECKPOINT_DIR = './checkpoints'  # Model checkpoint save path
OUTPUT_DIR = './output'  # Generated images and results output path

# Training dataset paths
TRAIN_CONDITION_IMAGES_DIR = os.path.join(PROCESSED_DATA_DIR, 'train/condition_images')
TRAIN_REAL_IMAGES_DIR = os.path.join(PROCESSED_DATA_DIR, 'train/real_images')

# Testing dataset paths
TEST_CONDITION_IMAGES_DIR = os.path.join(PROCESSED_DATA_DIR, 'test/condition_images')
TEST_REAL_IMAGES_DIR = os.path.join(PROCESSED_DATA_DIR, 'test/real_images')

# Training configuration
BATCH_SIZE = 4  # Batch size
LEARNING_RATE = 0.0001  # Learning rate
NUM_EPOCHS = 5000  # Number of training epochs

# Whether to enable mixed precision training
USE_MIXED_PRECISION = True

# Random seed setting
RANDOM_SEED = None
