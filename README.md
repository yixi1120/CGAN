# Pure CNN Edge-to-Image Mapping Model

This project implements a simple CNN model for converting edge images to real images. The model serves as a benchmark comparison model for conditional Generative Adversarial Network (cGAN) methods.

## Project Overview

To ensure that the pure CNN approach is completely different from cGAN, this project provides a very simplified training scheme that forms a clear contrast with cGAN in terms of training methods. We use a simple CNN structure without an adversarial network, focusing only on pixel-level reconstruction based on input edge images and corresponding real images. This approach will focus on the simplest training process.

## Model Architecture

The model adopts a simple Encoder-Decoder structure:

1. **Encoder**: Contains 4 convolutional layers, each with stride=2, gradually reducing feature map size and extracting features.
2. **Decoder**: Contains 4 transposed convolutional layers, gradually restoring image resolution, finally outputting an RGB image of the same size as the input.

This is a very simple convolutional network that does not use the generative adversarial network framework (i.e., no discriminator). It relies solely on the reconstruction loss (e.g., L1 or L2) between the input edge image and the real image.

## Training Method

1. **Loss Function**:
   - Uses **L1 Loss** (absolute error) or **MSE Loss** (mean squared error) to measure the difference between generated images and real images.

2. **Optimizer**:
   - Uses **Adam optimizer** to update model parameters.

3. **Simplified Training Loop**:
   - Unlike cGAN, this model has only one generator network, with no discriminator.
   - The model training only focuses on how to generate realistic images by minimizing reconstruction loss.

4. **Differences Between Training and Evaluation**:
   - In this method, we have no discriminator and rely entirely on pixel reconstruction loss to optimize the model, which is completely different from the adversarial training in cGAN.
   - Evaluation: The model's performance is evaluated through two main metrics:
     1. FCN score: Evaluates the semantic quality of generated images (range 0-1, closer to 1 is better)
     2. KL divergence: Measures the distribution difference between generated images and real images (smaller is better)

## Project Structure

```
cnn_edge2image/
│
├── train.py                  # Training script, responsible for model training process
├── test.py                   # Testing script, used to evaluate model performance (FCN score and KL divergence)
├── simple_cnn_model.py       # Model definition file, contains CNN network structure
├── utils.py                  # Utility functions, including dataset, saving checkpoints, etc.
├── requirements.txt          # Project dependency library list
├── README.md                 # Project documentation
│
├── checkpoints/              # Model checkpoint save directory
│   ├── final_model.pth       # Final model after training completion
│   ├── best_model.pth        # Model with lowest validation loss
│   └── checkpoint_xx.pth     # Intermediate models during training
│
├── samples/                  # Samples generated during training
│   └── sample_xxxx.png       # Sample images at different training steps
│
├── test_results/             # Test results save directory
│   ├── test_sample_xx.png    # Test sample images
│   └── evaluation_metrics.png # Evaluation metrics distribution chart
│
├── runs/                     # TensorBoard log directory
│   ├── train_xxxxxxxx-xxxxxx/ # Training logs
│   └── test_xxxxxxxx-xxxxxx/  # Testing logs
│
└── processed/                # Processed dataset
    ├── train/                # Training data
    │   ├── edges/            # Edge images
    │   └── real_images/      # Corresponding real images
    └── test/                 # Test data
        ├── edges/            # Edge images
        └── real_images/      # Corresponding real images
```

## Dataset Format

The dataset should be organized in the following format:

```
data_dir/
  ├── edges/        # Edge images folder
  └── real_images/  # Real images folder
```

Edge images and real images should have the same filename, for example:
- `edges/image_001.png` corresponds to `real_images/image_001.png`

## Installation

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install torch torchvision numpy matplotlib pillow opencv-python tqdm scipy tensorboard
```

## Training the Model

Use the following command to start training:

```bash
python train.py --data_dir path/to/dataset --batch_size 4 --epochs 100
```

Main parameters:

- `--data_dir`: Training data directory (required)
- `--val_dir`: Validation data directory (optional)
- `--image_size`: Image size, default 256
- `--in_channels`: Input channels, default 1 (grayscale edge image)
- `--out_channels`: Output channels, default 3 (RGB image)
- `--ngf`: Base width of model feature maps, default 64
- `--batch_size`: Batch size, default 4
- `--epochs`: Number of training epochs, default 100
- `--lr`: Learning rate, default 0.0001
- `--loss`: Loss function type, options "l1" (absolute error) or "mse" (mean squared error), default "l1"
- `--log_dir`: TensorBoard log save directory, default "runs"

For more parameters, please check the `train.py` file.

## Testing the Model

After training is complete, use the following command to test model performance:

```bash
python test.py --test_dir path/to/test/data --model_path checkpoints/final_model.pth
```

Main parameters:

- `--test_dir`: Test data directory (required)
- `--model_path`: Pretrained model path (required)
- `--output_dir`: Test results save directory, default is "test_results"
- `--batch_size`: Batch size, default 4
- `--sample_freq`: Sample saving frequency, default 10
- `--log_dir`: TensorBoard log save directory, default "runs"

The test script will:
1. Load the pretrained model
2. Generate images on the test set
3. Calculate FCN score (semantic quality metric) and KL divergence (distribution difference metric)
4. Save test samples and evaluation metric distribution charts
5. Record results to TensorBoard

Test results will be saved in the specified output directory, including:
- Generated sample image comparisons
- Evaluation metrics distribution charts
- Average FCN score and KL divergence

## Visualizing Training/Testing Process

This project integrates TensorBoard support for real-time monitoring of training and testing:

```bash
tensorboard --logdir=runs
```

TensorBoard visualization includes:
- Training loss curves
- Validation metrics (KL divergence and FCN score)
- Learning rate changes
- Generated sample images
- Evaluation metrics distribution
- Model structure diagram

## Generating Test Dataset

If you need to generate a simple edge image-real image pair dataset for testing, you can run:

```bash
python utils.py
```

This will create a simple test dataset with 10 image pairs.

## Comparison with cGAN

As a benchmark model, this pure CNN approach differs from the cGAN approach in several key ways:

1. **Model Structure**: Only a generator network, no discriminator
2. **Training Process**: No adversarial training, only focusing on reconstruction loss
3. **Loss Function**: Uses only pixel-level L1 or MSE loss for training
4. **Evaluation Method**: Uses FCN score to evaluate semantic quality, KL divergence to evaluate distribution difference

This simple CNN model can serve as a "control group" for evaluating the advantages of complex cGAN models relative to pure supervised learning methods. 