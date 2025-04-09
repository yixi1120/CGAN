# Deep Learning Image Generation and Evaluation System

## Project Introduction
This project is a deep learning-based image generation and evaluation system using UNet architecture for image generation, and providing FCN and KL divergence evaluation functions. It mainly adopts GAN-based methods, including generator (UNetGenerator) and discriminator (Discriminator) models.

## Main Features
- Generate target images based on conditional images
- Use FCN for image quality evaluation
- Calculate KL divergence for image similarity analysis
- Automatically generate evaluation reports and visualization results
- Automatically generate comparison displays of real images and generated images
- Support model training and testing

## Environment Requirements
- Python 3.7+
- PyTorch 1.10.0+
- CUDA (recommended, CPU operation supported)
- PIL
- NumPy
- TensorBoard
- OpenCV (cv2)

## Project Structure
```
.
├── data/
│   ├── raw/                         # Raw data
│   │   ├── condition_images/        # Original condition images
│   │   └── real_images/             # Original real images
│   └── processed/
│       ├── train/                   # Training data
│       │   ├── condition_images/    # Condition images for training
│       │   └── real_images/         # Real images for training
│       └── test/
│           ├── condition_images/    # Condition images for testing
│           └── real_images/         # Real images for testing
├── checkpoints/                     # Model checkpoint save directory
├── output/
│   ├── evaluation_results/          # Evaluation results output directory
│   │   ├── tensorboard/            # TensorBoard log directory
│   │   ├── comparison_images/      # Comparison display of generated and real images
│   └── generated_images/           # Generated images and training logs
├── src/
│   ├── generator.py                 # UNet generator model
│   ├── discriminator.py             # Discriminator model
│   ├── evaluator.py                 # Image evaluator
│   ├── utils.py                     # Utility functions
│   ├── precon.py                    # Data preprocessing
│   ├── config.py                    # Configuration file
│   ├── train.py                     # Training script
│   └── test.py                      # Testing and evaluation script
```

## Usage Instructions

### Data Preparation
```python
python src/precon.py
```
The preprocessing script will:
1. Read images from the raw data directory
2. Split the dataset into training and test sets
3. Store the processed data in the processed directory

### Training Model
```python
python src/train.py
```
The training script will:
1. Load the dataset
2. Initialize the generator and discriminator
3. Begin the GAN training process
4. Periodically save model checkpoints
5. Record training metrics in TensorBoard

### Testing Model
```python
python src/test.py
```
The testing script will:
1. Load the pre-trained generator model
2. Process test set images
3. Generate evaluation results
4. Create comparison displays of generated images and real images
5. Record evaluation metrics in TensorBoard

### Evaluation Metrics
- FCN IoU score: Evaluates the quality of generated images
- KL divergence: Measures the similarity between generated images and real images

### Viewing Results
1. Generated images will be saved in the `output/evaluation_results` directory
2. Comparison images can be viewed in the `output/evaluation_results/comparison_images` directory
3. Use TensorBoard to view detailed evaluation metrics:
```bash
tensorboard --logdir=output/evaluation_results/tensorboard
```

## Notes
- Ensure corresponding condition images and real images are prepared before testing
- Supported image file formats: PNG, JPG, JPEG
- Default image processing size is 256x256
- Comparison images will automatically display generated images and real images horizontally concatenated
- This project does not set random seeds, so results may vary between training and testing sessions
- To improve experiment reproducibility, consider adding random seed settings in train.py 