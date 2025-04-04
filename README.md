# 深度学习图像生成与评估系统

## 项目简介
本项目是一个基于深度学习的图像生成与评估系统，使用UNet架构进行图像生成，并提供FCN和KL散度评估功能。主要采用基于GAN的方法，包含生成器(UNetGenerator)和判别器(Discriminator)模型。

## 主要功能
- 基于条件图像生成目标图像
- 使用FCN进行图像质量评估
- 计算KL散度进行图像相似度分析
- 自动生成评估报告和可视化结果
- 自动生成真实图片与生成图片的对比展示
- 支持模型训练和测试

## 环境要求
- Python 3.7+
- PyTorch 1.10.0+
- CUDA（推荐，支持CPU运行）
- PIL
- NumPy
- TensorBoard
- OpenCV (cv2)

## 项目结构
```
.
├── data/
│   ├── raw/                         # 原始数据
│   │   ├── condition_images/        # 原始条件图像
│   │   └── real_images/             # 原始真实图像
│   └── processed/
│       ├── train/                   # 训练数据
│       │   ├── condition_images/    # 训练用条件图像
│       │   └── real_images/         # 训练用真实图像
│       └── test/
│           ├── condition_images/    # 测试用条件图像
│           └── real_images/         # 测试用真实图像
├── checkpoints/                     # 模型检查点保存目录
├── output/
│   ├── evaluation_results/          # 评估结果输出目录
│   │   ├── tensorboard/            # TensorBoard日志目录
│   │   ├── comparison_images/      # 生成图片与真实图片的对比展示
│   └── generated_images/           # 生成的图像和训练日志
├── src/
│   ├── generator.py                 # UNet生成器模型
│   ├── discriminator.py             # 判别器模型
│   ├── evaluator.py                 # 图像评估器
│   ├── utils.py                     # 工具函数
│   ├── precon.py                    # 数据预处理
│   ├── config.py                    # 配置文件
│   ├── train.py                     # 训练脚本
│   └── test.py                      # 测试和评估脚本
```

## 使用说明

### 数据准备
```python
python src/precon.py
```
预处理脚本会：
1. 从原始数据目录读取图像
2. 将数据集分为训练集和测试集
3. 将处理后的数据存入processed目录

### 训练模型
```python
python src/train.py
```
训练脚本会：
1. 加载数据集
2. 初始化生成器和判别器
3. 开始GAN训练过程
4. 定期保存模型检查点
5. 在TensorBoard中记录训练指标

### 测试模型
```python
python src/test.py
```
测试脚本会：
1. 加载预训练的生成器模型
2. 对测试集图像进行处理
3. 生成评估结果
4. 创建生成图片与真实图片的对比展示
5. 在TensorBoard中记录评估指标

### 评估指标
- FCN IoU分数：评估生成图像的质量
- KL散度：衡量生成图像与真实图像的相似度

### 查看结果
1. 生成的图像将保存在 `output/evaluation_results` 目录
2. 对比图片可在 `output/evaluation_results/comparison_images` 目录查看
3. 使用TensorBoard查看详细评估指标：
```bash
tensorboard --logdir=output/evaluation_results/tensorboard
```

## 注意事项
- 确保测试前已准备好对应的条件图像和真实图像
- 图像文件支持格式：PNG、JPG、JPEG
- 默认图像处理尺寸为256x256
- 对比图片会自动将生成图片和真实图片水平拼接展示
- 本项目未设置随机种子，因此每次训练和测试的结果可能有所不同
- 如需提高实验的可重复性，建议在train.py中添加随机种子设置 