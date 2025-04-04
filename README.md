# 纯CNN边缘图到真实图像的映射模型

这个项目实现了一个简单的CNN模型，用于将边缘图转换为真实图像。该模型作为条件生成对抗网络(cGAN)方法的基准对比模型。

## 项目概述

为了确保纯CNN方法与cGAN完全不同，本项目给出一种非常简化的训练方案，与cGAN在训练方法上形成鲜明对比。我们使用一个简单的CNN结构，它没有对抗网络，只通过输入边缘图像和对应的真实图像进行像素级重建。此方法将专注于最简单的训练流程。

## 模型结构

该模型采用了简单的Encoder-Decoder结构：

1. **编码器(Encoder)**: 包含4层卷积层，每层stride=2，逐步减小特征图尺寸并提取特征。
2. **解码器(Decoder)**: 包含4层转置卷积层，逐步恢复图像分辨率，最终输出与输入相同尺寸的RGB图像。

这是一个非常简单的卷积网络，它没有使用生成对抗网络的框架（即没有判别器）。只依赖输入边缘图像和真实图像之间的重建损失（例如L1或L2）。

## 训练方法

1. **损失函数**：
   - 使用**L1损失**（绝对误差）或**MSE损失**（均方误差）来衡量生成图像和真实图像之间的差异。

2. **优化器**：
   - 使用**Adam优化器**来更新模型参数。

3. **简化训练循环**：
   - 与cGAN不同，这个模型只有一个生成网络，不需要判别器。
   - 模型的训练只关注如何通过最小化重建损失来生成逼真的图像。

4. **训练与评估的区别**：
   - 在这个方法中，我们没有判别器，完全依赖于像素重建损失来优化模型，这与cGAN中的对抗训练完全不同。
   - 评估：通过两个主要指标评估模型性能：
     1. FCN分数：评估生成图像的语义质量（0-1范围，越接近1越好）
     2. KL散度：衡量生成图像与真实图像的分布差异（越小越好）

## 项目结构

```
cnn_edge2image/
│
├── train.py                  # 训练脚本，负责模型训练过程
├── test.py                   # 测试脚本，用于评估模型性能（FCN分数和KL散度）
├── simple_cnn_model.py       # 模型定义文件，包含CNN网络结构
├── utils.py                  # 工具函数，包含数据集、保存检查点等功能
├── requirements.txt          # 项目依赖库列表
├── README.md                 # 项目说明文档
│
├── checkpoints/              # 模型检查点保存目录
│   ├── final_model.pth       # 训练完成的最终模型
│   ├── best_model.pth        # 验证损失最低的模型
│   └── checkpoint_xx.pth     # 训练过程中的阶段性模型
│
├── samples/                  # 训练过程中生成的样本图像
│   └── sample_xxxx.png       # 各训练步骤的样本图像
│
├── test_results/             # 测试结果保存目录
│   ├── test_sample_xx.png    # 测试样本图像
│   └── evaluation_metrics.png # 评估指标分布图
│
├── runs/                     # TensorBoard日志目录
│   ├── train_xxxxxxxx-xxxxxx/ # 训练日志
│   └── test_xxxxxxxx-xxxxxx/  # 测试日志
│
└── processed/                # 处理好的数据集
    ├── train/                # 训练数据
    │   ├── edges/            # 边缘图像
    │   └── real_images/      # 对应的真实图像
    └── test/                 # 测试数据
        ├── edges/            # 边缘图像
        └── real_images/      # 对应的真实图像
```

## 数据集格式

数据集应按以下格式组织：

```
data_dir/
  ├── edges/        # 边缘图文件夹
  └── real_images/  # 真实图像文件夹
```

边缘图和真实图像应具有相同的文件名，例如：
- `edges/image_001.png` 对应 `real_images/image_001.png`

## 安装依赖

```bash
pip install -r requirements.txt
```

或手动安装：

```bash
pip install torch torchvision numpy matplotlib pillow opencv-python tqdm scipy tensorboard
```

## 训练模型

使用以下命令启动训练：

```bash
python train.py --data_dir path/to/dataset --batch_size 4 --epochs 100
```

主要参数说明：

- `--data_dir`: 训练数据目录（必需）
- `--val_dir`: 验证数据目录（可选）
- `--image_size`: 图像尺寸，默认256
- `--in_channels`: 输入通道数，默认1（灰度边缘图）
- `--out_channels`: 输出通道数，默认3（RGB图像）
- `--ngf`: 模型特征图基础宽度，默认64
- `--batch_size`: 批次大小，默认4
- `--epochs`: 训练轮数，默认100
- `--lr`: 学习率，默认0.0001
- `--loss`: 损失函数类型，可选"l1"（绝对误差）或"mse"（均方误差），默认"l1"
- `--log_dir`: TensorBoard日志保存目录，默认"runs"

更多参数请查看`train.py`文件。

## 测试模型

训练完成后，使用以下命令测试模型性能：

```bash
python test.py --test_dir path/to/test/data --model_path checkpoints/final_model.pth
```

主要参数说明：

- `--test_dir`: 测试数据目录（必需）
- `--model_path`: 预训练模型路径（必需）
- `--output_dir`: 测试结果保存目录，默认为"test_results"
- `--batch_size`: 批次大小，默认4
- `--sample_freq`: 保存样本的频率，默认10
- `--log_dir`: TensorBoard日志保存目录，默认"runs"

测试脚本将：
1. 加载预训练模型
2. 在测试集上生成图像
3. 计算FCN分数（语义质量指标）和KL散度（分布差异指标）
4. 保存测试样本和评估指标分布图
5. 记录结果到TensorBoard

测试结果将保存在指定的输出目录中，包括：
- 生成的样本图像对比
- 评估指标的统计分布图
- 平均FCN分数和KL散度

## 可视化训练/测试过程

本项目集成了TensorBoard支持，可以实时查看训练和测试过程：

```bash
tensorboard --logdir=runs
```

TensorBoard可视化内容包括：
- 训练损失曲线
- 验证指标（KL散度和FCN分数）
- 学习率变化
- 生成的样本图像
- 评估指标分布
- 模型结构图

## 生成测试数据集

如果需要为测试生成一个简单的边缘图-真实图像对数据集，可以运行：

```bash
python utils.py
```

这将创建一个包含10对图像的简单测试数据集。

## 与cGAN的比较

作为一个基准模型，这个纯CNN方法与cGAN方法的主要区别在于：

1. **模型结构**：只有一个生成网络，没有判别器
2. **训练过程**：没有对抗性训练，只关注重建损失
3. **损失函数**：仅使用像素级L1或MSE损失进行训练
4. **评估方式**：使用FCN分数评估语义质量，KL散度评估分布差异

这个简单的CNN模型可以作为一个"对照组"，用于评估复杂的cGAN模型相对于纯监督学习方法的优势。 