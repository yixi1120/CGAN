import torch
import torch.nn as nn


class SimpleCNNEdge2Image(nn.Module):
    """
    简单的CNN模型，用于从边缘图生成真实图像
    作为cGAN方法的基准对比模型
    """
    def __init__(self, in_channels=1, out_channels=3, ngf=64):
        """
        参数:
            in_channels: 输入通道数，边缘图为灰度图时为1，三通道时为3
            out_channels: 输出通道数，RGB图像为3
            ngf: 隐藏层通道数的基础宽度
        """
        super(SimpleCNNEdge2Image, self).__init__()

        # 编码部分: 4层卷积，每层stride=2，分辨率依次减半
        self.encoder = nn.Sequential(
            # 第一层: 输入 -> ngf
            nn.Conv2d(in_channels, ngf, kernel_size=4, 
                      stride=2, padding=1),  # H/2
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # 第二层: ngf -> ngf*2
            nn.Conv2d(ngf, ngf*2, kernel_size=4, 
                      stride=2, padding=1),  # H/4
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),

            # 第三层: ngf*2 -> ngf*4
            nn.Conv2d(ngf*2, ngf*4, kernel_size=4, 
                      stride=2, padding=1),  # H/8
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),

            # 第四层: ngf*4 -> ngf*8
            nn.Conv2d(ngf*4, ngf*8, kernel_size=4, 
                      stride=2, padding=1),  # H/16
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),
        )

        # 解码部分: 对应地用转置卷积上采样回去
        self.decoder = nn.Sequential(
            # 第一层: ngf*8 -> ngf*4
            nn.ConvTranspose2d(ngf*8, ngf*4, kernel_size=4, 
                               stride=2, padding=1),  # H/8
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),

            # 第二层: ngf*4 -> ngf*2
            nn.ConvTranspose2d(ngf*4, ngf*2, kernel_size=4, 
                               stride=2, padding=1),  # H/4
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),

            # 第三层: ngf*2 -> ngf
            nn.ConvTranspose2d(ngf*2, ngf, kernel_size=4, 
                               stride=2, padding=1),  # H/2
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # 第四层: ngf -> out_channels
            nn.ConvTranspose2d(ngf, out_channels, kernel_size=4, 
                               stride=2, padding=1),  # H
            nn.Tanh()  # 输出范围[-1,1]，根据数据归一化方式可改为Sigmoid
        )

    def forward(self, x):
        """
        前向传播
        参数:
            x: 输入边缘图，形状为[B, in_channels, H, W]
        返回:
            生成的真实图像，形状为[B, out_channels, H, W]
        """
        # 编码
        features = self.encoder(x)
        # 解码
        output = self.decoder(features)
        return output


if __name__ == "__main__":
    # 简单测试模型
    model = SimpleCNNEdge2Image(in_channels=1, out_channels=3, ngf=64)
    test_input = torch.randn(4, 1, 256, 256)  # batch_size=2, 1通道, 256x256
    test_output = model(test_input)
    print("模型总参数量:", sum(p.numel() for p in model.parameters()))
    print("输入尺寸:", test_input.shape)
    print("输出尺寸:", test_output.shape) 