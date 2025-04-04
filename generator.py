import torch
import torch.nn as nn
import torch.nn.functional as F


# 先定义卷积-解卷积模块
class CD(nn.Module):
    def __init__(self, in_channels, out_channels, use_dropout=False):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class DC(nn.Module):
    def __init__(self, in_channels, out_channels, use_dropout=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        ]
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetGenerator(nn.Module):
    def __init__(self, input_channels=3, output_channels=3):
        super().__init__()
        
        # 减少编码器层数，确保特征图不会太小
        self.enc1 = CD(input_channels, 64)    # 256 -> 128
        self.enc2 = CD(64, 128)               # 128 -> 64
        self.enc3 = CD(128, 256)              # 64 -> 32
        self.enc4 = CD(256, 512)              # 32 -> 16
        self.enc5 = CD(512, 512)              # 16 -> 8
        
        # 瓶颈层
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 减少解码器层数，与编码器对应
        use_dropout = True
        self.dec5 = DC(512, 512, use_dropout)    # 8 -> 16
        self.dec4 = DC(1024, 256, use_dropout)   # 16 -> 32
        self.dec3 = DC(512, 128)                 # 32 -> 64
        self.dec2 = DC(256, 64)                  # 64 -> 128
        
        # 最终输出层
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # 编码器
        e1 = self.enc1(x)      # 256 -> 128
        e2 = self.enc2(e1)     # 128 -> 64
        e3 = self.enc3(e2)     # 64 -> 32
        e4 = self.enc4(e3)     # 32 -> 16
        e5 = self.enc5(e4)     # 16 -> 8
        
        # 瓶颈层
        bottleneck = self.bottleneck(e5)
        
        # 解码器（带跳跃连接）
        d5 = self.dec5(bottleneck)
        d4 = self.dec4(torch.cat([d5, e4], 1))
        d3 = self.dec3(torch.cat([d4, e3], 1))
        d2 = self.dec2(torch.cat([d3, e2], 1))
        
        # 最终输出
        return self.final(torch.cat([d2, e1], 1))

# 权重初始化函数
def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0)

# 初始化生成器（输入3通道RGB，输出3通道RGB）
generator = UNetGenerator(input_channels=3, output_channels=3)

# 权重初始化（如论文所述）
generator.apply(weights_init)

# 前向传播示例
input_tensor = torch.randn(1, 3, 256, 256)  # 输入尺寸256x256
output = generator(input_tensor)  # 输出尺寸256x256





