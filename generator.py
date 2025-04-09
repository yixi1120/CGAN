import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: Try more convolution kernel size options to see the effect
# Define convolution-deconvolution modules first
class CD(nn.Module):
    def __init__(self, in_channels, out_channels, use_dropout=False):
        super().__init__()
        # Padding=1 seems most suitable for this task
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        if use_dropout:
            layers.append(nn.Dropout(0.5))  # Tests show that 0.5 dropout works well
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class DC(nn.Module):
    def __init__(self, in_channels, out_channels, use_dropout=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)  # Using ReLU here instead of LeakyReLU
        ]
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetGenerator(nn.Module):
    def __init__(self, input_channels=3, output_channels=3):
        super().__init__()
        
        # Reduce encoder layers to ensure feature maps don't get too small
        self.enc1 = CD(input_channels, 64)    # 256 -> 128
        self.enc2 = CD(64, 128)               # 128 -> 64
        self.enc3 = CD(128, 256)              # 64 -> 32
        self.enc4 = CD(256, 512)              # 32 -> 16
        self.enc5 = CD(512, 512)              # 16 -> 8
        
        # Bottleneck layer - maybe one layer is enough
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Reduce decoder layers to match with encoder
        # Using dropout in the first two layers for now
        use_dropout = True
        self.dec5 = DC(512, 512, use_dropout)    # 8 -> 16
        self.dec4 = DC(1024, 256, use_dropout)   # 16 -> 32
        self.dec3 = DC(512, 128)                 # 32 -> 64
        self.dec2 = DC(256, 64)                  # 64 -> 128
        
        # Final output layer, using tanh to ensure output is in [-1,1] range
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)      # 256 -> 128
        e2 = self.enc2(e1)     # 128 -> 64
        e3 = self.enc3(e2)     # 64 -> 32
        e4 = self.enc4(e3)     # 32 -> 16
        e5 = self.enc5(e4)     # 16 -> 8
        
        # Bottleneck layer
        bottleneck = self.bottleneck(e5)
        
        # Decoder (with skip connections)
        d5 = self.dec5(bottleneck)
        d4 = self.dec4(torch.cat([d5, e4], 1))  # Concatenate feature maps
        d3 = self.dec3(torch.cat([d4, e3], 1))
        d2 = self.dec2(torch.cat([d3, e2], 1))
        
        # Final output
        return self.final(torch.cat([d2, e1], 1))

# Weight initialization function - not sure how much this helps, but adding it
def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0)

# Initialize generator (input 3 channel RGB, output 3 channel RGB)
generator = UNetGenerator(input_channels=3, output_channels=3)

# Weight initialization (as described in the paper)
generator.apply(weights_init)

# Forward pass example
input_tensor = torch.randn(1, 3, 256, 256)  # Input size 256x256
output = generator(input_tensor)  # Output size 256x256





