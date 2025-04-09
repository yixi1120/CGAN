import torch
import torch.nn as nn


class SimpleCNNEdge2Image(nn.Module):
    """
    Simple CNN model for generating real images from edge images
    Serves as a benchmark comparison model for cGAN methods
    """
    def __init__(self, in_channels=1, out_channels=3, ngf=64):
        """
        Parameters:
            in_channels: Input channels, 1 for grayscale edge images, 3 for RGB
            out_channels: Output channels, 3 for RGB images
            ngf: Base width of hidden layer channels
        """
        super(SimpleCNNEdge2Image, self).__init__()

        # Encoder part: 4 convolutional layers, each with stride=2, resolution halves each time
        self.encoder = nn.Sequential(
            # First layer: input -> ngf
            nn.Conv2d(in_channels, ngf, kernel_size=4, 
                      stride=2, padding=1),  # H/2
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # Second layer: ngf -> ngf*2
            nn.Conv2d(ngf, ngf*2, kernel_size=4, 
                      stride=2, padding=1),  # H/4
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),

            # Third layer: ngf*2 -> ngf*4
            nn.Conv2d(ngf*2, ngf*4, kernel_size=4, 
                      stride=2, padding=1),  # H/8
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),

            # Fourth layer: ngf*4 -> ngf*8
            nn.Conv2d(ngf*4, ngf*8, kernel_size=4, 
                      stride=2, padding=1),  # H/16
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),
        )

        # Decoder part: Corresponding transposed convolutions to upsample back
        self.decoder = nn.Sequential(
            # First layer: ngf*8 -> ngf*4
            nn.ConvTranspose2d(ngf*8, ngf*4, kernel_size=4, 
                               stride=2, padding=1),  # H/8
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),

            # Second layer: ngf*4 -> ngf*2
            nn.ConvTranspose2d(ngf*4, ngf*2, kernel_size=4, 
                               stride=2, padding=1),  # H/4
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),

            # Third layer: ngf*2 -> ngf
            nn.ConvTranspose2d(ngf*2, ngf, kernel_size=4, 
                               stride=2, padding=1),  # H/2
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # Fourth layer: ngf -> out_channels
            nn.ConvTranspose2d(ngf, out_channels, kernel_size=4, 
                               stride=2, padding=1),  # H
            nn.Tanh()  # Output range [-1,1], can be changed to Sigmoid based on data normalization
        )

    def forward(self, x):
        """
        Forward pass
        Parameters:
            x: Input edge image, shape [B, in_channels, H, W]
        Returns:
            Generated real image, shape [B, out_channels, H, W]
        """
        # Encoding
        features = self.encoder(x)
        # Decoding
        output = self.decoder(features)
        return output


if __name__ == "__main__":
    # Simple model test
    model = SimpleCNNEdge2Image(in_channels=1, out_channels=3, ngf=64)
    test_input = torch.randn(4, 1, 256, 256)  # batch_size=2, 1 channel, 256x256
    test_output = model(test_input)
    print("Total model parameters:", sum(p.numel() for p in model.parameters()))
    print("Input shape:", test_input.shape)
    print("Output shape:", test_output.shape) 