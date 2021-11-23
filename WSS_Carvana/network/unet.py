import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):   # defines the computation performed by every call.
        return self.conv(x) # this is object, but callable


class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, middle_channels=[64, 128, 256, 512]):
        super().__init__()
        self.ups = nn.ModuleList() # this is a class. Holds submodules in a list
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for channel in middle_channels:
            self.downs.append(DoubleConv(in_channels, channel))
            in_channels = channel

        # Up part of UNET
        for channel in reversed(middle_channels):
            self.ups.append(nn.ConvTranspose2d(channel*2, channel, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(channel*2, channel))

        self.bottleneck = DoubleConv(middle_channels[-1], middle_channels[-1] * 2)
        self.final_conv = nn.Conv2d(middle_channels[0], out_channels, kernel_size=1)

    def forward(self, x):   # the computation performed by every call. "x" seems the image matrix
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)
