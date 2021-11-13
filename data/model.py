import torch
from torch import nn
import torch.nn.functional as F

class ResDenseBlock(nn.Module):
    """
    Achieves densely connected convolutional layers.
    """

    def __init__(self, channels: int, growths: int) -> None:
        super(ResDenseBlock, self).__init__()
        self.channels = channels
        self.growths = growths

        self.conv_list = [self.make_conv(i) for i in range(5)]
        self.leaky_relu = nn.LeakyReLU(0.2, True)
        self.identity = nn.Identity()

    def make_conv(self, i):
        conv = nn.Conv2d(self.channels + self.growths * i, self.growths, (3, 3), (1, 1), (1, 1))
        return conv

    def forward(self, x):
        identity = x

        out1 = self.leaky_relu(self.make_conv[0](x))
        out2 = self.leaky_relu(self.make_conv[1](torch.cat([x, out1], 1)))
        out3 = self.leaky_relu(self.make_conv[2](torch.cat([x, out1, out2], 1)))
        out4 = self.leaky_relu(self.make_conv[3](torch.cat([x, out1, out2, out3], 1)))
        out5 = self.identity(self.make_conv[4](torch.cat([x, out1, out2, out3, out4], 1)))
        out = out5 * 0.2 + identity

        return out

class ResResDenseBlock(nn.Module):
    """
    Multi-layer residual dense convolution block.
    """

    def __init__(self, channels: int, growths: int):
        super(ResResDenseBlock, self).__init__()
        self.rdb = [ResDenseBlock(channels, growths) for i in range(3)]

    def forward(self, x: torch.Tensor):
        identity = x
        out = x

        for i in range(3):
            out = self.rdb[i](out)

        out = out * 0.2 + identity

        return out

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        channels = [64, 64, 128, 128, 256, 256, 512, 512, 512, 512]
        layers = [nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1), bias=True),
                  nn.LeakyReLU(0.2, True)]
        
        for i in range(1,len(channels)):
            layers = layers + self.make_layer(channels[i-1], channels[i], i)

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Linear(512*4*4, 100),
            nn.LeakyReLU(0.2, True),
            nn.Liner(100,1)
        )

    def make_layer(self, in_channel, out_channel, idx):
        kernel = [(4,4),(3,3)]
        pads = [(2,2),(1,1)]
        layer = [nn.Conv2d(in_channel, out_channel, kernel[idx%2], pads[idx%2], (1,1), bias=False),
                 nn.BatchNorm2d(out_channel),
                 nn.LeakyReLU(0.2, True)]
        return layer

    def forward(self, x):
        out = self.features(x)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        fext = []
        for _ in range(16):
            fext += [ResResDenseBlock(64, 32)]
        self.fext = nn.Sequential(*fext)

        self.conv1 = nn.conv2d(3, 64, (3,3), (1,1), (1,1))
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, (3,3), (1,1), (1,1)),
            nn.LeakyReLU(0.2, True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, (3,3), (1,1), (1,1)),
            nn.LeakyReLU(0.2, True)
        )
        self.conv4 = nn.Conv2d(64, 3, (3,3), (1,1), (1,1))

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                m.weight.data *= 0.1
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                m.weight.data *= 0.1

    def forward(self, x):
        
        out1 = self.conv1(x)
        out = self.trunk(out1)
        out2 = self.conv2(out)
        out = out1 + out2
        out = self.upsampling(F.interpolate(out, scale_factor=2, mode="nearest"))
        out = self.upsampling(F.interpolate(out, scale_factor=2, mode="nearest"))
        out = self.conv3(out)
        out = self.conv4(out)

        return out
