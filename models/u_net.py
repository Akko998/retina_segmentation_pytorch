import torch
import torch.nn.functional as F
from torch import nn


# from ..utils import initialize_weights


class _EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super(_EncoderBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, padding=1, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, padding=1, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ]
        if dropout:
            layers.append(nn.Dropout())
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, dropout=False):
        super(_DecoderBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, middle_channels, padding=1, kernel_size=3),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels, padding=1, kernel_size=3),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2),
        ]
        if dropout:
            layers.append(nn.Dropout())
        self.decode = nn.Sequential(*layers)

    def forward(self, x):
        return self.decode(x)


class UNet(nn.Module):
    def __init__(self, n_ch, num_classes):
        super(UNet, self).__init__()
        self.enc1 = _EncoderBlock(n_ch, 32, dropout=True)
        self.enc2 = _EncoderBlock(32, 64, dropout=True)
        self.center = _DecoderBlock(64, 128, 64, dropout=True)
        self.dec2 = _DecoderBlock(128, 64, 32, dropout=True)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, padding=1, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, padding=1, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1),
            nn.Sigmoid()
        )
        # initialize_weights(self)

    def forward(self, x):
        enc1 = self.enc1(x)  # (#patch, 32, 48, 48)
        enc2 = self.enc2(enc1)  # (#patch, 64, 24, 24)
        center = self.center(enc2)  # (#patch, 64, 24, 24)
        dec2 = self.dec2(torch.cat([center, F.interpolate(enc2, center.size()[2:], mode='bilinear', align_corners=False)], 1))  # (#patch, 32, 48, 48)
        dec1 = self.dec1(torch.cat([dec2, F.interpolate(enc1, dec2.size()[2:], mode='bilinear', align_corners=False)], 1))  # (#patch, 1, 48, 48)
        return dec1
