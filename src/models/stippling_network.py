"""
Stippling Neural Network
A U-Net style architecture for generating stippled images from input photographs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Convolutional block with batch normalization and ReLU activation"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class EncoderBlock(nn.Module):
    """Encoder block with two convolutions and max pooling"""
    
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        skip = x
        x = self.pool(x)
        return x, skip


class DecoderBlock(nn.Module):
    """Decoder block with transposed convolution and skip connections"""
    
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
    
    def forward(self, x, skip):
        x = self.upconv(x)
        # Handle size mismatch
        if x.size() != skip.size():
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class AttentionBlock(nn.Module):
    """Attention mechanism for focusing on important features"""
    
    def __init__(self, channels):
        super(AttentionBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels * 2, channels // 8),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 8, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        avg_feat = self.avg_pool(x).view(b, c)
        max_feat = self.max_pool(x).view(b, c)
        combined = torch.cat([avg_feat, max_feat], dim=1)
        attention = self.fc(combined).view(b, c, 1, 1)
        return x * attention.expand_as(x)


class StipplingNetwork(nn.Module):
    """
    Deep Learning Network for 2D Image Stippling
    
    Architecture: U-Net style encoder-decoder with attention mechanisms
    Input: RGB image (3 channels)
    Output: Dot density map (1 channel) with values in [0, 1]
    """
    
    def __init__(self, in_channels=3, base_channels=64):
        super(StipplingNetwork, self).__init__()
        
        # Encoder path
        self.enc1 = EncoderBlock(in_channels, base_channels)
        self.enc2 = EncoderBlock(base_channels, base_channels * 2)
        self.enc3 = EncoderBlock(base_channels * 2, base_channels * 4)
        self.enc4 = EncoderBlock(base_channels * 4, base_channels * 8)
        
        # Bottleneck with attention
        self.bottleneck = nn.Sequential(
            ConvBlock(base_channels * 8, base_channels * 16),
            ConvBlock(base_channels * 16, base_channels * 16),
            AttentionBlock(base_channels * 16)
        )
        
        # Decoder path
        self.dec4 = DecoderBlock(base_channels * 16, base_channels * 8)
        self.dec3 = DecoderBlock(base_channels * 8, base_channels * 4)
        self.dec2 = DecoderBlock(base_channels * 4, base_channels * 2)
        self.dec1 = DecoderBlock(base_channels * 2, base_channels)
        
        # Output layer - generates dot density map
        self.output = nn.Sequential(
            nn.Conv2d(base_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()  # Density values in [0, 1]
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input image tensor [B, 3, H, W]
            
        Returns:
            Dot density map [B, 1, H, W]
        """
        # Encoder
        x, skip1 = self.enc1(x)
        x, skip2 = self.enc2(x)
        x, skip3 = self.enc3(x)
        x, skip4 = self.enc4(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        x = self.dec4(x, skip4)
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)
        
        # Output
        x = self.output(x)
        
        return x


def create_model(in_channels=3, base_channels=64):
    """
    Factory function to create a stippling network
    
    Args:
        in_channels: Number of input channels (default: 3 for RGB)
        base_channels: Base number of channels for the network (default: 64)
        
    Returns:
        StipplingNetwork model
    """
    return StipplingNetwork(in_channels=in_channels, base_channels=base_channels)
