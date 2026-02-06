"""
Stippling Neural Network (Exact Paper Implementation)
Paper: A Deep Learning Method for 2D Image Stippling (Li et al., 2021)
Architecture: ResNet50 Encoder -> MLP Style Mapping -> AdaIN Grid Decoder (2D) -> Generator
1 point per cell, 32x32 grid = 1024 points total
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


class AdaIN(nn.Module):
    """
    Adaptive Instance Normalization (2D)
    """
    def __init__(self, channels, style_dim):
        super(AdaIN, self).__init__()
        self.instance_norm = nn.InstanceNorm2d(channels)
        self.style_scale = nn.Linear(style_dim, channels)
        self.style_bias = nn.Linear(style_dim, channels)

    def forward(self, x, style):
        gamma = self.style_scale(style).view(x.size(0), x.size(1), 1, 1)
        beta = self.style_bias(style).view(x.size(0), x.size(1), 1, 1)
        return gamma * self.instance_norm(x) + beta


class GridDecoderBlock(nn.Module):
    """
    Upsample -> Conv2D -> AdaIN -> ReLU
    """
    def __init__(self, in_channels, out_channels, style_dim):
        super(GridDecoderBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.adain = AdaIN(out_channels, style_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, style):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv(x)
        x = self.relu(self.adain(x, style))
        return x


class StipplingNetwork(nn.Module):
    def __init__(self, num_points=1024, grid_size=32, style_dim=1024):
        super(StipplingNetwork, self).__init__()
        self.num_points = num_points
        self.grid_size = grid_size
        
        # --- ENCODER: ResNet50 ---
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        for param in self.encoder.parameters():
            param.requires_grad = False  # Frozen as per paper implication
        self.fc_z = nn.Linear(2048, style_dim)

        # --- STYLE MLP ---
        self.style_mlp = nn.Sequential(
            nn.Linear(style_dim, style_dim), nn.ReLU(),
            nn.Linear(style_dim, style_dim), nn.ReLU(),
            nn.Linear(style_dim, style_dim), nn.ReLU(),
        )

        # --- GRID DECODER ---
        # Starts at 4x4 spatial resolution
        self.const_input = nn.Parameter(torch.randn(1, 256, 4, 4))
        
        self.block1 = GridDecoderBlock(256, 128, style_dim)  # 4 -> 8
        self.block2 = GridDecoderBlock(128, 64, style_dim)   # 8 -> 16
        self.block3 = GridDecoderBlock(64, 32, style_dim)    # 16 -> 32

        # --- GENERATOR HEADS ---
        # No 3D tricks. Standard 2D convs on the 32x32 grid.
        
        # Density: predicts probability (paper mentions it, we output it but don't filter by it yet)
        self.density_head = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1), nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Location: dx, dy (Strict Sigmoid 0-1 range)
        self.location_head = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1), nn.ReLU(),
            nn.Conv2d(64, 2, kernel_size=1),
            nn.Sigmoid() 
        )
        
        # Color: r, g, b
        self.color_head = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1), nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=1),
            nn.Sigmoid()
        )

        self.decoder = nn.ModuleList([self.style_mlp, self.block1, self.block2, self.block3,
                                      self.density_head, self.location_head, self.color_head])

    def forward(self, x):
        B = x.size(0)
        
        # Encode
        z = self.fc_z(self.encoder(x).flatten(1))
        w = self.style_mlp(z)
        
        # Decode
        grid = self.const_input.repeat(B, 1, 1, 1)
        grid = self.block1(grid, w)
        grid = self.block2(grid, w)
        grid = self.block3(grid, w)  # [B, 32, 32, 32]
        
        # Predict
        density = self.density_head(grid)
        offsets = self.location_head(grid)
        colors = self.color_head(grid)
        
        return self.grid_to_points(offsets, colors)

    def grid_to_points(self, offsets, colors):
        """
        Direct mapping: 1 cell = 1 point.
        No Top-K. No Selection.
        Total points = 32*32 = 1024.
        """
        B = offsets.shape[0]
        device = offsets.device
        H, W = 32, 32
        
        # Generate Grid Anchors
        gy, gx = torch.meshgrid(torch.arange(H, device=device), 
                                torch.arange(W, device=device), indexing='ij')
        
        gx = gx.float().view(1, 1, H, W)
        gy = gy.float().view(1, 1, H, W)
        
        # Global Coordinates (Strict Sigmoid Logic)
        # global = (anchor + offset) / size
        global_x = (gx + offsets[:, 0:1, :, :]) / W
        global_y = (gy + offsets[:, 1:2, :, :]) / H
        
        # Flatten
        flat_x = global_x.view(B, -1)
        flat_y = global_y.view(B, -1)
        flat_r = colors[:, 0, :, :].view(B, -1)
        flat_g = colors[:, 1, :, :].view(B, -1)
        flat_b = colors[:, 2, :, :].view(B, -1)
        
        return torch.stack([flat_x, flat_y, flat_r, flat_g, flat_b], dim=2)


def create_model(num_points=1024, grid_size=32):
    """Factory function to create the stippling network"""
    return StipplingNetwork(num_points=num_points, grid_size=grid_size)
