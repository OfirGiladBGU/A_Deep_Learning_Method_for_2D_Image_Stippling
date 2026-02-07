"""
Stippling Neural Network (Li et al., 2021)
Architecture: ResNet50 Encoder -> Style MLP -> AdaIN 3D Grid Decoder -> Generator
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


class AdaIN(nn.Module):
    def __init__(self, channels, style_dim):
        super().__init__()
        self.instance_norm = nn.InstanceNorm3d(channels)
        self.style_scale = nn.Linear(style_dim, channels)
        self.style_bias = nn.Linear(style_dim, channels)

    def forward(self, x, style):
        gamma = self.style_scale(style).view(x.size(0), x.size(1), 1, 1, 1) + 1.0
        beta = self.style_bias(style).view(x.size(0), x.size(1), 1, 1, 1)
        return gamma * self.instance_norm(x) + beta


class GridConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, style_dim, dropout=0.2, is_last=False):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.adain = AdaIN(out_channels, style_dim)
        self.is_last = is_last

    def forward(self, x, style):
        x = self.conv(x)
        x = self.dropout(x)
        x = self.adain(x, style)
        if not self.is_last:
            x = F.elu(x)
        return x


class StipplingNetwork(nn.Module):
    def __init__(self, num_points=2500, grid_size=32, style_dim=1024):
        super().__init__()
        self.num_points = num_points
        self.grid_size = grid_size

        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.fc_z = nn.Linear(2048, style_dim)

        self.style_mlp = nn.Sequential(
            nn.Linear(style_dim, style_dim), nn.ReLU(),
            nn.Linear(style_dim, style_dim), nn.ReLU(),
            nn.Linear(style_dim, style_dim), nn.ReLU(),
        )

        self.P = nn.Parameter(torch.randn(1, 512, 2, 2, 2))
        self.dropout_p = nn.Dropout(0.2)
        self.adain_p = AdaIN(512, style_dim)

        self.layer_c512_a = GridConvBlock(512, 512, style_dim)
        self.layer_c512_b = GridConvBlock(512, 512, style_dim)
        self.layer_c256_a = GridConvBlock(512, 256, style_dim)
        self.layer_c256_b = GridConvBlock(256, 256, style_dim)
        self.layer_c128_a = GridConvBlock(256, 128, style_dim)
        self.layer_c128_b = GridConvBlock(128, 128, style_dim)
        self.layer_c64_a = GridConvBlock(128, 64, style_dim)
        self.layer_c64_b = GridConvBlock(64, 64, style_dim)
        self.layer_c62 = GridConvBlock(64, 62, style_dim, is_last=True)

        self.density_mlp = nn.Sequential(
            nn.Conv3d(62, 16, 1), nn.ELU(),
            nn.Conv3d(16, 8, 1), nn.ELU(),
            nn.Conv3d(8, 4, 1), nn.ELU(),
            nn.Conv3d(4, 1, 1),
        )
        nn.init.constant_(self.density_mlp[-1].bias, 2.0)

        self.location_mlp = nn.Sequential(
            nn.Conv3d(62, 64, 1), nn.ELU(),
            nn.Conv3d(64, 64, 1), nn.ELU(),
            nn.Conv3d(64, 32, 1), nn.ELU(),
            nn.Conv3d(32, 32, 1), nn.ELU(),
            nn.Conv3d(32, 16, 1), nn.ELU(),
            nn.Conv3d(16, 16, 1), nn.ELU(),
            nn.Conv3d(16, 8, 1), nn.ELU(),
            nn.Conv3d(8, 3, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        B = x.size(0)
        z = self.fc_z(self.encoder(x).flatten(1))
        w = self.style_mlp(z)

        out = self.P.repeat(B, 1, 1, 1, 1)
        out = self.dropout_p(out)
        out = F.elu(self.adain_p(out, w))

        out = self.layer_c512_a(out, w)

        out = F.interpolate(out, scale_factor=2)
        out = self.layer_c512_b(out, w)
        out = self.layer_c256_a(out, w)

        out = F.interpolate(out, scale_factor=2)
        out = self.layer_c256_b(out, w)
        out = self.layer_c128_a(out, w)

        out = F.interpolate(out, scale_factor=2)
        out = self.layer_c128_b(out, w)
        out = self.layer_c64_a(out, w)

        out = F.interpolate(out, scale_factor=2)
        out = self.layer_c64_b(out, w)
        grid_features = self.layer_c62(out, w)

        density_logits = self.density_mlp(grid_features)
        loc_color = self.location_mlp(grid_features)

        return self.projection(density_logits, loc_color)

    def projection(self, density_logits, loc_color):
        B = density_logits.shape[0]
        device = density_logits.device

        flat_density = torch.sigmoid(density_logits).view(B, -1)
        flat_loc = loc_color.view(B, 3, -1)

        H = self.grid_size
        W = self.grid_size
        D = self.grid_size
        gy, gx = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij",
        )
        gx = gx.unsqueeze(-1).expand(-1, -1, D).contiguous().view(1, 1, -1)
        gy = gy.unsqueeze(-1).expand(-1, -1, D).contiguous().view(1, 1, -1)

        global_x = (gx + flat_loc[:, 0:1, :]) / W
        global_y = (gy + flat_loc[:, 1:2, :]) / H

        _, indices = torch.topk(flat_density, self.num_points, dim=1)
        batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(-1, self.num_points)

        final_x = global_x[batch_indices, 0, indices]
        final_y = global_y[batch_indices, 0, indices]
        final_c = flat_loc[batch_indices, 2, indices]

        final_r = final_c
        final_g = final_c
        final_b = final_c

        return torch.stack([final_x, final_y, final_r, final_g, final_b], dim=2)


def create_model(num_points=2500, grid_size=32):
    return StipplingNetwork(num_points=num_points, grid_size=grid_size)
