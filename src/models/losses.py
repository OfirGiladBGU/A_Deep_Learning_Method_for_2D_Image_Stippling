"""
Loss functions for Deep Learning 2D Image Stippling (Li et al., 2021)
Paper losses:
- Chamfer Loss (Lc): Geometry matching between predicted and GT point sets
- Radial Spectrum Loss (Ls): Blue noise distribution quality
- Total: L = Lc + w * Ls (w=0.1 in paper)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ChamferLoss(nn.Module):
    """
    Chamfer Distance Loss for Point Sets.
    Computes bidirectional nearest neighbor distances between two point clouds.
    
    For each point in pred, find nearest in GT (and vice versa).
    L = mean(min_dist_pred_to_gt) + mean(min_dist_gt_to_pred)
    """
    def __init__(self):
        super(ChamferLoss, self).__init__()

    def forward(self, pred_points, gt_points):
        """
        Args:
            pred_points: [B, N, 2] predicted coordinates
            gt_points: [B, M, 2] ground truth coordinates
            
        Returns:
            Chamfer distance (scalar)
        """
        # Expand for pairwise distance computation
        # pred: [B, N, 1, 2], gt: [B, 1, M, 2]
        x = pred_points.unsqueeze(2)
        y = gt_points.unsqueeze(1)
        
        # Squared Euclidean distance: [B, N, M]
        dist = torch.pow(x - y, 2).sum(dim=-1)
        
        # For each predicted point, find nearest GT point
        min_dist_pred, _ = torch.min(dist, dim=2)  # [B, N]
        
        # For each GT point, find nearest predicted point
        min_dist_gt, _ = torch.min(dist, dim=1)    # [B, M]
        
        # Chamfer distance = mean of both directions
        chamfer = torch.mean(min_dist_pred) + torch.mean(min_dist_gt)
        
        return chamfer


class RadialSpectrumLoss(nn.Module):
    """
    Radial Power Spectrum Loss for Blue Noise quality.
    
    Computes the 1D radial power spectrum of the point distribution
    and compares it to an ideal Blue Noise spectrum.
    
    Ls = ||RS(X) - RS(Y)||_1
    """
    def __init__(self, image_size=256, num_bins=64):
        super(RadialSpectrumLoss, self).__init__()
        self.image_size = image_size
        self.num_bins = num_bins
        
        # Pre-compute ideal Blue Noise radial spectrum
        # Real Blue Noise has:
        # - Low energy at low frequencies (no clustering)
        # - Flat energy at medium/high frequencies
        self.register_buffer('ideal_spectrum', self._create_ideal_spectrum())
        
        # Pre-compute radius indices for radial averaging
        self._setup_radial_indices()
    
    def _create_ideal_spectrum(self):
        """
        Create ideal Blue Noise radial spectrum.
        Characteristics: suppressed low frequencies, flat high frequencies.
        """
        spectrum = torch.ones(self.num_bins)
        
        # Suppress low frequencies (the hallmark of Blue Noise)
        cutoff = self.num_bins // 4
        for i in range(cutoff):
            spectrum[i] = (i / cutoff) ** 2
        
        # Normalize
        spectrum = spectrum / spectrum.sum()
        return spectrum
    
    def _setup_radial_indices(self):
        """Pre-compute radius indices for efficient radial averaging"""
        H, W = self.image_size, self.image_size
        cy, cx = H // 2, W // 2
        
        y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        r = torch.sqrt((x - cx).float() ** 2 + (y - cy).float() ** 2)
        
        # Scale to bin indices
        max_r = np.sqrt(cx ** 2 + cy ** 2)
        r_bins = (r / max_r * (self.num_bins - 1)).long().clamp(0, self.num_bins - 1)
        
        self.register_buffer('radius_bins', r_bins.flatten())
    
    def soft_rasterize(self, points):
        """
        Differentiable soft rasterization of points onto a grid.
        Uses Gaussian splatting for smooth gradients.
        
        Args:
            points: [B, N, 2] in range [0, 1]
            
        Returns:
            density: [B, H, W] soft density map
        """
        B, N, _ = points.shape
        H, W = self.image_size, self.image_size
        device = points.device
        
        # Create coordinate grids
        x_grid = torch.linspace(0, 1, W, device=device).view(1, 1, 1, W)
        y_grid = torch.linspace(0, 1, H, device=device).view(1, 1, H, 1)
        
        # Point coordinates
        px = points[:, :, 0].view(B, N, 1, 1)
        py = points[:, :, 1].view(B, N, 1, 1)
        
        # Gaussian splatting
        sigma = 2.0 / W  # Spread based on image size
        dist_sq = (x_grid - px) ** 2 + (y_grid - py) ** 2
        gaussians = torch.exp(-dist_sq / (2 * sigma ** 2))
        
        # Sum contributions from all points
        density = torch.sum(gaussians, dim=1)  # [B, H, W]
        
        return density
    
    def compute_radial_spectrum(self, points):
        """
        Compute the 1D radial power spectrum of a point set.
        
        Steps:
        1. Soft rasterize points to density map
        2. Apply FFT
        3. Compute radial average of power spectrum
        """
        B = points.shape[0]
        device = points.device
        
        # 1. Soft rasterize
        density = self.soft_rasterize(points)
        
        # 2. FFT and power spectrum
        fft = torch.fft.fft2(density)
        fft_shifted = torch.fft.fftshift(fft)
        power = torch.abs(fft_shifted) ** 2  # [B, H, W]
        
        # 3. Radial average
        power_flat = power.view(B, -1)  # [B, H*W]
        
        # Accumulate power in radial bins
        radial_spectrum = torch.zeros(B, self.num_bins, device=device)
        bin_counts = torch.zeros(self.num_bins, device=device)
        
        # Efficient scatter_add for radial binning
        for i in range(self.num_bins):
            mask = (self.radius_bins == i)
            radial_spectrum[:, i] = power_flat[:, mask].mean(dim=1)
        
        # Normalize spectrum
        radial_spectrum = radial_spectrum / (radial_spectrum.sum(dim=1, keepdim=True) + 1e-8)
        
        return radial_spectrum
    
    def forward(self, pred_points, gt_points=None):
        """
        Compute radial spectrum loss.
        
        Args:
            pred_points: [B, N, 2] predicted coordinates
            gt_points: [B, M, 2] optional GT points (if None, compare to ideal)
            
        Returns:
            L1 loss between radial spectra
        """
        pred_spectrum = self.compute_radial_spectrum(pred_points[:, :, :2])
        
        if gt_points is not None:
            target_spectrum = self.compute_radial_spectrum(gt_points[:, :, :2])
        else:
            target_spectrum = self.ideal_spectrum.unsqueeze(0).expand(pred_spectrum.shape[0], -1)
        
        return F.l1_loss(pred_spectrum, target_spectrum)


class ColorLoss(nn.Module):
    """
    Color matching loss.
    Compares predicted point colors to the actual image colors at those locations.
    """
    def __init__(self):
        super(ColorLoss, self).__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, pred_points, input_image):
        """
        Args:
            pred_points: [B, N, 5] with (x, y, r, g, b)
            input_image: [B, 3, H, W] input image (0-1 range)
            
        Returns:
            MSE loss between predicted and sampled colors
        """
        coords = pred_points[:, :, :2]  # [B, N, 2]
        pred_colors = pred_points[:, :, 2:]  # [B, N, 3]
        
        # Convert coords to grid_sample format (-1 to 1)
        grid_coords = coords.view(coords.shape[0], coords.shape[1], 1, 2) * 2 - 1
        
        # Sample image colors at point locations
        sampled = F.grid_sample(input_image, grid_coords, align_corners=True, mode='bilinear')
        sampled_colors = sampled.view(coords.shape[0], 3, -1).permute(0, 2, 1)  # [B, N, 3]
        
        return self.mse(pred_colors, sampled_colors)


class StipplingLoss(nn.Module):
    """
    Combined loss for supervised stippling training.
    
    L_total = w_chamfer * L_chamfer + w_spectrum * L_spectrum + w_color * L_color
    
    Paper defaults:
    - w = 0.1 for spectrum loss relative to chamfer
    """
    def __init__(self, weight_chamfer=1.0, weight_spectrum=0.1, weight_color=1.0,
                 image_size=256, grid_size=64):
        super(StipplingLoss, self).__init__()
        
        self.weight_chamfer = weight_chamfer
        self.weight_spectrum = weight_spectrum
        self.weight_color = weight_color
        
        self.chamfer_loss = ChamferLoss()
        self.spectrum_loss = RadialSpectrumLoss(image_size=image_size)
        self.color_loss = ColorLoss()
    
    def forward(self, pred_points, gt_points, input_image):
        """
        Args:
            pred_points: [B, N, 5] predicted (x, y, r, g, b)
            gt_points: [B, M, 5] ground truth (x, y, r, g, b)
            input_image: [B, 3, H, W] raw input image (0-1 range)
            
        Returns:
            total_loss, dict of component losses
        """
        # 1. Chamfer loss on coordinates
        loss_chamfer = self.chamfer_loss(pred_points[:, :, :2], gt_points[:, :, :2])
        
        # 2. Radial spectrum loss (compare to GT spectrum)
        loss_spectrum = self.spectrum_loss(pred_points[:, :, :2], gt_points[:, :, :2])
        
        # 3. Color loss
        loss_color = self.color_loss(pred_points, input_image)
        
        # Combined loss
        total_loss = (self.weight_chamfer * loss_chamfer +
                      self.weight_spectrum * loss_spectrum +
                      self.weight_color * loss_color)
        
        components = {
            'chamfer': loss_chamfer.item(),
            'spectrum': loss_spectrum.item(),
            'color': loss_color.item(),
            'total': total_loss.item()
        }
        
        return total_loss, components
