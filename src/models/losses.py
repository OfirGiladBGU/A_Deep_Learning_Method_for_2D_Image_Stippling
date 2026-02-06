"""
Loss functions for training the stippling network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG features
    Measures perceptual similarity between generated and target images
    """
    
    def __init__(self, feature_layers=[3, 8, 15, 22]):
        super(PerceptualLoss, self).__init__()
        
        # Load pretrained VGG16
        try:
            # Try new API first
            from torchvision.models import VGG16_Weights
            vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        except (ImportError, AttributeError):
            # Fall back to older API if VGG16_Weights not available
            import warnings
            warnings.warn("Using deprecated pretrained parameter. Consider updating torchvision.", DeprecationWarning)
            vgg = models.vgg16(pretrained=True).features
        
        # Extract feature extraction layers
        self.feature_extractors = nn.ModuleList()
        prev_layer = 0
        
        for layer_idx in feature_layers:
            self.feature_extractors.append(vgg[prev_layer:layer_idx+1])
            prev_layer = layer_idx + 1
        
        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False
        
        self.feature_layers = feature_layers
    
    def forward(self, generated, target):
        """
        Calculate perceptual loss
        
        Args:
            generated: Generated density map [B, 1, H, W]
            target: Target image [B, 3, H, W]
            
        Returns:
            Perceptual loss value
        """
        # Convert grayscale to RGB by repeating channels
        if generated.size(1) == 1:
            generated = generated.repeat(1, 3, 1, 1)
        
        loss = 0.0
        x_gen = generated
        x_tar = target
        
        # Extract features from each layer
        for extractor in self.feature_extractors:
            x_gen = extractor(x_gen)
            x_tar = extractor(x_tar)
            
            # Calculate L2 loss between features
            loss += F.mse_loss(x_gen, x_tar)
        
        return loss / len(self.feature_extractors)


class StipplingLoss(nn.Module):
    """
    Combined loss for stippling network training
    Combines reconstruction loss with perceptual loss
    """
    
    def __init__(self, perceptual_weight=0.1, use_perceptual=True):
        super(StipplingLoss, self).__init__()
        
        self.use_perceptual = use_perceptual
        self.perceptual_weight = perceptual_weight
        
        # Reconstruction loss (L1 is better for preserving details)
        self.l1_loss = nn.L1Loss()
        
        # Perceptual loss
        if self.use_perceptual:
            self.perceptual_loss = PerceptualLoss()
    
    def forward(self, generated, target_gray, target_rgb=None):
        """
        Calculate combined loss
        
        Args:
            generated: Generated density map [B, 1, H, W]
            target_gray: Target grayscale image [B, 1, H, W]
            target_rgb: Optional target RGB image for perceptual loss [B, 3, H, W]
            
        Returns:
            Total loss, dict of individual losses
        """
        # Reconstruction loss (L1)
        recon_loss = self.l1_loss(generated, target_gray)
        
        total_loss = recon_loss
        losses = {'reconstruction': recon_loss.item()}
        
        # Perceptual loss
        if self.use_perceptual and target_rgb is not None:
            perc_loss = self.perceptual_loss(generated, target_rgb)
            total_loss += self.perceptual_weight * perc_loss
            losses['perceptual'] = perc_loss.item()
        
        losses['total'] = total_loss.item()
        
        return total_loss, losses


class GradientLoss(nn.Module):
    """
    Gradient loss to preserve edges and details
    """
    
    def __init__(self):
        super(GradientLoss, self).__init__()
    
    def forward(self, generated, target):
        """
        Calculate gradient loss
        
        Args:
            generated: Generated density map [B, 1, H, W]
            target: Target image [B, 1, H, W]
            
        Returns:
            Gradient loss value
        """
        # Sobel filters for gradients
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=generated.dtype, device=generated.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               dtype=generated.dtype, device=generated.device).view(1, 1, 3, 3)
        
        # Calculate gradients for generated
        gen_grad_x = F.conv2d(generated, sobel_x, padding=1)
        gen_grad_y = F.conv2d(generated, sobel_y, padding=1)
        gen_grad = torch.sqrt(gen_grad_x ** 2 + gen_grad_y ** 2 + 1e-6)
        
        # Calculate gradients for target
        tar_grad_x = F.conv2d(target, sobel_x, padding=1)
        tar_grad_y = F.conv2d(target, sobel_y, padding=1)
        tar_grad = torch.sqrt(tar_grad_x ** 2 + tar_grad_y ** 2 + 1e-6)
        
        # L1 loss on gradients
        return F.l1_loss(gen_grad, tar_grad)


class CombinedStipplingLoss(nn.Module):
    """
    Advanced combined loss with reconstruction, perceptual, and gradient components
    """
    
    def __init__(self, recon_weight=1.0, perceptual_weight=0.1, gradient_weight=0.5):
        super(CombinedStipplingLoss, self).__init__()
        
        self.recon_weight = recon_weight
        self.perceptual_weight = perceptual_weight
        self.gradient_weight = gradient_weight
        
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = PerceptualLoss()
        self.gradient_loss = GradientLoss()
    
    def forward(self, generated, target_gray, target_rgb):
        """
        Calculate combined loss with all components
        
        Args:
            generated: Generated density map [B, 1, H, W]
            target_gray: Target grayscale image [B, 1, H, W]
            target_rgb: Target RGB image [B, 3, H, W]
            
        Returns:
            Total loss, dict of individual losses
        """
        # Reconstruction loss
        recon_loss = self.l1_loss(generated, target_gray)
        
        # Perceptual loss
        perc_loss = self.perceptual_loss(generated, target_rgb)
        
        # Gradient loss
        grad_loss = self.gradient_loss(generated, target_gray)
        
        # Combined loss
        total_loss = (self.recon_weight * recon_loss + 
                     self.perceptual_weight * perc_loss + 
                     self.gradient_weight * grad_loss)
        
        losses = {
            'reconstruction': recon_loss.item(),
            'perceptual': perc_loss.item(),
            'gradient': grad_loss.item(),
            'total': total_loss.item()
        }
        
        return total_loss, losses
