"""
Image processing utilities for VGG-19 Stippling Network
"""

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

def load_image(image_path, target_size=224):
    """
    Load and preprocess an image for VGG-19.
    Args:
        image_path: Path to the image file.
        target_size: 224 is standard for VGG-19.
    Returns:
        Tensor [1, 3, 224, 224] normalized for ImageNet.
    """
    img = Image.open(image_path).convert('RGB')
    
    # Standard VGG-19 Preprocessing
    transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        # Normalize with ImageNet mean/std (Required for pre-trained VGG)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    return transform(img).unsqueeze(0)

def denormalize_image(tensor):
    """
    Convert VGG-normalized tensor back to viewable image (for debugging/loss calc).
    """
    # Undo the VGG normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(tensor.device)
    
    img = tensor * std + mean
    return torch.clamp(img, 0, 1)
