"""
Image processing utilities for loading, saving, and preprocessing images
"""

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


def load_image(image_path, size=None):
    """
    Load an image from file
    
    Args:
        image_path: Path to the image file
        size: Optional tuple (width, height) to resize the image
        
    Returns:
        PIL Image object
    """
    img = Image.open(image_path).convert('RGB')
    
    if size is not None:
        img = img.resize(size, Image.BILINEAR)
    
    return img


def save_image(image, save_path):
    """
    Save an image to file
    
    Args:
        image: PIL Image, numpy array, or torch tensor
        save_path: Path where to save the image
    """
    if isinstance(image, torch.Tensor):
        # Convert tensor to numpy
        if image.dim() == 4:
            image = image.squeeze(0)
        image = image.cpu().detach().numpy()
        if image.shape[0] in [1, 3]:  # Channel first
            image = np.transpose(image, (1, 2, 0))
        image = (image * 255).astype(np.uint8)
    
    if isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[2] == 1:
            image = image.squeeze(2)
        image = Image.fromarray(image)
    
    image.save(save_path)


def preprocess_image(image, target_size=512):
    """
    Preprocess image for network input
    
    Args:
        image: PIL Image or file path
        target_size: Target size for the longest edge (default: 512)
        
    Returns:
        Preprocessed tensor [1, 3, H, W]
    """
    if isinstance(image, str):
        image = load_image(image)
    
    # Calculate new size maintaining aspect ratio
    w, h = image.size
    if w > h:
        new_w = target_size
        new_h = int(h * target_size / w)
    else:
        new_h = target_size
        new_w = int(w * target_size / h)
    
    # Make dimensions divisible by 16 (for U-Net)
    new_w = (new_w // 16) * 16
    new_h = (new_h // 16) * 16
    
    # Transform pipeline
    transform = transforms.Compose([
        transforms.Resize((new_h, new_w), transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    tensor = transform(image).unsqueeze(0)
    return tensor


def postprocess_density_map(density_map):
    """
    Convert density map tensor to displayable image
    
    Args:
        density_map: Tensor [1, 1, H, W] or [1, H, W]
        
    Returns:
        PIL Image
    """
    if density_map.dim() == 4:
        density_map = density_map.squeeze(0).squeeze(0)
    elif density_map.dim() == 3:
        density_map = density_map.squeeze(0)
    
    density_map = density_map.cpu().detach().numpy()
    density_map = (density_map * 255).astype(np.uint8)
    
    return Image.fromarray(density_map)


def tensor_to_numpy(tensor):
    """
    Convert tensor to numpy array for visualization
    
    Args:
        tensor: PyTorch tensor
        
    Returns:
        Numpy array in [0, 1] range
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    array = tensor.cpu().detach().numpy()
    
    if array.shape[0] in [1, 3]:  # Channel first
        array = np.transpose(array, (1, 2, 0))
    
    if array.shape[-1] == 1:
        array = array.squeeze(-1)
    
    # Normalize to [0, 1]
    array = np.clip(array, 0, 1)
    
    return array
