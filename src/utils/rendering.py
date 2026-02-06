"""
Rendering utilities for creating stippled images from density maps
"""

import numpy as np
import torch
import cv2
from PIL import Image, ImageDraw


def extract_dots_from_density(density_map, num_dots=5000, threshold=0.3):
    """
    Extract dot positions from a density map using non-maximum suppression
    
    Args:
        density_map: Numpy array [H, W] with values in [0, 1]
        num_dots: Target number of dots to extract
        threshold: Minimum density threshold for dot placement
        
    Returns:
        List of (x, y, radius) tuples
    """
    # Apply threshold
    mask = density_map > threshold
    
    # Get coordinates where density is above threshold
    coords = np.argwhere(mask)
    
    if len(coords) == 0:
        return []
    
    # Sample based on density values
    densities = density_map[mask]
    probabilities = densities / densities.sum()
    
    # Sample indices
    num_samples = min(num_dots, len(coords))
    indices = np.random.choice(len(coords), size=num_samples, replace=False, p=probabilities)
    
    selected_coords = coords[indices]
    selected_densities = densities[indices]
    
    # Convert to (x, y, radius) format
    dots = []
    for (y, x), density in zip(selected_coords, selected_densities):
        # Radius proportional to density (inverted - darker areas get bigger dots)
        radius = max(1, int((1 - density) * 5) + 1)
        dots.append((int(x), int(y), radius))
    
    return dots


def extract_dots_adaptive(density_map, dot_density_scale=1.0):
    """
    Extract dots with adaptive spacing based on density map
    
    Args:
        density_map: Numpy array [H, W] with values in [0, 1]
        dot_density_scale: Scale factor for number of dots (default: 1.0)
        
    Returns:
        List of (x, y, radius) tuples
    """
    h, w = density_map.shape
    dots = []
    
    # Grid-based sampling with adaptive spacing
    base_spacing = 8
    
    for i in range(0, h, base_spacing):
        for j in range(0, w, base_spacing):
            # Get local density
            local_density = density_map[i:i+base_spacing, j:j+base_spacing].mean()
            
            # Decide whether to place a dot based on density
            if np.random.random() < local_density * dot_density_scale:
                # Random position within the grid cell
                y = i + np.random.randint(0, min(base_spacing, h - i))
                x = j + np.random.randint(0, min(base_spacing, w - j))
                
                # Dot size inversely proportional to local density
                # (darker areas = higher density = smaller dots for better detail)
                radius = max(1, int((1 - local_density) * 4) + 1)
                
                dots.append((int(x), int(y), radius))
    
    return dots


def render_stipples(density_map, background_color=255, dot_color=0, method='adaptive', **kwargs):
    """
    Render stippled image from density map
    
    Args:
        density_map: Numpy array [H, W] with values in [0, 1]
        background_color: Background color (0-255)
        dot_color: Dot color (0-255)
        method: Extraction method ('adaptive' or 'threshold')
        **kwargs: Additional arguments for extraction methods
        
    Returns:
        PIL Image of the stippled result
    """
    h, w = density_map.shape
    
    # Create white background
    img = Image.new('L', (w, h), color=background_color)
    draw = ImageDraw.Draw(img)
    
    # Extract dots
    if method == 'adaptive':
        dots = extract_dots_adaptive(density_map, **kwargs)
    else:
        dots = extract_dots_from_density(density_map, **kwargs)
    
    # Draw dots
    for x, y, radius in dots:
        draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=dot_color)
    
    return img


def render_stipples_gaussian(density_map, background_color=255, dot_color=0, num_dots=5000):
    """
    Render stippled image with Gaussian-shaped dots for smoother appearance
    
    Args:
        density_map: Numpy array [H, W] with values in [0, 1]
        background_color: Background color (0-255)
        dot_color: Dot color (0-255)
        num_dots: Number of dots to place
        
    Returns:
        Numpy array of the stippled result
    """
    h, w = density_map.shape
    
    # Create background
    img = np.full((h, w), background_color, dtype=np.float32)
    
    # Extract dots
    dots = extract_dots_from_density(density_map, num_dots=num_dots)
    
    # Draw each dot with Gaussian falloff
    for x, y, radius in dots:
        # Create small Gaussian kernel
        size = radius * 3
        if size % 2 == 0:
            size += 1
        
        kernel = cv2.getGaussianKernel(size, radius / 2)
        kernel = kernel @ kernel.T
        kernel = kernel / kernel.max()
        
        # Calculate position
        y_start = max(0, y - size // 2)
        y_end = min(h, y + size // 2 + 1)
        x_start = max(0, x - size // 2)
        x_end = min(w, x + size // 2 + 1)
        
        # Calculate kernel slice
        k_y_start = max(0, size // 2 - y)
        k_y_end = k_y_start + (y_end - y_start)
        k_x_start = max(0, size // 2 - x)
        k_x_end = k_x_start + (x_end - x_start)
        
        # Blend dot
        if y_end > y_start and x_end > x_start:
            kernel_slice = kernel[k_y_start:k_y_end, k_x_start:k_x_end]
            img[y_start:y_end, x_start:x_end] -= kernel_slice * (background_color - dot_color)
    
    img = np.clip(img, dot_color, background_color).astype(np.uint8)
    return Image.fromarray(img)


def create_stipple_image(density_map, output_size=None, method='adaptive', **kwargs):
    """
    Main function to create stippled image from density map tensor
    
    Args:
        density_map: Torch tensor [1, 1, H, W] or numpy array [H, W]
        output_size: Optional (width, height) to resize output
        method: Rendering method ('adaptive', 'threshold', 'gaussian')
        **kwargs: Additional arguments for rendering methods
        
    Returns:
        PIL Image of the stippled result
    """
    # Convert to numpy if needed
    if isinstance(density_map, torch.Tensor):
        if density_map.dim() == 4:
            density_map = density_map.squeeze(0).squeeze(0)
        elif density_map.dim() == 3:
            density_map = density_map.squeeze(0)
        density_map = density_map.cpu().detach().numpy()
    
    # Ensure 2D
    if density_map.ndim == 3:
        density_map = density_map.squeeze(-1)
    
    # Render based on method
    if method == 'gaussian':
        img = render_stipples_gaussian(density_map, **kwargs)
    else:
        img = render_stipples(density_map, method=method, **kwargs)
    
    # Resize if requested
    if output_size is not None:
        img = img.resize(output_size, Image.LANCZOS)
    
    return img
