"""
Simple example demonstrating the stippling network
This example creates a stippled version of an image using an untrained model
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from PIL import Image, ImageDraw

from src.models.stippling_network import create_model
from src.utils.image_processing import preprocess_image
from src.utils.rendering import create_stipple_image


def create_sample_image(size=(512, 512), filename='sample_input.png'):
    """Create a sample image with simple shapes for testing"""
    img = Image.new('RGB', size, color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw some shapes
    # Circle
    draw.ellipse([50, 50, 200, 200], fill='black')
    
    # Rectangle
    draw.rectangle([250, 50, 400, 200], fill='gray')
    
    # Triangle (polygon)
    draw.polygon([(50, 300), (200, 300), (125, 450)], fill='darkgray')
    
    # Gradient effect (multiple circles with varying opacity)
    for i in range(10):
        gray_value = int(255 * (i / 10))
        draw.ellipse([250 + i*10, 300, 400 + i*10, 450], 
                    fill=f'rgb({gray_value}, {gray_value}, {gray_value})')
    
    img.save(filename)
    return filename


def main():
    """
    Demonstrate stippling on a sample image
    Note: This uses an untrained model, so results will be random/noise-like
    For good results, the model needs to be trained on a dataset
    """
    
    print("=" * 60)
    print("Deep Learning Stippling - Example Demo")
    print("=" * 60)
    
    # Create output directory
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('data/sample_images', exist_ok=True)
    
    # Create sample image
    print("\n1. Creating sample input image...")
    sample_path = 'data/sample_images/sample_input.png'
    create_sample_image(filename=sample_path)
    print(f"   Created: {sample_path}")
    
    # Create model
    print("\n2. Creating stippling network...")
    model = create_model()
    model.eval()
    print("   Model created (untrained)")
    
    # Process image
    print("\n3. Processing image...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   Using device: {device}")
    
    model = model.to(device)
    input_tensor = preprocess_image(sample_path, target_size=512).to(device)
    
    # Generate density map
    print("\n4. Generating density map...")
    with torch.no_grad():
        density_map = model(input_tensor)
    
    # Save density map
    density_output = 'outputs/density_map.png'
    density_img = density_map.squeeze().cpu().numpy()
    density_img = (density_img * 255).astype(np.uint8)
    Image.fromarray(density_img).save(density_output)
    print(f"   Saved: {density_output}")
    
    # Render stippled images with different methods
    print("\n5. Rendering stippled images...")
    
    methods = ['adaptive', 'threshold', 'gaussian']
    for method in methods:
        print(f"   Rendering with {method} method...")
        stippled = create_stipple_image(density_map, method=method)
        output_path = f'outputs/stippled_{method}.png'
        stippled.save(output_path)
        print(f"   Saved: {output_path}")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
    print("\nOutputs saved in 'outputs/' directory:")
    print("  - density_map.png: Raw density map from network")
    print("  - stippled_adaptive.png: Adaptive dot placement")
    print("  - stippled_threshold.png: Threshold-based dot placement")
    print("  - stippled_gaussian.png: Gaussian-smoothed dots")
    print("\nNote: This example uses an UNTRAINED model.")
    print("For good stippling results, train the model using train.py")
    print("on a dataset of images.")
    print("=" * 60)


if __name__ == '__main__':
    main()
