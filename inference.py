"""
Inference script for generating stippled images
"""

import os
import argparse
import torch
from PIL import Image

from src.models.stippling_network import create_model
from src.utils.image_processing import preprocess_image, save_image
from src.utils.rendering import create_stipple_image


def generate_stipple(model, image_path, output_path, device='cpu', method='adaptive'):
    """
    Generate stippled image from input image
    
    Args:
        model: Trained stippling network
        image_path: Path to input image
        output_path: Path to save output
        device: Device to run inference on
        method: Stippling method ('adaptive', 'threshold', 'gaussian')
    """
    # Load and preprocess image
    print(f"Loading image: {image_path}")
    input_tensor = preprocess_image(image_path).to(device)
    
    # Generate density map
    print("Generating density map...")
    model.eval()
    with torch.no_grad():
        density_map = model(input_tensor)
    
    # Save density map
    density_output = output_path.replace('.png', '_density.png')
    save_image(density_map, density_output)
    print(f"Saved density map: {density_output}")
    
    # Render stippled image
    print(f"Rendering stippled image using {method} method...")
    stippled_img = create_stipple_image(density_map, method=method)
    
    # Save stippled image
    stippled_img.save(output_path)
    print(f"Saved stippled image: {output_path}")
    
    return stippled_img


def main():
    parser = argparse.ArgumentParser(description='Generate stippled images using trained model')
    parser.add_argument('--input', '-i', type=str, required=True, help='Input image path')
    parser.add_argument('--output', '-o', type=str, required=True, help='Output image path')
    parser.add_argument('--model', '-m', type=str, default=None, help='Path to trained model checkpoint')
    parser.add_argument('--device', '-d', type=str, default='cpu', choices=['cpu', 'cuda'], help='Device to use')
    parser.add_argument('--method', type=str, default='adaptive', 
                       choices=['adaptive', 'threshold', 'gaussian'],
                       help='Stippling rendering method')
    
    args = parser.parse_args()
    
    # Check if CUDA is available
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Create model
    print("Creating model...")
    model = create_model()
    
    # Load checkpoint if provided
    if args.model and os.path.exists(args.model):
        print(f"Loading checkpoint: {args.model}")
        checkpoint = torch.load(args.model, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
    else:
        print("Warning: No checkpoint provided or file not found. Using untrained model.")
    
    model = model.to(args.device)
    
    # Generate stippled image
    generate_stipple(model, args.input, args.output, args.device, args.method)
    
    print("Done!")


if __name__ == '__main__':
    main()
