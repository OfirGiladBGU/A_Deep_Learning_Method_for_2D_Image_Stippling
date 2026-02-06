"""
Visualization script to show sample stippling results
"""

import os
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms

from src.models.stippling_network import create_model
from src.utils.image_processing import denormalize_image


def load_model(checkpoint_path, num_points=2048, device='cpu'):
    """Load trained model from checkpoint"""
    model = create_model(num_points=num_points)
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
    
    return model.to(device)


def extract_gt_points(target_path):
    """Extract ground truth points from binary target image"""
    target = Image.open(target_path).convert('L')
    target_arr = np.array(target)
    orig_size = target.size[0]
    
    black_pixels = np.where(target_arr < 10)
    y_coords = black_pixels[0] / orig_size
    x_coords = black_pixels[1] / orig_size
    
    return x_coords, y_coords


def visualize_samples(model, data_dir, output_dir, num_samples=5, device='cpu'):
    """Generate and visualize stippling results"""
    os.makedirs(output_dir, exist_ok=True)
    
    source_dir = os.path.join(data_dir, 'source')
    target_dir = os.path.join(data_dir, 'target')
    
    # Get sample images
    image_files = [f for f in os.listdir(source_dir) if f.endswith('.png')][:num_samples]
    
    # VGG preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    model.eval()
    
    for i, filename in enumerate(image_files):
        source_path = os.path.join(source_dir, filename)
        target_path = os.path.join(target_dir, filename)
        
        # Load source image
        source_img = Image.open(source_path).convert('RGB')
        input_tensor = transform(source_img).unsqueeze(0).to(device)
        
        # Generate predicted points
        with torch.no_grad():
            pred_points = model(input_tensor)
        
        pred_points = pred_points[0].cpu().numpy()
        pred_x = pred_points[:, 0]
        pred_y = pred_points[:, 1]
        pred_colors = pred_points[:, 2:]
        
        # Get ground truth points
        gt_x, gt_y = extract_gt_points(target_path)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 1. Original image
        axes[0].imshow(source_img)
        axes[0].set_title('Source Image')
        axes[0].axis('off')
        
        # 2. Ground truth stipples
        axes[1].set_facecolor('white')
        axes[1].scatter(gt_x * 224, (1 - gt_y) * 224, c='black', s=5)
        axes[1].set_xlim(0, 224)
        axes[1].set_ylim(0, 224)
        axes[1].set_title(f'Ground Truth ({len(gt_x)} points)')
        axes[1].axis('off')
        axes[1].set_aspect('equal')
        
        # 3. Predicted stipples
        axes[2].set_facecolor('white')
        # Use grayscale for colors (average of RGB)
        gray_colors = pred_colors.mean(axis=1)
        axes[2].scatter(pred_x * 224, (1 - pred_y) * 224, c=gray_colors, cmap='gray_r', s=5, vmin=0, vmax=1)
        axes[2].set_xlim(0, 224)
        axes[2].set_ylim(0, 224)
        axes[2].set_title(f'Predicted ({len(pred_x)} points)')
        axes[2].axis('off')
        axes[2].set_aspect('equal')
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(output_dir, f'sample_{i+1}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Saved: {output_path}")
    
    print(f"\nGenerated {num_samples} sample visualizations in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Visualize stippling results')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/stipple_net_10.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to data directory with source/ and target/ folders')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory for visualizations')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to visualize')
    parser.add_argument('--points', type=int, default=2048,
                        help='Number of stipple points')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.checkpoint, num_points=args.points, device=device)
    
    # Generate visualizations
    visualize_samples(model, args.data, args.output, args.num_samples, device)


if __name__ == '__main__':
    main()
