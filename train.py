"""
Training script for Deep Learning 2D Image Stippling (Li et al., 2021)
Uses supervised learning with source/target pairs
Architecture: ResNet50 + AdaIN Grid Decoder
"""

import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

from config import Config
from src.models.stippling_network import create_model
from src.models.losses import StipplingLoss
from src.utils.image_processing import denormalize_image


class StipplingDataset(Dataset):
    """
    Dataset for supervised stippling training.
    - Source: grayscale images (input to network)
    - Target: binary images with black dots (ground truth point positions)
    """
    def __init__(self, data_dir, image_size=Config.IMAGE_SIZE, num_points=Config.NUM_POINTS):
        self.source_dir = os.path.join(data_dir, 'source')
        self.target_dir = os.path.join(data_dir, 'target')
        self.image_size = image_size
        self.num_points = num_points
        
        # Get all source images
        self.image_files = [f for f in os.listdir(self.source_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # ImageNet preprocessing for ResNet50
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.image_files)
    
    def extract_points_from_target(self, target_path, source_img):
        """
        Extract point coordinates from binary target image.
        Returns [N, 5] tensor with (x, y, r, g, b) normalized to [0,1]
        """
        # Load target and find black pixels
        target = Image.open(target_path).convert('L')
        target_arr = np.array(target)
        width, height = target.size
        
        # Find black pixels (stipple points)
        black_pixels = np.where(target_arr < 10)
        y_coords = black_pixels[0]
        x_coords = black_pixels[1]
        
        # Normalize to [0, 1]
        x_norm = x_coords / max(width, 1)
        y_norm = y_coords / max(height, 1)
        
        # Sample or pad to exactly num_points
        n_found = len(x_coords)
        if n_found == 0:
            return torch.zeros((self.num_points, 5), dtype=torch.float32)

        replace = n_found < self.num_points
        indices = np.random.choice(n_found, self.num_points, replace=replace)
        
        x_norm = x_norm[indices]
        y_norm = y_norm[indices]
        
        # Get colors from source image at these coordinates
        source_arr = np.array(source_img.resize((width, height)))
        
        if source_arr.ndim == 2:
            colors = source_arr[y_coords[indices], x_coords[indices]]
            colors = np.stack([colors, colors, colors], axis=1) / 255.0
        else:
            colors = source_arr[y_coords[indices], x_coords[indices]] / 255.0
        
        # Stack into [N, 5]: (x, y, r, g, b)
        points = np.zeros((self.num_points, 5), dtype=np.float32)
        points[:, 0] = x_norm
        points[:, 1] = y_norm
        points[:, 2:] = colors
        
        return torch.from_numpy(points)
    
    def __getitem__(self, idx):
        filename = self.image_files[idx]
        source_path = os.path.join(self.source_dir, filename)
        target_path = os.path.join(self.target_dir, filename)
        
        try:
            source_img = Image.open(source_path).convert('RGB')
            gt_points = self.extract_points_from_target(target_path, source_img)
            input_tensor = self.transform(source_img)
            return input_tensor, gt_points
            
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            return torch.zeros((3, self.image_size, self.image_size)), \
                   torch.zeros((self.num_points, 5))


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(dataloader, desc="Training")
    
    for images, gt_points in pbar:
        images = images.to(device)
        gt_points = gt_points.to(device)
        
        # Forward pass
        pred_points = model(images)
        
        # Denormalize images for color loss
        raw_images = denormalize_image(images)
        
        # Calculate loss
        loss, components = criterion(pred_points, gt_points, raw_images)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        pbar.set_postfix({
            'Loss': f"{loss.item():.4f}",
            'Chamfer': f"{components['chamfer']:.4f}",
            'Spectrum': f"{components['spectrum']:.4f}",
            'Color': f"{components['color']:.4f}"
        })
    
    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, 
                        help='Path to data directory with source/ and target/ folders')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=Config.BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=Config.LEARNING_RATE)
    parser.add_argument('--points', type=int, default=Config.NUM_POINTS, help='Number of stipple dots')
    parser.add_argument('--image_size', type=int, default=Config.IMAGE_SIZE)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--subset', type=int, default=None, help='Use only N images for quick testing')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Setup model (ResNet50 + AdaIN Grid Decoder)
    print(f"Initializing Model with {args.points} points...")
    model = create_model(num_points=args.points, grid_size=Config.GRID_SIZE).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=args.lr, amsgrad=True)
    
    # Paper loss: L = Lc + 0.1 * Ls + Lcolor
    criterion = StipplingLoss(
        weight_chamfer=1.0,
        weight_spectrum=0.1,
        weight_color=1.0,
        image_size=args.image_size
    ).to(device)
    
    # Dataset
    dataset = StipplingDataset(args.data, image_size=args.image_size, num_points=args.points)
    
    if args.subset is not None and args.subset < len(dataset):
        from torch.utils.data import Subset
        indices = list(range(args.subset))
        dataset = Subset(dataset, indices)
        print(f"Using subset of {args.subset} images")
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Dataset size: {len(dataset)} images")
    print("Starting Training...")
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        avg_loss = train_epoch(model, dataloader, criterion, optimizer, device)
        print(f"Epoch Loss: {avg_loss:.5f}")
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f"{args.save_dir}/stipple_net_{epoch+1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")


if __name__ == '__main__':
    main()
