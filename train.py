"""
Training script for the stippling network
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

from src.models.stippling_network import create_model
from src.models.losses import StipplingLoss


class ImageDataset(Dataset):
    """
    Dataset for training stippling network
    Loads images and converts to grayscale targets
    """
    
    def __init__(self, image_dir, image_size=512, augment=True):
        self.image_dir = image_dir
        self.image_size = image_size
        self.augment = augment
        
        # Get all image files
        self.image_files = [f for f in os.listdir(image_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        # Transforms
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.target_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])
        
        if augment:
            self.augment_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        
        # Apply augmentation
        if self.augment:
            image = self.augment_transform(image)
        
        # Transform
        input_tensor = self.transform(image)
        target_tensor = self.target_transform(image)
        
        return input_tensor, target_tensor, input_tensor


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    loss_components = {'reconstruction': 0.0, 'perceptual': 0.0, 'total': 0.0}
    
    progress_bar = tqdm(dataloader, desc='Training')
    for batch_idx, (inputs, targets_gray, targets_rgb) in enumerate(progress_bar):
        inputs = inputs.to(device)
        targets_gray = targets_gray.to(device)
        targets_rgb = targets_rgb.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # Calculate loss
        loss, losses = criterion(outputs, targets_gray, targets_rgb)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item()
        for key, value in losses.items():
            if key in loss_components:
                loss_components[key] += value
        
        # Update progress bar
        progress_bar.set_postfix({'loss': loss.item()})
    
    # Average losses
    num_batches = len(dataloader)
    total_loss /= num_batches
    for key in loss_components:
        loss_components[key] /= num_batches
    
    return total_loss, loss_components


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for inputs, targets_gray, targets_rgb in dataloader:
            inputs = inputs.to(device)
            targets_gray = targets_gray.to(device)
            targets_rgb = targets_rgb.to(device)
            
            outputs = model(inputs)
            loss, _ = criterion(outputs, targets_gray, targets_rgb)
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description='Train stippling network')
    parser.add_argument('--data', type=str, required=True, help='Path to training images directory')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--image-size', type=int, default=512, help='Image size')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Create dataset and dataloader
    print(f"Loading dataset from {args.data}...")
    dataset = ImageDataset(args.data, image_size=args.image_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    print(f"Found {len(dataset)} images")
    
    # Create model
    print("Creating model...")
    model = create_model()
    model = model.to(args.device)
    
    # Create loss and optimizer
    criterion = StipplingLoss(perceptual_weight=0.1, use_perceptual=True)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    
    start_epoch = 0
    
    # Resume from checkpoint if specified
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    
    # Training loop
    print(f"Starting training on {args.device}...")
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_loss, loss_components = train_epoch(model, dataloader, criterion, optimizer, args.device)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"  Reconstruction: {loss_components['reconstruction']:.4f}")
        if 'perceptual' in loss_components:
            print(f"  Perceptual: {loss_components['perceptual']:.4f}")
        
        # Update learning rate
        scheduler.step()
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'model_epoch_{epoch + 1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
    
    # Save final model
    final_path = os.path.join(args.checkpoint_dir, 'model_final.pth')
    torch.save({
        'epoch': args.epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, final_path)
    print(f"\nTraining complete! Final model saved: {final_path}")


if __name__ == '__main__':
    main()
