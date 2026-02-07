import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T

from config import Config
from src.models.stippling_network import create_model

def generate_stipple(model, image_path, output_path, device='cpu'):
    # Preprocess
    img = Image.open(image_path).convert('RGB')
    transform = T.Compose([
        T.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        T.ToTensor()
    ])
    input_tensor = transform(img).unsqueeze(0).to(device)

    # Forward Pass
    model.eval()
    with torch.no_grad():
        # Output is [1, N, 5]
        point_data = model(input_tensor)
    
    # Extract data
    points = point_data[0].cpu().numpy()
    coords = points[:, :2]  # x, y
    colors = points[:, 2:]  # r, g, b

    # --- Render ---
    # We use Matplotlib to plot the scatter points
    plt.figure(figsize=(10, 10), facecolor='white')
    
    # Note: Matplotlib coordinates have (0,0) at bottom-left, images at top-left
    # We invert Y to match image coordinates
    plt.scatter(coords[:, 0] * Config.IMAGE_SIZE,
                (1 - coords[:, 1]) * Config.IMAGE_SIZE,
                c=colors, 
                s=10) # s is dot size
    
    plt.axis('off')
    plt.xlim(0, Config.IMAGE_SIZE)
    plt.ylim(0, Config.IMAGE_SIZE)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    # Load your model and run
    pass
