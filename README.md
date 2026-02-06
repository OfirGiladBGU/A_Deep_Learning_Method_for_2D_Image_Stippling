# A Deep Learning Method for 2D Image Stippling

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Implementation of the paper "A Deep Learning Method for 2D Image Stippling" (Li et al., ICIG 2021).

## Overview

This project implements a neural network that learns to create artistic stippled representations of images. The network directly predicts point coordinates (x, y) and colors (r, g, b) for each stipple dot, trained with supervised learning using source/target image pairs.

## Architecture

The implementation follows the exact paper architecture:

- **Encoder**: ResNet50 (pretrained, frozen) → FC layer to 1024-dim style vector
- **Style MLP**: 3-layer MLP for style mapping
- **Grid Decoder**: AdaIN-based upsampling from 4×4 → 32×32 spatial grid
- **Generator Heads**: Conv2d heads for density, location (dx, dy), and color (r, g, b)
- **Output**: 1024 points (32×32 grid, 1 point per cell) with coordinates and colors

## Features

- **Point-based Output**: Directly predicts (x, y, r, g, b) for each stipple point
- **AdaIN Grid Decoder**: Adaptive Instance Normalization for style conditioning
- **Supervised Training**: Uses source/target image pairs for training
- **Multiple Loss Functions**: 
  - Chamfer Distance (point matching)
  - Spectrum Loss (blue noise quality)
  - Color Loss (point color matching)
- **Easy Visualization**: Generate side-by-side comparisons with ground truth

## Data Path

Training data is located at:
```
/groups/asharf_group/ofirgila/ControlNet/training/data_grads_v3_2048/
├── source/   # Input grayscale images
└── target/   # Binary stipple images (black dots on white)
```

## Installation

### Prerequisites
- Python 3.8 or higher
- PyTorch 2.0 or higher
- CUDA-capable GPU (recommended for training)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/OfirGiladBGU/A-_Deep_Learning_Method_for_2D_Image_Stippling.git
cd A-_Deep_Learning_Method_for_2D_Image_Stippling
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Demo

Run the example demo to see the stippling network in action:

```bash
python examples/demo.py
```

This will create sample images and generate stippled versions using different rendering methods. Output will be saved in the `outputs/` directory.

### Training

To train the model:

```bash
python train.py --data /path/to/data --epochs 15 --batch_size 16 --points 1024
```

**Arguments:**
- `--data`: Path to data directory with `source/` and `target/` folders
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size for training (default: 16)
- `--lr`: Learning rate (default: 1e-4)
- `--points`: Number of stipple points (default: 1024, matches 32x32 grid)
- `--subset`: Use subset of data for quick testing (optional)

**Example (quick test):**
```bash
python train.py --data /groups/asharf_group/ofirgila/ControlNet/training/data_grads_v3_2048 --epochs 15 --batch_size 16 --points 1024 --subset 1000
```

**Example (full training):**
```bash
python train.py --data /groups/asharf_group/ofirgila/ControlNet/training/data_grads_v3_2048 --epochs 50 --batch_size 16 --points 1024
```

### Visualization

Generate visualizations comparing predictions with ground truth:

```bash
python visualize.py --checkpoint checkpoints/stipple_net_15.pth --data /groups/asharf_group/ofirgila/ControlNet/training/data_grads_v3_2048 --output results --num_samples 5 --points 1024
```

## Project Structure

```
A_Deep_Learning_Method_for_2D_Image_Stippling/
├── src/
│   ├── models/
│   │   ├── stippling_network.py  # ResNet50 + AdaIN Grid Decoder
│   │   └── losses.py             # Chamfer, Spectrum, Color losses
│   └── utils/
│       ├── image_processing.py   # Image I/O and preprocessing
│       └── rendering.py          # Stipple rendering utilities
├── examples/
│   └── demo.py                   # Example demonstration
├── checkpoints/                  # Model checkpoints
├── results/                      # Visualization outputs
├── train.py                      # Training script
├── visualize.py                  # Visualization script
├── inference.py                  # Single image inference
├── config.py                     # Configuration settings
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Method Details

### Neural Network Architecture

- **Encoder**: ResNet50 (pretrained, frozen) → 2048 features → FC → 1024-dim style vector
- **Style MLP**: 3 fully-connected layers with ReLU, all 1024 dimensions
- **Grid Decoder**: 4×4 → 8×8 → 16×16 → 32×32 using AdaIN upsampling blocks
- **Generator Heads**:
  - Density: Conv2d(32→64→1) with Sigmoid
  - Location: Conv2d(32→64→2) with Sigmoid (dx, dy offsets)
  - Color: Conv2d(32→64→3) with Sigmoid (r, g, b)
- **Output**: [B, 1024, 5] tensor with (x, y, r, g, b) for each point

### Loss Functions

The network is trained with a combined loss (from `src/models/losses.py`):
1. **Chamfer Distance**: Measures point-to-point matching between predicted and target points
2. **Spectrum Loss**: Encourages blue noise distribution (computed via FFT)
3. **Color Loss**: L1 loss on point colors

Default weights: Chamfer=1.0, Spectrum=0.1, Color=1.0

### Point Generation

1. Grid decoder produces 32×32 feature map
2. Location head predicts local offset (0-1) within each cell
3. Global coordinate = (cell_index + offset) / grid_size
4. Each cell produces exactly 1 point (no Top-K selection needed)
5. Total: 32×32 = 1024 points

## Model Performance

The model learns to:
- Predict 1024 stipple points per image
- Match spatial distribution of ground truth points (Chamfer loss)
- Maintain blue noise spectral properties (Spectrum loss)
- Preserve local image colors at each point

**Training Results (15 epochs, 1000 images):**
- Final Loss: ~0.03
- Chamfer Loss: ~0.01
- Spectrum Loss: ~0.005
- Training speed: ~5.8 it/s on RTX 6000

## Dataset

### Training Data Location

```
/groups/asharf_group/ofirgila/ControlNet/training/data_grads_v3_2048/
├── source/   # ~40,000 grayscale input images (512x512 PNG)
└── target/   # Corresponding stipple images (black dots on white)
```

### Dataset Format

For training, prepare a directory with:
- `source/` folder: Input images (grayscale or RGB, PNG format)
- `target/` folder: Binary stipple images (black dots < threshold 10 on white background)
- Matching filenames between source and target
- Images should be square (512x512 recommended)

The target images contain black pixels representing stipple dot locations. The training script automatically:
- Extracts point coordinates from black pixels
- Samples exactly 1024 points (to match 32×32 grid)
- Associates colors from source image at each point location

## Configuration

Settings in `config.py`:

```python
class Config:
    NUM_POINTS = 1024      # Matches 32x32 grid exactly
    GRID_SIZE = 32         # Spatial grid dimension
    IMAGE_SIZE = 224       # ResNet input size
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    
    # Loss Weights
    WEIGHT_CHAMFER = 1.0   # Point matching
    WEIGHT_SPECTRUM = 0.1  # Blue noise quality
    WEIGHT_COLOR = 1.0     # Color matching
```

## Citation

If you use this implementation, please cite the original paper:

```
A Deep Learning Method for 2D Image Stippling
ICIG 2021: Image and Graphics pp 301-312
https://link.springer.com/chapter/10.1007/978-3-030-89029-2_24
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Based on the paper "A Deep Learning Method for 2D Image Stippling" from ICIG 2021
- Inspired by traditional stippling and halftoning techniques
- Built with PyTorch and standard deep learning best practices

## Troubleshooting

### Out of Memory
- Reduce `--batch-size` during training
- Reduce `--image-size` to process smaller images
- Use CPU instead of GPU for inference

### Poor Results
- The model requires training on a dataset to produce good results
- Increase training epochs
- Use larger and more diverse training dataset
- Adjust perceptual loss weight

### Installation Issues
- Ensure Python 3.8+ is installed
- Update pip: `pip install --upgrade pip`
- For CUDA issues, ensure CUDA toolkit matches PyTorch version

## Contact

For questions or issues, please open an issue on GitHub.