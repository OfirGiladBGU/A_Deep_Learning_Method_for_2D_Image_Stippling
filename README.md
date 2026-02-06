# A Deep Learning Method for 2D Image Stippling

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Implementation of a deep learning method for converting images into stippled art - a technique that represents images using dots.

## Overview

This project implements a neural network that learns to create artistic stippled representations of images. Stippling is an artistic technique where images are represented using dots of varying sizes and densities to create tonal variations.

The implementation uses a U-Net style architecture with attention mechanisms to generate density maps that are then rendered into stippled images.

## Features

- **Deep Neural Network**: U-Net architecture with encoder-decoder design
- **Attention Mechanisms**: Focus on important image features
- **Multiple Rendering Methods**: 
  - Adaptive dot placement based on density
  - Threshold-based dot extraction
  - Gaussian-smoothed dots for artistic effect
- **Flexible Training**: Configurable loss functions (reconstruction + perceptual)
- **Easy Inference**: Simple command-line interface for generating stippled images

## Architecture

The stippling network consists of:
1. **Encoder**: Extracts features from input images (4 downsampling blocks)
2. **Bottleneck**: Processes features with attention mechanism
3. **Decoder**: Generates density map (4 upsampling blocks with skip connections)
4. **Output**: Single-channel density map indicating dot placement probability

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

To train the model on your own dataset:

```bash
python train.py --data path/to/images --epochs 100 --batch-size 4 --lr 1e-4
```

**Arguments:**
- `--data`: Path to directory containing training images
- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Batch size for training (default: 4)
- `--lr`: Learning rate (default: 1e-4)
- `--image-size`: Size to resize images (default: 512)
- `--checkpoint-dir`: Directory to save model checkpoints (default: checkpoints)
- `--device`: Device to use: 'cuda' or 'cpu' (default: auto-detect)
- `--resume`: Path to checkpoint to resume training from

**Example:**
```bash
python train.py --data ./data/training_images --epochs 50 --batch-size 8 --device cuda
```

### Inference

Generate stippled images from trained model:

```bash
python inference.py --input path/to/image.jpg --output path/to/output.png --model path/to/checkpoint.pth
```

**Arguments:**
- `--input` / `-i`: Input image path
- `--output` / `-o`: Output image path
- `--model` / `-m`: Path to trained model checkpoint (optional)
- `--device` / `-d`: Device to use: 'cuda' or 'cpu'
- `--method`: Rendering method: 'adaptive', 'threshold', or 'gaussian' (default: adaptive)

**Example:**
```bash
python inference.py -i photo.jpg -o stippled.png -m checkpoints/model_final.pth --method gaussian
```

## Project Structure

```
A-_Deep_Learning_Method_for_2D_Image_Stippling/
├── src/
│   ├── models/
│   │   ├── stippling_network.py  # Main network architecture
│   │   └── losses.py             # Loss functions
│   └── utils/
│       ├── image_processing.py   # Image I/O and preprocessing
│       └── rendering.py          # Stipple rendering utilities
├── examples/
│   └── demo.py                   # Example demonstration
├── data/
│   └── sample_images/            # Sample input images
├── outputs/                      # Generated outputs
├── checkpoints/                  # Model checkpoints
├── train.py                      # Training script
├── inference.py                  # Inference script
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Method Details

### Neural Network Architecture

- **Encoder**: 4 blocks with [64, 128, 256, 512] channels
- **Bottleneck**: 1024 channels with channel attention
- **Decoder**: 4 blocks with skip connections from encoder
- **Output**: Single-channel density map with sigmoid activation

### Loss Functions

The network is trained with a combined loss:
1. **Reconstruction Loss (L1)**: Ensures density map matches target intensity
2. **Perceptual Loss (VGG)**: Maintains perceptual similarity with input
3. **Gradient Loss** (optional): Preserves edges and details

### Rendering Methods

1. **Adaptive**: Grid-based sampling with density-dependent dot placement
2. **Threshold**: Extract dots above density threshold using NMS
3. **Gaussian**: Smooth dots with Gaussian kernels for artistic effect

## Model Performance

The model learns to:
- Preserve important image features and edges
- Adapt dot density to local image intensity
- Create visually pleasing stippled representations
- Generalize across different image types

**Note**: Results improve significantly with training on a diverse dataset of images.

## Dataset Preparation

For training, prepare a directory of images:
- Supported formats: PNG, JPG, JPEG, BMP
- Recommended: 100+ diverse images
- Images will be automatically resized to specified size
- Data augmentation is applied during training

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