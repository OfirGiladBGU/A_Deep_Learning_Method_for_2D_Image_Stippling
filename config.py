class Config:
    # Model Architecture
    NUM_POINTS = 1024  # Matched to 32x32 grid size
    GRID_SIZE = 32     # The paper's specified grid dimension
    
    # VGG Input
    IMAGE_SIZE = 224   # Standard for VGG
    
    # Training
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    
    # Loss Weights (Paper defaults)
    WEIGHT_CHAMFER = 1.0
    WEIGHT_SPECTRUM = 0.1
    WEIGHT_COLOR = 1.0
