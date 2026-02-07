class Config:
    # Model Architecture
    NUM_POINTS = 2500  # Paper setting
    GRID_SIZE = 32     # Grid dimension
    
    # VGG Input
    IMAGE_SIZE = 256   # Paper setting
    
    # Training
    BATCH_SIZE = 8
    LEARNING_RATE = 5e-4
    
    # Loss Weights (Paper defaults)
    WEIGHT_CHAMFER = 1.0
    WEIGHT_SPECTRUM = 0.1
    WEIGHT_COLOR = 1.0
