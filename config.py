"""
Configuration file for stippling network
"""

import os


class Config:
    """Base configuration"""
    
    # Model architecture
    MODEL_INPUT_CHANNELS = 3
    MODEL_BASE_CHANNELS = 64
    
    # Training
    BATCH_SIZE = 4
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4
    IMAGE_SIZE = 512
    
    # Loss weights
    RECONSTRUCTION_WEIGHT = 1.0
    PERCEPTUAL_WEIGHT = 0.1
    GRADIENT_WEIGHT = 0.5
    
    # Data
    DATA_DIR = 'data/training_images'
    CHECKPOINT_DIR = 'checkpoints'
    OUTPUT_DIR = 'outputs'
    
    # Training options
    NUM_WORKERS = 4
    PIN_MEMORY = True
    USE_AUGMENTATION = True
    
    # Optimizer
    OPTIMIZER = 'adam'
    WEIGHT_DECAY = 1e-5
    
    # Learning rate scheduler
    LR_SCHEDULER = 'step'
    LR_STEP_SIZE = 30
    LR_GAMMA = 0.5
    
    # Checkpointing
    SAVE_EVERY = 10  # Save checkpoint every N epochs
    
    # Inference
    INFERENCE_SIZE = 512
    INFERENCE_METHOD = 'adaptive'  # 'adaptive', 'threshold', or 'gaussian'
    
    # Rendering parameters
    DOT_DENSITY_SCALE = 1.0
    NUM_DOTS_THRESHOLD = 5000
    DENSITY_THRESHOLD = 0.3
    BACKGROUND_COLOR = 255
    DOT_COLOR = 0
    
    @staticmethod
    def create_directories():
        """Create necessary directories"""
        os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        os.makedirs(Config.DATA_DIR, exist_ok=True)


class TrainingConfig(Config):
    """Configuration for training"""
    pass


class InferenceConfig(Config):
    """Configuration for inference"""
    BATCH_SIZE = 1
