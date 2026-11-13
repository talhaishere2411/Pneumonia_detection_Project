"""
Shared configuration for the pneumonia detection project.
This file defines all constants and paths used across Dev A and Dev B's work.
"""
from pathlib import Path

# Image configuration
IMG_SIZE = (224, 224)
INPUT_SHAPE = (224, 224, 3)

# Training configuration
BATCH_SIZE = 16
SEED = 42
VAL_SPLIT = 0.15  # Carved out of train/, test/ stays untouched

# Class configuration
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]  # index 0 = NORMAL, index 1 = PNEUMONIA

# Model configuration
LAST_CONV_LAYER = "block5_conv3"  # For VGG16
MODEL_FILE = "vgg16_pneumonia.keras"


def find_data_root() -> Path:
    """
    Auto-detect the data root directory across different environments.
    
    Checks in order:
    1. /kaggle/input (Kaggle notebooks)
    2. /content (Google Colab)
    3. data/ (local development)
    
    Returns:
        Path: The root directory containing train/ and test/ subdirectories
        
    Raises:
        FileNotFoundError: If no valid data directory is found
    """
    candidate_paths = [
        Path("/kaggle/input"),
        Path("/content"),
        Path("data")
    ]
    
    for base_path in candidate_paths:
        if not base_path.exists():
            continue
            
        # Search for train directories containing NORMAL subdirectory
        for train_dir in base_path.rglob("train"):
            if (train_dir / "NORMAL").exists():
                # Verify that parent also contains test/
                data_root = train_dir.parent
                if (data_root / "test").exists():
                    return data_root
    
    raise FileNotFoundError(
        "Could not find valid data directory. Expected structure:\n"
        "  <data_root>/\n"
        "    train/\n"
        "      NORMAL/\n"
        "      PNEUMONIA/\n"
        "    test/\n"
        "      NORMAL/\n"
        "      PNEUMONIA/"
    )


# Optionally set DATA_ROOT at import time (can fail gracefully in some contexts)
try:
    DATA_ROOT = find_data_root()
except FileNotFoundError:
    DATA_ROOT = None  # Will need to be set manually or data downloaded
