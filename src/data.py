"""
Data loading and preprocessing utilities (Dev A's responsibility).
Handles dataset generation with ImageDataGenerator and class weight computation.
"""
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input
from typing import Tuple, Dict
from pathlib import Path

from . import config


def get_generators(root=None) -> Tuple[ImageDataGenerator, ImageDataGenerator, ImageDataGenerator]:
    """
    Create train, validation, and test data generators with proper preprocessing.
    
    Uses VGG16's preprocess_input for ImageNet-compatible preprocessing.
    Train set uses data augmentation; validation and test sets do not.
    
    Args:
        root: Root directory containing train/ and test/ subdirectories.
              If None, uses config.find_data_root() to auto-detect.
    
    Returns:
        Tuple of (train_generator, val_generator, test_generator)
        
    Raises:
        FileNotFoundError: If data directory structure is invalid
    """
    # Auto-detect data root if not provided
    if root is None:
        root = config.find_data_root()
    
    root = Path(root)
    train_dir = root / "train"
    test_dir = root / "test"
    
    # Validate directory structure
    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
    
    # Training generator with augmentation
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,  # VGG16 preprocessing (NOT rescale=1/255)
        validation_split=config.VAL_SPLIT,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    
    train_generator = train_datagen.flow_from_directory(
        directory=str(train_dir),
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode="categorical",
        classes=config.CLASS_NAMES,
        subset="training",
        shuffle=True,
        seed=config.SEED
    )
    
    # Validation generator (no augmentation)
    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=config.VAL_SPLIT
    )
    
    val_generator = val_datagen.flow_from_directory(
        directory=str(train_dir),
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode="categorical",
        classes=config.CLASS_NAMES,
        subset="validation",
        shuffle=False,
        seed=config.SEED
    )
    
    # Test generator (no augmentation, official test set)
    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )
    
    test_generator = test_datagen.flow_from_directory(
        directory=str(test_dir),
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode="categorical",
        classes=config.CLASS_NAMES,
        shuffle=False
    )
    
    print(f"✓ Train samples: {train_generator.samples}")
    print(f"✓ Validation samples: {val_generator.samples}")
    print(f"✓ Test samples: {test_generator.samples}")
    print(f"✓ Class indices: {train_generator.class_indices}")
    
    return train_generator, val_generator, test_generator


def get_class_weights(train_gen: ImageDataGenerator) -> Dict[int, float]:
    """
    Compute balanced class weights for handling class imbalance.
    
    Uses sklearn's compute_class_weight with 'balanced' strategy to give
    higher weight to underrepresented classes.
    
    Args:
        train_gen: Training data generator with .classes attribute
        
    Returns:
        Dictionary mapping class index to weight: {0: weight_normal, 1: weight_pneumonia}
    """
    # Compute class weights using sklearn
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.array([0, 1]),
        y=train_gen.classes
    )
    
    # Convert to dictionary format required by Keras
    class_weight_dict = {
        0: float(class_weights[0]),
        1: float(class_weights[1])
    }
    
    print(f"✓ Class weights computed: {class_weight_dict}")
    
    return class_weight_dict
