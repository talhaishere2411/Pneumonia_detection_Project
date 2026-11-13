"""
Model definition and training utilities (Dev A's responsibility).
Placeholder for Phase 1 implementation.
"""
import tensorflow as tf


def build_vgg16_model(input_shape: tuple, num_classes: int = 2) -> tf.keras.Model:
    """
    Build VGG16 model with transfer learning.
    
    Args:
        input_shape: Input image shape (height, width, channels)
        num_classes: Number of output classes
        
    Returns:
        Compiled Keras model
    """
    # TODO: Implement in Phase 1
    raise NotImplementedError("Model building will be implemented in Phase 1")


def train_model(model: tf.keras.Model, 
                train_ds: tf.data.Dataset,
                val_ds: tf.data.Dataset,
                epochs: int = 10) -> tf.keras.callbacks.History:
    """
    Train the model with given datasets.
    
    Args:
        model: Compiled Keras model
        train_ds: Training dataset
        val_ds: Validation dataset
        epochs: Number of training epochs
        
    Returns:
        Training history
    """
    # TODO: Implement in Phase 1
    raise NotImplementedError("Model training will be implemented in Phase 1")
