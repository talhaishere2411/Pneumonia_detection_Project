"""
Grad-CAM visualization utilities (Dev B's responsibility).
Placeholder for Phase 2 implementation.
"""
import numpy as np
import tensorflow as tf
from typing import Tuple


def make_gradcam_heatmap(img_array: np.ndarray,
                         model: tf.keras.Model,
                         last_conv_layer_name: str,
                         pred_index: int = None) -> np.ndarray:
    """
    Generate Grad-CAM heatmap for a given image.
    
    Args:
        img_array: Preprocessed image array
        model: Trained Keras model
        last_conv_layer_name: Name of the last convolutional layer
        pred_index: Class index to generate heatmap for (None = predicted class)
        
    Returns:
        Heatmap array
    """
    # TODO: Implement in Phase 2
    raise NotImplementedError("Grad-CAM will be implemented in Phase 2")


def overlay_heatmap(img: np.ndarray, 
                    heatmap: np.ndarray,
                    alpha: float = 0.4) -> np.ndarray:
    """
    Overlay heatmap on original image.
    
    Args:
        img: Original image
        heatmap: Grad-CAM heatmap
        alpha: Transparency factor
        
    Returns:
        Overlayed image
    """
    # TODO: Implement in Phase 2
    raise NotImplementedError("Heatmap overlay will be implemented in Phase 2")
