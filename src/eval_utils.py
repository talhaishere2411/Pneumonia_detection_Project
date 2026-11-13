"""
Evaluation and metrics utilities (Dev B's responsibility).
Placeholder for Phase 3 implementation.
"""
import tensorflow as tf
import numpy as np
from typing import Dict, Tuple
from sklearn.metrics import classification_report, confusion_matrix


def evaluate_model(model: tf.keras.Model,
                   test_ds: tf.data.Dataset) -> Dict[str, float]:
    """
    Evaluate model performance on test dataset.
    
    Args:
        model: Trained Keras model
        test_ds: Test dataset
        
    Returns:
        Dictionary of evaluation metrics
    """
    # TODO: Implement in Phase 3
    raise NotImplementedError("Model evaluation will be implemented in Phase 3")


def generate_classification_report(y_true: np.ndarray,
                                   y_pred: np.ndarray,
                                   class_names: list) -> str:
    """
    Generate detailed classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        
    Returns:
        Classification report string
    """
    # TODO: Implement in Phase 3
    raise NotImplementedError("Classification report will be implemented in Phase 3")


def plot_confusion_matrix(y_true: np.ndarray,
                         y_pred: np.ndarray,
                         class_names: list) -> None:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
    """
    # TODO: Implement in Phase 3
    raise NotImplementedError("Confusion matrix plotting will be implemented in Phase 3")
