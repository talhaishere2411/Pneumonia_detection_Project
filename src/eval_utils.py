"""
Evaluation and metrics utilities (Dev B's responsibility).
Provides comprehensive model evaluation, metrics computation, and analysis tools.
"""
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, Any
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay
)

from . import config


def get_model_prediction(model: tf.keras.Model, 
                        processed_img: np.ndarray) -> Dict[str, Any]:
    """
    Run inference on a preprocessed image array and return prediction details.
    
    Handles logits-based models by applying softmax to compute probabilities
    and confidence scores.
    
    Args:
        model: Trained Keras model (outputs logits, NOT softmax)
        processed_img: Preprocessed image batch with shape (1, H, W, 3)
        
    Returns:
        Dictionary containing:
        - class_idx: Predicted class index (int)
        - class_name: Predicted class name (str)
        - confidence: Confidence score of the prediction (float in [0, 1])
        - probabilities: Softmax probabilities for all classes (list)
        - raw_predictions: Raw model output logits (list)
        
    Example:
        >>> result = get_model_prediction(model, processed_img)
        >>> print(f"{result['class_name']}: {result['confidence']:.2%}")
        PNEUMONIA: 94.23%
    """
    # Run model inference
    predictions = model.predict(processed_img, verbose=0)
    
    # Get predicted class
    predicted_class_idx = np.argmax(predictions[0])
    
    # Compute softmax probabilities from logits
    probabilities = tf.nn.softmax(predictions[0]).numpy()
    confidence = float(np.max(probabilities))
    
    # Map to class name
    predicted_class_name = config.CLASS_NAMES[predicted_class_idx]
    
    return {
        "class_idx": int(predicted_class_idx),
        "class_name": predicted_class_name,
        "confidence": confidence,
        "probabilities": probabilities.tolist(),
        "raw_predictions": predictions[0].tolist()
    }


def predict_probs(model: tf.keras.Model, 
                 generator: tf.keras.preprocessing.image.ImageDataGenerator) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate predictions for an entire dataset using a data generator.
    
    IMPORTANT: Generator MUST have shuffle=False to maintain alignment
    between predictions and true labels.
    
    Args:
        model: Trained Keras model (outputs logits)
        generator: Keras ImageDataGenerator with shuffle=False
        
    Returns:
        Tuple of:
        - y_true: True class labels, shape (N,)
        - probs: Predicted probabilities, shape (N, num_classes)
        
    Example:
        >>> y_true, probs = predict_probs(model, test_gen)
        >>> y_pred = probs.argmax(axis=1)
        >>> accuracy = (y_true == y_pred).mean()
    """
    # Reset generator to ensure we start from the beginning
    generator.reset()
    
    # Get predictions (logits)
    logits = model.predict(generator, verbose=0)
    
    # Convert logits to probabilities
    probs = tf.nn.softmax(logits, axis=1).numpy()
    
    # Get true labels from generator
    y_true = generator.classes
    
    return y_true, probs


def full_report(y_true: np.ndarray,
               probs: np.ndarray,
               tag: str,
               out_dir: str = "results") -> Dict[str, Any]:
    """
    Generate and save comprehensive evaluation report.
    
    Creates:
    - JSON file with classification metrics (precision, recall, F1, ROC-AUC)
    - Confusion matrix PNG
    - ROC curve PNG
    - Console output with formatted metrics
    
    Args:
        y_true: True class labels, shape (N,)
        probs: Predicted probabilities, shape (N, num_classes)
        tag: Identifier for output files (e.g., "vgg16_test", "baseline_val")
        out_dir: Directory to save output files (default: "results")
        
    Returns:
        Dictionary containing all computed metrics
        
    Example:
        >>> metrics = full_report(y_true, probs, "vgg16_test")
        >>> print(f"ROC-AUC: {metrics['roc_auc']:.3f}")
    """
    # Create output directory if it doesn't exist
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Get predicted classes
    y_pred = probs.argmax(axis=1)
    
    # Generate classification report
    report = classification_report(
        y_true, 
        y_pred, 
        target_names=config.CLASS_NAMES,
        output_dict=True
    )
    
    # Add ROC-AUC score (using probability of positive class)
    report["roc_auc"] = float(roc_auc_score(y_true, probs[:, 1]))
    
    # Save metrics to JSON
    metrics_path = out_path / f"{tag}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[OK] Saved metrics to: {metrics_path}")
    
    # Print formatted classification report
    print("\n" + "="*60)
    print(f"Classification Report: {tag}")
    print("="*60)
    print(classification_report(y_true, y_pred, target_names=config.CLASS_NAMES))
    print(f"ROC-AUC Score: {report['roc_auc']:.4f}")
    print("="*60 + "\n")
    
    # Plot and save confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=config.CLASS_NAMES
    )
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix: {tag}")
    
    cm_path = out_path / f"{tag}_confusion.png"
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    print(f"[OK] Saved confusion matrix to: {cm_path}")
    plt.show()
    plt.close()
    
    # Plot and save ROC curve
    fpr, tpr, _ = roc_curve(y_true, probs[:, 1])
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f"AUC = {report['roc_auc']:.3f}")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(f"ROC Curve: {tag}", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    
    roc_path = out_path / f"{tag}_roc.png"
    plt.savefig(roc_path, dpi=150, bbox_inches="tight")
    print(f"[OK] Saved ROC curve to: {roc_path}")
    plt.show()
    plt.close()
    
    return report


def plot_history(history: tf.keras.callbacks.History,
                tag: str,
                out_dir: str = "results") -> None:
    """
    Plot and save training history curves.
    
    Creates a figure with loss and accuracy curves for both training
    and validation sets.
    
    Args:
        history: Keras History object from model.fit()
        tag: Identifier for output file (e.g., "stage1", "stage2")
        out_dir: Directory to save output file (default: "results")
        
    Example:
        >>> history = model.fit(train_gen, validation_data=val_gen, epochs=10)
        >>> plot_history(history, "vgg16_stage1")
    """
    # Create output directory if it doesn't exist
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    axes[0].plot(history.history["loss"], label="Training", linewidth=2)
    axes[0].plot(history.history["val_loss"], label="Validation", linewidth=2)
    axes[0].set_title("Model Loss", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=10)
    axes[0].set_ylabel("Loss", fontsize=10)
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Plot accuracy
    axes[1].plot(history.history["accuracy"], label="Training", linewidth=2)
    axes[1].plot(history.history["val_accuracy"], label="Validation", linewidth=2)
    axes[1].set_title("Model Accuracy", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=10)
    axes[1].set_ylabel("Accuracy", fontsize=10)
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    curves_path = out_path / f"{tag}_curves.png"
    plt.savefig(curves_path, dpi=150, bbox_inches="tight")
    print(f"[OK] Saved training curves to: {curves_path}")
    plt.show()
    plt.close()


def misclassified_indices(y_true: np.ndarray, 
                         probs: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Identify indices of true positives, true negatives, false positives, and false negatives.
    
    Uses PNEUMONIA (class index 1) as the positive class, which is the clinically
    important class for this diagnosis task.
    
    Args:
        y_true: True class labels, shape (N,)
        probs: Predicted probabilities, shape (N, num_classes)
        
    Returns:
        Dictionary with keys:
        - "tp": True Positive indices (correctly predicted PNEUMONIA)
        - "tn": True Negative indices (correctly predicted NORMAL)
        - "fp": False Positive indices (incorrectly predicted PNEUMONIA)
        - "fn": False Negative indices (missed PNEUMONIA cases)
        
    Example:
        >>> indices = misclassified_indices(y_true, probs)
        >>> print(f"Missed pneumonia cases: {len(indices['fn'])}")
        >>> fn_paths = [test_gen.filepaths[i] for i in indices['fn']]
    """
    y_pred = probs.argmax(axis=1)
    
    # PNEUMONIA (index 1) is the positive class
    tp_indices = np.where((y_true == 1) & (y_pred == 1))[0]
    tn_indices = np.where((y_true == 0) & (y_pred == 0))[0]
    fp_indices = np.where((y_true == 0) & (y_pred == 1))[0]
    fn_indices = np.where((y_true == 1) & (y_pred == 0))[0]
    
    return {
        "tp": tp_indices,
        "tn": tn_indices,
        "fp": fp_indices,
        "fn": fn_indices
    }


def print_misclassification_summary(y_true: np.ndarray,
                                   probs: np.ndarray) -> None:
    """
    Print a summary of classification results and error breakdown.
    
    Args:
        y_true: True class labels
        probs: Predicted probabilities
    """
    indices = misclassified_indices(y_true, probs)
    
    total = len(y_true)
    correct = len(indices["tp"]) + len(indices["tn"])
    incorrect = len(indices["fp"]) + len(indices["fn"])
    
    print("\n" + "="*60)
    print("Classification Summary")
    print("="*60)
    print(f"Total samples: {total}")
    print(f"Correct predictions: {correct} ({100*correct/total:.1f}%)")
    print(f"Incorrect predictions: {incorrect} ({100*incorrect/total:.1f}%)")
    print("\nBreakdown:")
    print(f"  True Positives (TP):  {len(indices['tp']):4d} - Correctly identified PNEUMONIA")
    print(f"  True Negatives (TN):  {len(indices['tn']):4d} - Correctly identified NORMAL")
    print(f"  False Positives (FP): {len(indices['fp']):4d} - NORMAL incorrectly as PNEUMONIA")
    print(f"  False Negatives (FN): {len(indices['fn']):4d} - PNEUMONIA missed (critical!)")
    print("="*60 + "\n")
