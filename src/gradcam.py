"""
Grad-CAM visualization utilities (Dev B's responsibility).
Implements Gradient-weighted Class Activation Mapping for model explainability.
"""
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import preprocess_input
from typing import Tuple, Optional, List
from pathlib import Path

from . import config


def load_and_preprocess_image(image_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a chest X-ray image and preprocess it for VGG16 evaluation.
    
    This function handles the complete image loading pipeline:
    1. Load image from disk and resize to target size
    2. Convert to array format for display (RGB uint8)
    3. Preprocess for VGG16 model input (ImageNet normalization)
    
    Args:
        image_path: Path to the chest X-ray image file
        
    Returns:
        Tuple of:
        - img: PIL Image object (for compatibility)
        - img_array: Original image as numpy array with batch dimension [1, H, W, 3]
        - processed_img: Preprocessed image ready for model input [1, H, W, 3]
        
    Raises:
        FileNotFoundError: If image_path does not exist
        ValueError: If image cannot be loaded or is invalid
    """
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    try:
        # Load image and resize to model input size
        img = tf.keras.utils.load_img(
            image_path,
            target_size=config.IMG_SIZE,
            color_mode="rgb"
        )
        
        # Convert to numpy array with batch dimension
        img_array = tf.keras.utils.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 224, 224, 3)
        
        # Preprocess for VGG16 (ImageNet normalization, NOT rescale 1/255)
        processed_img = preprocess_input(img_array.copy())
        
        return img, img_array, processed_img
        
    except Exception as e:
        raise ValueError(f"Failed to load or preprocess image {image_path}: {e}")


def make_gradcam_heatmap(img_array: np.ndarray,
                         model: tf.keras.Model,
                         last_conv_layer_name: str = None,
                         pred_index: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Grad-CAM heatmap for a given input array and model.
    
    Grad-CAM uses the gradients of the target class flowing into the final
    convolutional layer to produce a coarse localization map highlighting
    important regions in the image for prediction.
    
    The model MUST output logits (not softmax probabilities) for correct
    gradient computation. Softmax probabilities are computed separately
    for display purposes.
    
    Args:
        img_array: Preprocessed image array with shape (1, H, W, 3)
        model: Trained Keras model with logits output (NO softmax activation)
        last_conv_layer_name: Name of the last convolutional layer
                             (default: uses config.LAST_CONV_LAYER)
        pred_index: Class index to generate heatmap for. If None, uses
                   the predicted class (argmax of model output)
        
    Returns:
        Tuple of:
        - heatmap: Normalized heatmap array in range [0, 1], shape (H, W)
        - probabilities: Softmax probabilities for all classes, shape (num_classes,)
        
    Raises:
        ValueError: If last_conv_layer_name is not found in the model
    """
    if last_conv_layer_name is None:
        last_conv_layer_name = config.LAST_CONV_LAYER
    
    # Verify the layer exists
    try:
        conv_layer = model.get_layer(last_conv_layer_name)
    except ValueError:
        raise ValueError(
            f"Layer '{last_conv_layer_name}' not found in model. "
            f"Available layers: {[layer.name for layer in model.layers]}"
        )
    
    # Create gradient model that outputs both conv features and predictions
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[conv_layer.output, model.output]
    )
    
    # Compute gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array, training=False)
        
        # Use predicted class if not specified
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        
        # Get the score for the target class (logits)
        class_channel = predictions[:, pred_index]
    
    # Gradient of the target class score with respect to conv layer output
    grads = tape.gradient(class_channel, conv_outputs)
    
    # Global average pooling of gradients -> importance weights per channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight each channel by its importance and sum
    conv_outputs = conv_outputs[0]  # Remove batch dimension
    heatmap = tf.reduce_sum(
        conv_outputs * pooled_grads,
        axis=-1
    )
    
    # Apply ReLU to focus on features that have positive influence
    heatmap = tf.maximum(heatmap, 0)
    
    # Normalize to [0, 1] range
    heatmap /= tf.reduce_max(heatmap) + tf.keras.backend.epsilon()
    
    # Convert logits to probabilities for display
    probabilities = tf.nn.softmax(predictions[0]).numpy()
    
    return heatmap.numpy(), probabilities


def overlay_heatmap(img: np.ndarray, 
                    heatmap: np.ndarray,
                    alpha: float = 0.4) -> np.ndarray:
    """
    Overlay the Grad-CAM heatmap onto the original image.
    
    Creates a visualization by blending a colorized heatmap (using JET colormap)
    with the original grayscale or RGB image.
    
    Args:
        img: Original image as numpy array (uint8, RGB or grayscale)
             Shape: (H, W) or (H, W, 3)
        heatmap: Grad-CAM heatmap in range [0, 1], shape (h, w) where
                h, w can differ from img dimensions (will be resized)
        alpha: Transparency/blend factor for heatmap (0=invisible, 1=opaque)
               Default 0.4 provides good balance
        
    Returns:
        Overlayed image as numpy array (uint8, RGB), same size as input img
    """
    # Ensure image is uint8 RGB format
    img = np.asarray(img).astype("uint8")
    
    # Convert grayscale to RGB if needed
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    
    # Resize heatmap to match image dimensions
    heatmap_resized = cv2.resize(
        heatmap,
        (img.shape[1], img.shape[0])  # (width, height) for cv2
    )
    
    # Convert heatmap to uint8 and apply JET colormap
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    
    # OpenCV uses BGR, convert to RGB
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Blend original image with heatmap
    overlay = cv2.addWeighted(
        img,
        1 - alpha,
        heatmap_colored,
        alpha,
        0
    )
    
    return overlay


def plot_gradcam_grid(model: tf.keras.Model,
                     paths: List[str],
                     true_labels: Optional[List[int]] = None,
                     ncols: int = 4,
                     save_path: Optional[str] = None) -> None:
    """
    Create a grid visualization showing original images and Grad-CAM overlays.
    
    For each image, displays:
    - Left column: Original image
    - Right column: Grad-CAM overlay with prediction and confidence
    
    Args:
        model: Trained Keras model (must output logits)
        paths: List of image file paths to visualize
        true_labels: Optional list of ground truth labels (class indices)
                    If provided, shown in the title for comparison
        ncols: Number of columns in the grid (default: 4)
               Total columns will be ncols * 2 (original + overlay)
        save_path: Optional path to save the figure
                  If None, only displays the plot
        
    Raises:
        FileNotFoundError: If any image path is invalid
        ValueError: If true_labels length doesn't match paths length
    """
    if true_labels is not None and len(true_labels) != len(paths):
        raise ValueError(
            f"Length mismatch: {len(paths)} paths but {len(true_labels)} labels"
        )
    
    n = len(paths)
    nrows = int(np.ceil(n / ncols))
    
    # Create figure with 2 columns per image (original | overlay)
    fig, axes = plt.subplots(
        nrows, 
        ncols * 2, 
        figsize=(4 * ncols, 2.6 * nrows)
    )
    axes = np.atleast_2d(axes)
    
    for i, path in enumerate(paths):
        r, c = divmod(i, ncols)
        
        # Load and process image
        _, img_array, processed = load_and_preprocess_image(path)
        rgb = img_array[0].astype("uint8")  # Remove batch dimension
        
        # Generate Grad-CAM
        heatmap, probs = make_gradcam_heatmap(processed, model)
        pred_idx = int(probs.argmax())
        
        # Plot original and overlay
        ax_orig = axes[r, 2 * c]
        ax_heat = axes[r, 2 * c + 1]
        
        ax_orig.imshow(rgb)
        ax_orig.axis("off")
        
        ax_heat.imshow(overlay_heatmap(rgb, heatmap, alpha=0.4))
        ax_heat.axis("off")
        
        # Create title with prediction info
        title = f"pred: {config.CLASS_NAMES[pred_idx]} ({probs[pred_idx]:.2f})"
        if true_labels is not None:
            title = f"true: {config.CLASS_NAMES[true_labels[i]]}\n" + title
        
        ax_heat.set_title(title, fontsize=8)
    
    # Hide extra subplots if grid not fully filled
    for i in range(n, nrows * ncols):
        r, c = divmod(i, ncols)
        axes[r, 2 * c].axis("off")
        axes[r, 2 * c + 1].axis("off")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[OK] Saved Grad-CAM grid to: {save_path}")
    
    plt.show()
