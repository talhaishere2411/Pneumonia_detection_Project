"""
Kaggle Testing Template - Quick Start
Copy this entire script into a Kaggle notebook cell
"""

# ============================================================
# CONFIGURATION - UPDATE THESE!
# ============================================================
GITHUB_REPO_URL = "https://github.com/YOUR_USERNAME/pneumonia-gradcam.git"
MODEL_DATASET = "pneumonia-vgg16-model"  # Your Kaggle dataset name
MODEL_FILENAME = "vgg16_pneumonia.h5"     # Your model file (.h5 or .keras)

# ============================================================
# SETUP
# ============================================================
import os, sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path

# Clone repository
REPO = "pneumonia-gradcam"
if not os.path.exists(f"/kaggle/working/{REPO}"):
    os.system(f"git clone {GITHUB_REPO_URL} /kaggle/working/{REPO}")
sys.path.insert(0, f"/kaggle/working/{REPO}")

# Import modules
from src import config
from src.data import get_generators
from src.gradcam import load_and_preprocess_image, make_gradcam_heatmap, overlay_heatmap, plot_gradcam_grid
from src.eval_utils import predict_probs, full_report, misclassified_indices

print("✓ Setup complete!")

# ============================================================
# LOAD MODEL AND DATA
# ============================================================
# Auto-detect model path (.h5 or .keras both work!)
MODEL_PATH = f"/kaggle/input/{MODEL_DATASET}/{MODEL_FILENAME}"
print(f"Loading model from: {MODEL_PATH}")

# Check if file exists
if not Path(MODEL_PATH).exists():
    print(f"❌ Model not found at: {MODEL_PATH}")
    print("\nAvailable files:")
    os.system(f"ls -la /kaggle/input/{MODEL_DATASET}/")
    raise FileNotFoundError(f"Model file not found. Update MODEL_FILENAME in config.")

model = tf.keras.models.load_model(MODEL_PATH)
train_gen, val_gen, test_gen = get_generators()

print(f"✓ Model loaded: {model.input_shape} -> {model.output_shape}")
print(f"✓ Test samples: {test_gen.samples}")

# ============================================================
# EVALUATE
# ============================================================
print("\nEvaluating model...")
y_true, probs = predict_probs(model, test_gen)
metrics = full_report(y_true, probs, "test", out_dir="/kaggle/working")

print(f"\n{'='*60}")
print(f"RESULTS: Accuracy={metrics['accuracy']:.2%}, AUC={metrics['roc_auc']:.4f}")
print(f"{'='*60}")

# ============================================================
# GRAD-CAM ANALYSIS
# ============================================================
indices = misclassified_indices(y_true, probs)

print("\nGenerating Grad-CAM visualizations...")
for category in ['tp', 'tn', 'fp', 'fn']:
    if len(indices[category]) > 0:
        idx = indices[category][:8]
        paths = [test_gen.filepaths[i] for i in idx]
        labels = [test_gen.classes[i] for i in idx]
        plot_gradcam_grid(model, paths, labels, ncols=4, 
                         save_path=f"/kaggle/working/gradcam_{category}.png")
        print(f"✓ {category.upper()}: {len(indices[category])} cases")

print(f"\n{'='*60}")
print("DONE! Check /kaggle/working/ for results")
print(f"{'='*60}")
