# Testing Your Trained Model on Kaggle

## 📋 Prerequisites

You should have:
- ✅ Trained model file (`.h5` or `.keras`)
- ✅ Config file (JSON)
- ✅ Metadata files (JSON)
- ✅ This GitHub repository with your code
- ✅ Kaggle account (phone verified for GPU access)

---

## 🚀 Step-by-Step Guide

### **Step 1: Upload Your Model to Kaggle**

#### Option A: Create a Kaggle Dataset (Recommended)

1. **Go to Kaggle**: https://www.kaggle.com
2. **Click "Datasets"** → **"New Dataset"**
3. **Upload your files**:
   ```
   vgg16_pneumonia.h5         (or .keras)
   config.json                (if you have it)
   metadata.json              (training metrics, etc.)
   ```
4. **Set dataset details**:
   - **Title**: `pneumonia-vgg16-model`
   - **Visibility**: Private (or Public if you want to share)
5. **Click "Create"**

Your dataset URL will be something like:
```
https://www.kaggle.com/datasets/YOUR_USERNAME/pneumonia-vgg16-model
```

#### Option B: Upload as Notebook Output (Alternative)

If you trained in a Kaggle notebook:
1. The model in `/kaggle/working/` is automatically saved
2. Go to your notebook → **Output** tab
3. Click **"New Dataset"** → name it `pneumonia-vgg16-model`

---

### **Step 2: Create Testing Notebook on Kaggle**

1. **Go to Kaggle** → **Code** → **New Notebook**
2. **Settings** → Configure:
   - **Accelerator**: GPU T4 x2 (for faster inference)
   - **Internet**: ON (to clone your GitHub repo)
   - **Language**: Python

---

### **Step 3: Add Required Inputs**

In the right panel → **"Add Input"**:

1. **Dataset**: Search `chest-xray-pneumonia` → Add Paul Mooney's dataset
2. **Your Model**: Search `pneumonia-vgg16-model` → Add your uploaded model
3. *(Optional)* If you uploaded your code as a dataset, add that too

---

### **Step 4: Testing Code - Copy & Paste This**

Create a new Kaggle notebook and paste this code:

```python
# ============================================================
# CELL 1: Setup and Clone Repository
# ============================================================
import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path

# Clone your GitHub repository
REPO_NAME = "pneumonia-gradcam"  # Change to your repo name
GITHUB_URL = "https://github.com/YOUR_USERNAME/pneumonia-gradcam.git"  # Change this!

if not os.path.exists(f"/kaggle/working/{REPO_NAME}"):
    !git clone {GITHUB_URL} /kaggle/working/{REPO_NAME}
else:
    !git -C /kaggle/working/{REPO_NAME} pull

# Add to Python path
sys.path.insert(0, f"/kaggle/working/{REPO_NAME}")

# Enable autoreload for development
%load_ext autoreload
%autoreload 2

print("✓ Repository cloned and ready!")

# ============================================================
# CELL 2: Import Your Modules
# ============================================================
from src import config
from src.data import get_generators
from src.gradcam import (
    load_and_preprocess_image, 
    make_gradcam_heatmap, 
    overlay_heatmap,
    plot_gradcam_grid
)
from src.eval_utils import (
    predict_probs, 
    full_report, 
    misclassified_indices,
    print_misclassification_summary
)

print("✓ Modules imported successfully!")

# ============================================================
# CELL 3: Load Your Trained Model
# ============================================================
# Find your model file (adjust path if needed)
model_path = "/kaggle/input/pneumonia-vgg16-model/vgg16_pneumonia.h5"

# Try .keras format if .h5 doesn't exist
if not Path(model_path).exists():
    model_path = "/kaggle/input/pneumonia-vgg16-model/vgg16_pneumonia.keras"

print(f"Loading model from: {model_path}")
model = tf.keras.models.load_model(model_path)

print("\n✓ Model loaded successfully!")
print(f"Input shape: {model.input_shape}")
print(f"Output shape: {model.output_shape}")

# Verify the model architecture
model.summary()

# ============================================================
# CELL 4: Load Test Data
# ============================================================
print("Loading data generators...")
train_gen, val_gen, test_gen = get_generators()

print(f"\n✓ Data loaded!")
print(f"Test samples: {test_gen.samples}")
print(f"Classes: {test_gen.class_indices}")

# ============================================================
# CELL 5: Evaluate Model on Test Set
# ============================================================
print("Evaluating model on test set...")
print("This may take a few minutes...\n")

# Get predictions
y_true, probs = predict_probs(model, test_gen)

# Generate comprehensive report
metrics = full_report(y_true, probs, tag="kaggle_test", out_dir="/kaggle/working")

print("\n" + "="*60)
print("EVALUATION COMPLETE!")
print("="*60)
print(f"Accuracy: {metrics['accuracy']:.2%}")
print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
print(f"\nNORMAL     - Precision: {metrics['NORMAL']['precision']:.2%}, Recall: {metrics['NORMAL']['recall']:.2%}")
print(f"PNEUMONIA  - Precision: {metrics['PNEUMONIA']['precision']:.2%}, Recall: {metrics['PNEUMONIA']['recall']:.2%}")

# ============================================================
# CELL 6: Analyze Misclassifications
# ============================================================
print("Analyzing misclassifications...\n")

indices = misclassified_indices(y_true, probs)
print_misclassification_summary(y_true, probs)

# ============================================================
# CELL 7: Grad-CAM Visualization - True Positives
# ============================================================
print("\n" + "="*60)
print("Generating Grad-CAM for TRUE POSITIVES")
print("="*60)

tp_indices = indices['tp'][:8]  # First 8 TPs
tp_paths = [test_gen.filepaths[i] for i in tp_indices]
tp_labels = [test_gen.classes[i] for i in tp_indices]

plot_gradcam_grid(
    model, 
    tp_paths, 
    true_labels=tp_labels,
    ncols=4,
    save_path="/kaggle/working/gradcam_tp.png"
)

# ============================================================
# CELL 8: Grad-CAM Visualization - True Negatives
# ============================================================
print("\n" + "="*60)
print("Generating Grad-CAM for TRUE NEGATIVES")
print("="*60)

tn_indices = indices['tn'][:8]
tn_paths = [test_gen.filepaths[i] for i in tn_indices]
tn_labels = [test_gen.classes[i] for i in tn_indices]

plot_gradcam_grid(
    model, 
    tn_paths, 
    true_labels=tn_labels,
    ncols=4,
    save_path="/kaggle/working/gradcam_tn.png"
)

# ============================================================
# CELL 9: Grad-CAM Visualization - False Positives
# ============================================================
print("\n" + "="*60)
print("Generating Grad-CAM for FALSE POSITIVES")
print("(Normal predicted as Pneumonia)")
print("="*60)

if len(indices['fp']) > 0:
    fp_indices = indices['fp'][:8]
    fp_paths = [test_gen.filepaths[i] for i in fp_indices]
    fp_labels = [test_gen.classes[i] for i in fp_indices]
    
    plot_gradcam_grid(
        model, 
        fp_paths, 
        true_labels=fp_labels,
        ncols=4,
        save_path="/kaggle/working/gradcam_fp.png"
    )
else:
    print("No false positives! Perfect specificity!")

# ============================================================
# CELL 10: Grad-CAM Visualization - False Negatives
# ============================================================
print("\n" + "="*60)
print("Generating Grad-CAM for FALSE NEGATIVES")
print("(Pneumonia missed by model - CRITICAL!)")
print("="*60)

if len(indices['fn']) > 0:
    fn_indices = indices['fn'][:8]
    fn_paths = [test_gen.filepaths[i] for i in fn_indices]
    fn_labels = [test_gen.classes[i] for i in fn_indices]
    
    plot_gradcam_grid(
        model, 
        fn_paths, 
        true_labels=fn_labels,
        ncols=4,
        save_path="/kaggle/working/gradcam_fn.png"
    )
else:
    print("No false negatives! Perfect sensitivity!")

# ============================================================
# CELL 11: Test Single Image (Interactive)
# ============================================================
print("\n" + "="*60)
print("SINGLE IMAGE TEST")
print("="*60)

# Pick a random test image
import random
random.seed(42)
test_idx = random.randint(0, len(test_gen.filepaths) - 1)
test_path = test_gen.filepaths[test_idx]
true_label = test_gen.classes[test_idx]

print(f"Testing: {Path(test_path).name}")
print(f"True label: {config.CLASS_NAMES[true_label]}")

# Load and preprocess
img, img_array, processed = load_and_preprocess_image(test_path)

# Get prediction
pred_logits = model.predict(processed, verbose=0)
pred_probs = tf.nn.softmax(pred_logits[0]).numpy()
pred_class = pred_probs.argmax()

print(f"Predicted: {config.CLASS_NAMES[pred_class]} ({pred_probs[pred_class]:.2%} confidence)")
print(f"Probabilities: NORMAL={pred_probs[0]:.2%}, PNEUMONIA={pred_probs[1]:.2%}")

# Generate Grad-CAM
heatmap, _ = make_gradcam_heatmap(processed, model)
overlay_img = overlay_heatmap(img_array[0].astype("uint8"), heatmap, alpha=0.4)

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(img_array[0].astype("uint8"))
axes[0].set_title(f"Original\nTrue: {config.CLASS_NAMES[true_label]}")
axes[0].axis("off")

axes[1].imshow(heatmap, cmap="jet")
axes[1].set_title("Grad-CAM Heatmap")
axes[1].axis("off")

axes[2].imshow(overlay_img)
axes[2].set_title(f"Overlay\nPred: {config.CLASS_NAMES[pred_class]} ({pred_probs[pred_class]:.1%})")
axes[2].axis("off")

plt.tight_layout()
plt.savefig("/kaggle/working/single_test.png", dpi=150, bbox_inches="tight")
plt.show()

# ============================================================
# CELL 12: Summary Report
# ============================================================
print("\n" + "="*60)
print("TESTING COMPLETE - SUMMARY")
print("="*60)
print(f"\n✓ Model: {Path(model_path).name}")
print(f"✓ Test samples evaluated: {len(y_true)}")
print(f"✓ Overall accuracy: {metrics['accuracy']:.2%}")
print(f"✓ ROC-AUC: {metrics['roc_auc']:.4f}")
print(f"\n✓ Generated files in /kaggle/working/:")
print("  - kaggle_test_metrics.json")
print("  - kaggle_test_confusion.png")
print("  - kaggle_test_roc.png")
print("  - gradcam_tp.png (True Positives)")
print("  - gradcam_tn.png (True Negatives)")
print("  - gradcam_fp.png (False Positives)")
print("  - gradcam_fn.png (False Negatives)")
print("  - single_test.png")
print("\n" + "="*60)
print("Download these files from the Output section!")
print("="*60)
```

---

## 📊 **What This Does**

1. **Clones your code** from GitHub
2. **Loads your trained model** from Kaggle dataset
3. **Evaluates** on test set (624 images)
4. **Generates metrics**: Precision, Recall, F1, ROC-AUC
5. **Creates visualizations**: Confusion matrix, ROC curve
6. **Grad-CAM analysis**: TP, TN, FP, FN cases
7. **Single image demo**: Interactive prediction with heatmap

---

## 📥 **Getting Results**

After running the notebook:

1. **Go to Output tab** in your Kaggle notebook
2. **Download files**:
   - `kaggle_test_metrics.json` - Detailed metrics
   - `kaggle_test_confusion.png` - Confusion matrix
   - `kaggle_test_roc.png` - ROC curve
   - `gradcam_*.png` - Visualization grids

3. **Optional: Save as Dataset**
   - Click "New Dataset" in Output tab
   - Name it `pneumonia-test-results`
   - Share with collaborators

---

## 🔧 **Troubleshooting**

### **Issue: "Model file not found"**
```python
# List available files
!ls -la /kaggle/input/pneumonia-vgg16-model/
```
Then adjust the path in Cell 3.

### **Issue: "Layer 'block5_conv3' not found"**
Your model architecture might be different. Check:
```python
for layer in model.layers:
    print(layer.name)
```
Update `config.LAST_CONV_LAYER` if needed.

### **Issue: "Git clone failed"**
- Make sure **Internet is ON** in notebook settings
- Make sure your GitHub repo is **Public** (or use authentication)
- Check the URL is correct

### **Issue: "Data not found"**
```python
# Check data location
!ls -la /kaggle/input/
!ls -la /kaggle/input/chest-xray-pneumonia/
```
The dataset should show `train/` and `test/` folders.

### **Issue: Model outputs softmax instead of logits**
If Grad-CAM heatmaps are blank, your model might have softmax activation. Rebuild model:
```python
from src.model import build_vgg16, compile_model

# Rebuild model structure
new_model = build_vgg16()
compile_model(new_model)

# Load only the weights
new_model.load_weights(model_path)
model = new_model
```

---

## 💡 **Advanced: Compare Multiple Models**

If you have multiple models (baseline, VGG16, VGG19):

```python
models = {
    "Baseline CNN": "/kaggle/input/models/baseline_cnn.h5",
    "VGG16 Stage1": "/kaggle/input/models/vgg16_stage1.h5",
    "VGG16 Final": "/kaggle/input/models/vgg16_pneumonia.h5",
}

results = []
for name, path in models.items():
    print(f"\nEvaluating {name}...")
    model = tf.keras.models.load_model(path)
    y_true, probs = predict_probs(model, test_gen)
    metrics = full_report(y_true, probs, name.replace(" ", "_").lower())
    results.append({
        "Model": name,
        "Accuracy": metrics['accuracy'],
        "Precision": metrics['PNEUMONIA']['precision'],
        "Recall": metrics['PNEUMONIA']['recall'],
        "F1": metrics['PNEUMONIA']['f1-score'],
        "ROC-AUC": metrics['roc_auc']
    })

# Display comparison table
import pandas as pd
comparison = pd.DataFrame(results)
print("\n" + "="*60)
print("MODEL COMPARISON")
print("="*60)
print(comparison.to_string(index=False))
```

---

## 📝 **Quick Start Checklist**

- [ ] Upload model to Kaggle dataset
- [ ] Create new Kaggle notebook
- [ ] Add chest X-ray dataset as input
- [ ] Add your model dataset as input
- [ ] Enable GPU and Internet in settings
- [ ] Copy testing code (cells 1-12)
- [ ] Update `GITHUB_URL` with your repo
- [ ] Run all cells
- [ ] Download results from Output tab

---

## 🎯 **Expected Output**

You should see:
- ✅ Accuracy around **85-95%** (depending on training)
- ✅ Pneumonia Recall **>90%** (critical metric)
- ✅ ROC-AUC **>0.90**
- ✅ Grad-CAM heatmaps showing lung focus areas
- ✅ All visualizations saved

---

## 📧 **Need Help?**

If you encounter issues:
1. Check the **Console output** for error messages
2. Verify all file paths are correct
3. Make sure Internet and GPU are enabled
4. Try running cells one at a time to isolate issues

Good luck testing your model! 🚀
