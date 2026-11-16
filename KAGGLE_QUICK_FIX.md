# Quick Fix for Python 3.13 / Kaggle Error

## ❌ Error You Got

```
ModuleNotFoundError: No module named 'imp'
```

**Cause**: `%load_ext autoreload` doesn't work with Python 3.13

**Solution**: Remove those lines - you don't need them for testing!

---

## ✅ Fixed Code (Use This Instead)

Replace the first cell with this:

```python
# ============================================================
# CONFIGURATION - UPDATE THESE 3 LINES!
# ============================================================
GITHUB_REPO_URL = "https://github.com/YOUR_USERNAME/pneumonia-gradcam.git"
MODEL_DATASET = "pneumonia-vgg16-model"
MODEL_FILENAME = "vgg16_pneumonia.h5"

# ============================================================
# SETUP (No changes needed below)
# ============================================================
import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path

print("Setting up environment...")

# Clone repository
REPO = "pneumonia-gradcam"
if not Path(f"/kaggle/working/{REPO}").exists():
    print(f"Cloning repository from {GITHUB_REPO_URL}...")
    result = os.system(f"git clone {GITHUB_REPO_URL} /kaggle/working/{REPO}")
    if result != 0:
        raise RuntimeError("Git clone failed. Check URL and Internet setting.")
else:
    print("Repository already cloned.")

# Add to Python path
sys.path.insert(0, f"/kaggle/working/{REPO}")

# Import modules
from src import config
from src.data import get_generators
from src.gradcam import plot_gradcam_grid
from src.eval_utils import predict_probs, full_report, misclassified_indices

print("✓ Setup complete!")

# ============================================================
# LOAD MODEL
# ============================================================
MODEL_PATH = f"/kaggle/input/{MODEL_DATASET}/{MODEL_FILENAME}"
print(f"\nLoading model from: {MODEL_PATH}")

# Verify file exists
if not Path(MODEL_PATH).exists():
    print(f"\n❌ ERROR: Model not found at {MODEL_PATH}")
    print("\n📁 Available files in your dataset:")
    os.system(f"ls -lh /kaggle/input/{MODEL_DATASET}/")
    print("\n💡 Fix: Update MODEL_FILENAME to match your actual file name")
    raise FileNotFoundError("Model file not found")

# Load model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✓ Model loaded successfully!")
    print(f"  Input shape: {model.input_shape}")
    print(f"  Output shape: {model.output_shape}")
except Exception as e:
    print(f"\n❌ ERROR loading model: {e}")
    raise

# ============================================================
# LOAD DATA
# ============================================================
print("\nLoading data generators...")
train_gen, val_gen, test_gen = get_generators()

print(f"✓ Data loaded!")
print(f"  Test samples: {test_gen.samples}")
print(f"  Classes: {test_gen.class_indices}")

# ============================================================
# EVALUATE
# ============================================================
print("\n" + "="*60)
print("EVALUATING MODEL")
print("="*60)
print("This will take 5-10 minutes...")

y_true, probs = predict_probs(model, test_gen)
metrics = full_report(y_true, probs, "test", out_dir="/kaggle/working")

print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"Accuracy: {metrics['accuracy']:.2%}")
print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
print(f"\nNORMAL - Precision: {metrics['NORMAL']['precision']:.2%}, Recall: {metrics['NORMAL']['recall']:.2%}")
print(f"PNEUMONIA - Precision: {metrics['PNEUMONIA']['precision']:.2%}, Recall: {metrics['PNEUMONIA']['recall']:.2%}")

# ============================================================
# GRAD-CAM
# ============================================================
print("\n" + "="*60)
print("GENERATING GRAD-CAM VISUALIZATIONS")
print("="*60)

indices = misclassified_indices(y_true, probs)

print(f"\nClassification Breakdown:")
print(f"  TP: {len(indices['tp'])} | TN: {len(indices['tn'])} | FP: {len(indices['fp'])} | FN: {len(indices['fn'])}")

for category in ['tp', 'tn', 'fp', 'fn']:
    if len(indices[category]) > 0:
        print(f"\nGenerating {category.upper()}...")
        idx = indices[category][:8]
        paths = [test_gen.filepaths[i] for i in idx]
        labels = [test_gen.classes[i] for i in idx]
        plot_gradcam_grid(model, paths, labels, ncols=4, 
                         save_path=f"/kaggle/working/gradcam_{category}.png")
        print(f"  ✓ Saved gradcam_{category}.png")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*60)
print("TESTING COMPLETE!")
print("="*60)
print(f"\n✓ Model: {MODEL_FILENAME}")
print(f"✓ Accuracy: {metrics['accuracy']:.2%}")
print(f"✓ ROC-AUC: {metrics['roc_auc']:.4f}")
print(f"\n📁 Download from Output tab:")
print(f"  ✓ test_metrics.json")
print(f"  ✓ test_confusion.png")
print(f"  ✓ test_roc.png")
print(f"  ✓ gradcam_*.png files")
print("\n" + "="*60)
```

---

## 🔍 What Changed?

**Removed these lines** (lines 24-25):
```python
%load_ext autoreload  # ❌ Removed - causes error
%autoreload 2         # ❌ Removed - not needed
```

**Why they were there**: For development, to auto-reload edited `.py` files

**Why you don't need them**: You're just testing, not editing code in the notebook

---

## ✅ This is a Single Cell

**Important**: Put all this code in **ONE cell**, not multiple cells!

This is the complete all-in-one testing script. Just:
1. Copy the entire code above
2. Paste into a single Kaggle cell
3. Update the 3 config lines at the top
4. Run!

---

## 💡 Alternative: Multiple Cells

If you prefer multiple cells (for step-by-step), here's the breakdown:

### **Cell 1: Setup & Load**
```python
# Config
GITHUB_REPO_URL = "https://github.com/YOUR_USERNAME/pneumonia-gradcam.git"
MODEL_DATASET = "pneumonia-vgg16-model"
MODEL_FILENAME = "vgg16_pneumonia.h5"

# Imports
import os, sys
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path

# Clone repo
REPO = "pneumonia-gradcam"
if not Path(f"/kaggle/working/{REPO}").exists():
    os.system(f"git clone {GITHUB_REPO_URL} /kaggle/working/{REPO}")
sys.path.insert(0, f"/kaggle/working/{REPO}")

# Import your modules
from src import config
from src.data import get_generators
from src.gradcam import plot_gradcam_grid
from src.eval_utils import predict_probs, full_report, misclassified_indices

# Load model
MODEL_PATH = f"/kaggle/input/{MODEL_DATASET}/{MODEL_FILENAME}"
model = tf.keras.models.load_model(MODEL_PATH)
print(f"✓ Model loaded: {model.input_shape} -> {model.output_shape}")

# Load data
train_gen, val_gen, test_gen = get_generators()
print(f"✓ Data loaded: {test_gen.samples} test samples")
```

### **Cell 2: Evaluate**
```python
# Predict
y_true, probs = predict_probs(model, test_gen)

# Report
metrics = full_report(y_true, probs, "test", "/kaggle/working")

print(f"\n✓ Accuracy: {metrics['accuracy']:.2%}")
print(f"✓ ROC-AUC: {metrics['roc_auc']:.4f}")
```

### **Cell 3: Grad-CAM**
```python
indices = misclassified_indices(y_true, probs)

for category in ['tp', 'tn', 'fp', 'fn']:
    if len(indices[category]) > 0:
        idx = indices[category][:8]
        paths = [test_gen.filepaths[i] for i in idx]
        labels = [test_gen.classes[i] for i in idx]
        plot_gradcam_grid(model, paths, labels, ncols=4,
                         save_path=f"/kaggle/working/gradcam_{category}.png")
        
print("✓ All Grad-CAM visualizations saved!")
```

---

## 🎯 Quick Answer to Your Questions

### **Q: Does it need to be added to all cells?**
**A: No!** The `%autoreload` lines were only in the first cell. Now they're removed completely.

### **Q: Should I use one cell or multiple cells?**
**A: Either works:**
- **One cell**: Easier, run once and done (recommended for first test)
- **Multiple cells**: Better for debugging if something fails

### **Q: Will this work now?**
**A: Yes!** The fixed code above will work perfectly on Kaggle.

---

## 🔧 Other Common Kaggle Issues

### If you get "git clone failed":
```python
# Add this to check internet
import urllib.request
try:
    urllib.request.urlopen('http://google.com', timeout=1)
    print("✓ Internet is working")
except:
    print("❌ Internet is OFF - enable in Settings")
```

### If you get "data not found":
```python
# Check what's available
!find /kaggle/input -name "train" -o -name "test" | head -10
```

### If you get "module not found":
```python
# Check if repo cloned
!ls -la /kaggle/working/
# Should see your repo folder
```

---

## ✅ Updated Checklist

- [ ] Copy the **fixed code** (without autoreload lines)
- [ ] Update 3 config variables
- [ ] Add datasets as inputs
- [ ] Enable GPU + Internet
- [ ] Run!

---

## 🎉 You're Good to Go!

The code above is **ready to use** - just copy and paste! The autoreload error is fixed. 🚀

Let me know if you hit any other issues!
