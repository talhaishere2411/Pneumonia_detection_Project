# Testing with .h5 Model Files - Quick Setup

## 📁 What You Have

Your model dataset contains:
```
pneumonia-vgg16-model/
├── vgg16_pneumonia.h5       ← Your trained model (Keras format)
├── config.json              ← Training configuration
└── metadata.json            ← Training metrics/info
```

**Good news**: Both `.h5` and `.keras` formats work perfectly with `tf.keras.models.load_model()`!

---

## 🚀 Quick Setup (3 Steps)

### **Step 1: Upload to Kaggle** (2 minutes)

1. Go to https://www.kaggle.com/datasets
2. Click **"New Dataset"**
3. **Upload all 3 files**:
   - ✅ `vgg16_pneumonia.h5`
   - ✅ `config.json` (optional, but useful)
   - ✅ `metadata.json` (optional, but useful)
4. **Dataset title**: `pneumonia-vgg16-model`
5. **Visibility**: Private
6. Click **"Create"**

💡 **Tip**: Upload all files even if you only need the `.h5` - keeps everything organized!

---

### **Step 2: Create Kaggle Notebook**

1. Go to https://www.kaggle.com/code
2. Click **"New Notebook"**
3. **Settings** (right panel):
   - Accelerator: **GPU T4 x2** ✓
   - Internet: **ON** ✓
   - Language: **Python** ✓

---

### **Step 3: Copy Testing Code**

Paste this into your Kaggle notebook:

```python
# ============================================================
# CONFIGURATION - UPDATE THESE 3 LINES!
# ============================================================
GITHUB_REPO_URL = "https://github.com/YOUR_USERNAME/pneumonia-gradcam.git"
MODEL_DATASET = "pneumonia-vgg16-model"     # Your Kaggle dataset name
MODEL_FILENAME = "vgg16_pneumonia.h5"       # Your .h5 file name

# ============================================================
# SETUP (No changes needed below)
# ============================================================
import os, sys
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

sys.path.insert(0, f"/kaggle/working/{REPO}")

# Import modules
from src import config
from src.data import get_generators
from src.gradcam import plot_gradcam_grid
from src.eval_utils import predict_probs, full_report, misclassified_indices

print("✓ Setup complete!")

# ============================================================
# LOAD MODEL (.h5 format works perfectly!)
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

# Load model (works with both .h5 and .keras!)
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✓ Model loaded successfully!")
    print(f"  Input shape: {model.input_shape}")
    print(f"  Output shape: {model.output_shape}")
except Exception as e:
    print(f"\n❌ ERROR loading model: {e}")
    print("\n💡 Possible fixes:")
    print("  1. Make sure the file is a valid Keras model")
    print("  2. Check if custom layers were used during training")
    print("  3. Try loading with compile=False:")
    print("     model = tf.keras.models.load_model(MODEL_PATH, compile=False)")
    raise

# ============================================================
# LOAD DATA
# ============================================================
print("\nLoading data generators...")
train_gen, val_gen, test_gen = get_generators()

print(f"✓ Data loaded!")
print(f"  Train samples: {train_gen.samples}")
print(f"  Validation samples: {val_gen.samples}")
print(f"  Test samples: {test_gen.samples}")
print(f"  Classes: {test_gen.class_indices}")

# ============================================================
# EVALUATE MODEL
# ============================================================
print("\n" + "="*60)
print("EVALUATING MODEL ON TEST SET")
print("="*60)
print("This will take 5-10 minutes for 624 images...")

y_true, probs = predict_probs(model, test_gen)
metrics = full_report(y_true, probs, "test", out_dir="/kaggle/working")

print("\n" + "="*60)
print("EVALUATION COMPLETE!")
print("="*60)
print(f"\n📊 Key Metrics:")
print(f"  Overall Accuracy: {metrics['accuracy']:.2%}")
print(f"  ROC-AUC Score: {metrics['roc_auc']:.4f}")
print(f"\n  NORMAL:")
print(f"    Precision: {metrics['NORMAL']['precision']:.2%}")
print(f"    Recall: {metrics['NORMAL']['recall']:.2%}")
print(f"    F1-Score: {metrics['NORMAL']['f1-score']:.4f}")
print(f"\n  PNEUMONIA:")
print(f"    Precision: {metrics['PNEUMONIA']['precision']:.2%}")
print(f"    Recall: {metrics['PNEUMONIA']['recall']:.2%} ← Critical!")
print(f"    F1-Score: {metrics['PNEUMONIA']['f1-score']:.4f}")

# ============================================================
# GRAD-CAM VISUALIZATION
# ============================================================
print("\n" + "="*60)
print("GENERATING GRAD-CAM VISUALIZATIONS")
print("="*60)

indices = misclassified_indices(y_true, probs)

print(f"\n📈 Classification Breakdown:")
print(f"  True Positives (TP): {len(indices['tp'])} - Correctly identified pneumonia")
print(f"  True Negatives (TN): {len(indices['tn'])} - Correctly identified normal")
print(f"  False Positives (FP): {len(indices['fp'])} - Normal misclassified as pneumonia")
print(f"  False Negatives (FN): {len(indices['fn'])} - Pneumonia missed (CRITICAL!)")

categories = {
    'tp': 'True Positives',
    'tn': 'True Negatives', 
    'fp': 'False Positives',
    'fn': 'False Negatives'
}

for category, label in categories.items():
    if len(indices[category]) > 0:
        print(f"\nGenerating Grad-CAM for {label}...")
        idx = indices[category][:8]  # First 8 cases
        paths = [test_gen.filepaths[i] for i in idx]
        labels = [test_gen.classes[i] for i in idx]
        
        plot_gradcam_grid(
            model, 
            paths, 
            labels, 
            ncols=4, 
            save_path=f"/kaggle/working/gradcam_{category}.png"
        )
        print(f"  ✓ Saved gradcam_{category}.png ({len(idx)} cases)")
    else:
        print(f"\n✓ No {label}! (Perfect {category.upper()})")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*60)
print("TESTING COMPLETE!")
print("="*60)
print(f"\n✓ Model: {MODEL_FILENAME}")
print(f"✓ Test samples: {len(y_true)}")
print(f"✓ Accuracy: {metrics['accuracy']:.2%}")
print(f"✓ ROC-AUC: {metrics['roc_auc']:.4f}")

print(f"\n📁 Generated Files (download from Output tab):")
print(f"  ✓ test_metrics.json - Detailed metrics")
print(f"  ✓ test_confusion.png - Confusion matrix")
print(f"  ✓ test_roc.png - ROC curve")
print(f"  ✓ gradcam_tp.png - True Positives visualization")
print(f"  ✓ gradcam_tn.png - True Negatives visualization")
if len(indices['fp']) > 0:
    print(f"  ✓ gradcam_fp.png - False Positives (errors)")
if len(indices['fn']) > 0:
    print(f"  ✓ gradcam_fn.png - False Negatives (missed cases)")

print("\n" + "="*60)
print("Go to Output tab → Download files → Done! 🎉")
print("="*60)
```

---

## 📊 What This Does

1. ✅ Clones your GitHub repository
2. ✅ Loads your `.h5` model file
3. ✅ Loads test dataset (624 X-ray images)
4. ✅ Evaluates model performance
5. ✅ Calculates all metrics (Precision, Recall, F1, ROC-AUC)
6. ✅ Generates visualizations (confusion matrix, ROC curve)
7. ✅ Creates Grad-CAM heatmaps for all case types
8. ✅ Saves everything to `/kaggle/working/`

**Time**: ~5-10 minutes total

---

## 🔍 About the JSON Files

### `config.json` (Training Configuration)
Contains training settings like:
```json
{
  "epochs": 10,
  "batch_size": 32,
  "learning_rate": 0.0001,
  "optimizer": "adam",
  "validation_split": 0.15
}
```

### `metadata.json` (Training Results)
Contains training history:
```json
{
  "final_accuracy": 0.92,
  "final_val_accuracy": 0.89,
  "training_time": "45 minutes",
  "best_epoch": 8
}
```

💡 **Note**: These JSON files are optional for testing. The `.h5` file contains everything needed to run the model!

---

## 🔧 Troubleshooting

### ❌ "Model file not found"

**Check what you uploaded:**
```python
!ls -la /kaggle/input/pneumonia-vgg16-model/
```

**Output should show:**
```
vgg16_pneumonia.h5
config.json
metadata.json
```

**Fix**: Update `MODEL_FILENAME` to match exact name (case-sensitive!)

---

### ❌ "Git clone failed"

**Check 1**: Internet enabled?
- Settings → Internet → **ON** ✓

**Check 2**: Repo is public?
- Go to GitHub → Repository → Settings → General
- Make sure it's **Public** (or add authentication)

**Check 3**: URL correct?
```python
# Test the URL
!curl -I https://github.com/YOUR_USERNAME/pneumonia-gradcam
# Should return "200 OK"
```

---

### ❌ "Layer 'block5_conv3' not found"

Your model architecture might be different.

**Check layers:**
```python
print("Model layers:")
for i, layer in enumerate(model.layers):
    print(f"{i}: {layer.name}")
```

**If block5_conv3 is missing**, find the last conv layer and update config:
```python
# Update before generating Grad-CAM
config.LAST_CONV_LAYER = "your_actual_conv_layer_name"
```

---

### ❌ "Grad-CAM heatmaps are blank"

Model might have softmax activation on the output layer.

**Fix**: Rebuild without softmax:
```python
from src.model import build_vgg16, compile_model

# Build architecture without softmax
new_model = build_vgg16()
compile_model(new_model)

# Load weights from your .h5 file
new_model.load_weights(MODEL_PATH)

# Use new_model instead
model = new_model
```

---

### ❌ "Data not found"

**Check if dataset is added:**
```python
!ls -la /kaggle/input/
```

**Should see:**
- `chest-xray-pneumonia/` (or similar)
- `pneumonia-vgg16-model/`

**Fix**: Add the chest X-ray dataset as input:
1. Right panel → **Add Input**
2. Search: `chest-xray-pneumonia`
3. Add **Paul Mooney's dataset**

---

## 💡 Pro Tips

### 1. Check Files First
Before running everything, verify files:
```python
# List your model dataset
!ls -lh /kaggle/input/pneumonia-vgg16-model/

# List X-ray dataset
!find /kaggle/input -name "train" -o -name "test" | head -5
```

### 2. Load Model with Error Handling
```python
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✓ Loaded successfully")
except Exception as e:
    print(f"❌ Error: {e}")
    # Try without compiling
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("✓ Loaded without compilation")
```

### 3. Test Small Batch First
```python
# Test on first 32 images only
test_gen_small = test_gen
test_gen_small.samples = 32
y_true, probs = predict_probs(model, test_gen_small)
```

### 4. Download Results Immediately
After running, **immediately download** from Output tab.
Kaggle sessions can disconnect!

---

## ✅ Quick Checklist

**Before Running:**
- [ ] `.h5` file uploaded to Kaggle dataset
- [ ] Dataset name and filename are correct in code
- [ ] GitHub repo is public and pushed
- [ ] Kaggle notebook has GPU + Internet enabled
- [ ] Both datasets added as inputs

**After Running:**
- [ ] Model loaded successfully
- [ ] Evaluation completed
- [ ] Metrics look reasonable (accuracy 85-95%)
- [ ] Grad-CAM images generated
- [ ] All files downloaded from Output tab

---

## 📥 What to Download

From **Output** tab, download:

```
test_metrics.json          ← Open in text editor
test_confusion.png         ← View confusion matrix
test_roc.png              ← View ROC curve
gradcam_tp.png            ← True positives visualization
gradcam_tn.png            ← True negatives visualization
gradcam_fp.png            ← False positives (if any)
gradcam_fn.png            ← False negatives (if any)
```

---

## 🎯 Expected Results

### **Good Performance:**
- ✅ Accuracy: 85-95%
- ✅ Pneumonia Recall: >90%
- ✅ ROC-AUC: >0.90
- ✅ Grad-CAM focuses on lungs

### **Needs Improvement:**
- ⚠️ Accuracy <85% or >98%
- ⚠️ Pneumonia Recall <85%
- ⚠️ Heatmaps on corners/edges
- ⚠️ Many false negatives

---

## 🚀 Ready to Test!

1. **Copy the complete code above**
2. **Update 3 variables** at the top
3. **Add inputs** (chest X-ray + your model)
4. **Run the cell**
5. **Download results** from Output tab

That's it! Your `.h5` model will work perfectly. 🎉

---

## 📞 Still Need Help?

Common issues are usually:
1. Wrong file path → Check with `!ls`
2. Internet not enabled → Settings → Internet ON
3. Dataset not added → Add Input → Search dataset
4. Wrong GitHub URL → Verify it's public and correct

**Time to test: ~10 minutes** ⏱️
