# Phase 1 Implementation - Usage Guide

## Overview
Dev A's modules (`src/data.py` and `src/model.py`) are now complete and tested.

## ✅ What's Implemented

### `src/data.py`
- **`get_generators(root=None)`**: Creates train/val/test data generators
  - Uses VGG16's `preprocess_input` (ImageNet preprocessing)
  - Train: augmented (shear, zoom, horizontal flip)
  - Val/Test: no augmentation
  - Returns: `(train_gen, val_gen, test_gen)`

- **`get_class_weights(train_gen)`**: Computes balanced class weights
  - Handles class imbalance (NORMAL vs PNEUMONIA)
  - Returns: `{0: weight, 1: weight}` dictionary

### `src/model.py`
- **`build_vgg16(dropout=0.5)`**: Builds VGG16 transfer learning model
  - VGG16 base (frozen) + Flatten + Dropout + Dense(2, logits)
  - Outputs raw logits (no softmax)
  - Total params: 14.7M (50K trainable initially)

- **`compile_model(model, lr=1e-4)`**: Compiles model for Stage 1 training
  - Adam optimizer with learning rate 1e-4
  - CategoricalCrossentropy with `from_logits=True`
  - Metric: accuracy

- **`unfreeze_block5(model, lr=1e-5)`**: Enables Stage 2 fine-tuning
  - Unfreezes Block 5 layers
  - Lower learning rate (1e-5) to prevent catastrophic forgetting
  - Trainable params increase: 50K → 7.1M

## 📊 Test Results

```
✓ Configuration: PASSED
✓ Model Building: PASSED
✓ Model Compilation: PASSED
✓ Block5 Unfreezing: PASSED
✓ Phase 1 Implementation: COMPLETE
```

Run tests: `python test_phase1.py`

## 🚀 Quick Start Example

```python
from src.data import get_generators, get_class_weights
from src.model import build_vgg16, compile_model, unfreeze_block5

# 1. Load data
train_gen, val_gen, test_gen = get_generators()
class_weights = get_class_weights(train_gen)

# 2. Build and compile model (Stage 1)
model = build_vgg16(dropout=0.5)
compile_model(model, lr=1e-4)

# 3. Train Stage 1 (frozen base)
history1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    class_weight=class_weights
)

# 4. Fine-tune Stage 2 (unfreeze Block5)
unfreeze_block5(model, lr=1e-5)
history2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=5,
    class_weight=class_weights
)

# 5. Save model
model.save('models/vgg16_pneumonia.keras')
```

## 📁 Dataset Requirements

Download the Kaggle Chest X-Ray dataset:
- **URL**: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
- **Structure**:
  ```
  data/chest_xray/
  ├── train/
  │   ├── NORMAL/
  │   └── PNEUMONIA/
  └── test/
      ├── NORMAL/
      └── PNEUMONIA/
  ```

The code will auto-detect the dataset in:
1. `/kaggle/input` (Kaggle notebooks)
2. `/content` (Google Colab)
3. `data/` (local development)

## 🔧 Configuration

All settings are in `src/config.py`:
- **Image size**: 224×224 (VGG16 input)
- **Batch size**: 32
- **Validation split**: 15% from training data
- **Classes**: NORMAL (0), PNEUMONIA (1)
- **Seed**: 42 (reproducibility)

## ⚙️ Key Design Decisions

1. **VGG16 Preprocessing**: Uses `preprocess_input` (ImageNet normalization), NOT `rescale=1/255`
2. **Logits Output**: Dense layer has NO softmax activation to prevent gradient vanishing
3. **from_logits=True**: Loss function handles softmax internally
4. **Two-Stage Training**: 
   - Stage 1: Train only classification head (frozen base)
   - Stage 2: Fine-tune Block5 with lower learning rate
5. **Class Weights**: Handles imbalanced dataset (more PNEUMONIA than NORMAL samples)

## 📝 Next Steps (Phase 2 - Dev B)

Phase 2 will implement:
- `src/gradcam.py`: Grad-CAM visualization
- `src/eval_utils.py`: Model evaluation and metrics
- Integration with Gradio for interactive demo

## 🧪 Testing

Run the test suite to validate installation:
```bash
python test_phase1.py
```

Expected output:
- Configuration validated
- Model built with correct architecture
- Compilation successful
- Block5 unfreezing works correctly
- All tests PASSED ✓
