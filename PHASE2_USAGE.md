# Phase 2 Implementation - Usage Guide

## Overview
Dev B's modules (`src/gradcam.py` and `src/eval_utils.py`) are now complete and tested.

## ✅ What's Implemented

### `src/gradcam.py` - Grad-CAM Visualization

#### `load_and_preprocess_image(image_path)`
Loads and preprocesses a single image for model input.
- **Returns**: `(img, img_array, processed_img)`
- **Usage**:
  ```python
  img, img_array, processed = load_and_preprocess_image("path/to/xray.jpg")
  prediction = model.predict(processed)
  ```

#### `make_gradcam_heatmap(img_array, model, last_conv_layer_name=None, pred_index=None)`
Generates Grad-CAM heatmap showing model focus areas.
- **Args**:
  - `img_array`: Preprocessed image batch (1, H, W, 3)
  - `model`: Trained model (MUST output logits, not softmax)
  - `last_conv_layer_name`: Conv layer name (default: config.LAST_CONV_LAYER)
  - `pred_index`: Target class (None = predicted class)
- **Returns**: `(heatmap, probabilities)`
- **Usage**:
  ```python
  heatmap, probs = make_gradcam_heatmap(processed, model)
  print(f"Predicted: {config.CLASS_NAMES[probs.argmax()]}")
  ```

#### `overlay_heatmap(img, heatmap, alpha=0.4)`
Overlays colorized heatmap on original image.
- **Returns**: RGB image with heatmap overlay
- **Usage**:
  ```python
  overlay_img = overlay_heatmap(img_array[0].astype("uint8"), heatmap)
  plt.imshow(overlay_img)
  ```

#### `plot_gradcam_grid(model, paths, true_labels=None, ncols=4, save_path=None)`
Creates grid visualization of multiple images with Grad-CAM.
- **Usage**:
  ```python
  plot_gradcam_grid(
      model, 
      image_paths=["xray1.jpg", "xray2.jpg"],
      true_labels=[0, 1],
      save_path="results/gradcam_grid.png"
  )
  ```

### `src/eval_utils.py` - Model Evaluation

#### `get_model_prediction(model, processed_img)`
Single image prediction with confidence scores.
- **Returns**: Dict with `class_idx`, `class_name`, `confidence`, `probabilities`
- **Usage**:
  ```python
  result = get_model_prediction(model, processed)
  print(f"{result['class_name']}: {result['confidence']:.2%}")
  ```

#### `predict_probs(model, generator)`
Batch predictions on entire dataset.
- **Important**: Generator MUST have `shuffle=False`
- **Returns**: `(y_true, probs)`
- **Usage**:
  ```python
  y_true, probs = predict_probs(model, test_gen)
  y_pred = probs.argmax(axis=1)
  ```

#### `full_report(y_true, probs, tag, out_dir="results")`
Comprehensive evaluation with metrics, confusion matrix, and ROC curve.
- **Creates**:
  - `{tag}_metrics.json`: Precision, recall, F1, ROC-AUC
  - `{tag}_confusion.png`: Confusion matrix
  - `{tag}_roc.png`: ROC curve
- **Usage**:
  ```python
  metrics = full_report(y_true, probs, "vgg16_test")
  print(f"ROC-AUC: {metrics['roc_auc']:.3f}")
  ```

#### `plot_history(history, tag, out_dir="results")`
Plots training/validation curves from model.fit().
- **Usage**:
  ```python
  history = model.fit(train_gen, validation_data=val_gen, epochs=10)
  plot_history(history, "stage1")
  ```

#### `misclassified_indices(y_true, probs)`
Identifies TP, TN, FP, FN cases for error analysis.
- **Returns**: Dict with arrays of indices for each category
- **Usage**:
  ```python
  indices = misclassified_indices(y_true, probs)
  fn_paths = [test_gen.filepaths[i] for i in indices['fn']]
  # Analyze false negative cases
  ```

#### `print_misclassification_summary(y_true, probs)`
Prints detailed breakdown of classification results.

## 🚀 Complete Example Workflow

```python
from src.data import get_generators
from src.model import build_vgg16, compile_model
from src.gradcam import load_and_preprocess_image, make_gradcam_heatmap, overlay_heatmap
from src.eval_utils import predict_probs, full_report, misclassified_indices
import matplotlib.pyplot as plt

# 1. Load data and model
train_gen, val_gen, test_gen = get_generators()
model = tf.keras.models.load_model("models/vgg16_pneumonia.keras")

# 2. Evaluate on test set
y_true, probs = predict_probs(model, test_gen)
metrics = full_report(y_true, probs, "vgg16_final")

# 3. Analyze misclassifications
indices = misclassified_indices(y_true, probs)
print(f"False Negatives (missed pneumonia): {len(indices['fn'])}")

# 4. Visualize Grad-CAM for specific case
fn_idx = indices['fn'][0]  # First false negative
image_path = test_gen.filepaths[fn_idx]

_, img_array, processed = load_and_preprocess_image(image_path)
heatmap, probs = make_gradcam_heatmap(processed, model)

# Show original and overlay
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(img_array[0].astype("uint8"))
ax1.set_title("Original")
ax2.imshow(overlay_heatmap(img_array[0].astype("uint8"), heatmap))
ax2.set_title(f"Grad-CAM (pred: {probs.argmax()})")
plt.savefig("results/fn_analysis.png")
```

## 📊 Test Results

```
✓ Grad-CAM Functions: PASSED
✓ Evaluation Functions: PASSED
✓ Module Integration: PASSED
✓ Phase 2 Implementation: COMPLETE
```

Run tests: `python test_phase2.py`

## 🔍 Key Implementation Details

### Grad-CAM Design Decisions

1. **Logits-based Model**: Model must output raw logits (no softmax) for correct gradient computation
2. **Softmax Applied Separately**: Probabilities computed via `tf.nn.softmax()` for display only
3. **Layer Verification**: Validates that `last_conv_layer_name` exists before processing
4. **Normalized Heatmaps**: Output in [0, 1] range for consistent visualization
5. **JET Colormap**: Uses OpenCV's JET colormap for medical imaging convention

### Evaluation Utilities Features

1. **Comprehensive Metrics**: Precision, recall, F1-score, ROC-AUC all computed
2. **Clinically Relevant**: Focuses on PNEUMONIA recall (false negatives are critical)
3. **Reproducibility**: All outputs saved as files (JSON + PNG)
4. **Error Analysis**: TP/TN/FP/FN breakdown for detailed investigation
5. **Batch Predictions**: Efficient evaluation on full datasets

## 🎯 Integration with Plan.md

### Alignment with Specifications

✓ **B2**: `gradcam.py` with all required functions  
✓ **B3**: `eval_utils.py` with comprehensive metrics  
✓ **Type Hints**: All functions properly annotated  
✓ **Docstrings**: Google-style documentation throughout  
✓ **Error Handling**: Validates inputs and provides clear error messages  
✓ **Dependencies**: TensorFlow, NumPy, OpenCV, scikit-learn correctly imported  

### Changes from Provided Code

1. **Added `plot_gradcam_grid()`**: Grid visualization for multiple images (from plan.md)
2. **Enhanced Documentation**: Comprehensive docstrings with examples
3. **Type Hints**: Full type annotations for better IDE support
4. **Error Handling**: Input validation and informative error messages
5. **Path Objects**: Uses `pathlib.Path` for cross-platform compatibility
6. **Additional Functions**: 
   - `print_misclassification_summary()` for error analysis
   - Enhanced `full_report()` with more visualizations

### Why These Changes?

- **Modularity**: Functions are reusable across notebooks
- **Debugging**: Better error messages speed up development
- **Reproducibility**: All outputs saved automatically
- **Code Quality**: Follows project conventions from plan.md
- **Completeness**: Implements full specification from Phase 2

## 📝 Next Steps (Phase 3 - Integration)

Now that both Dev A and Dev B modules are complete:

1. **Train Model**: Use Dev A's pipeline to train VGG16
2. **Evaluate**: Use Dev B's `full_report()` for comprehensive metrics
3. **Analyze**: Use Grad-CAM to understand model decisions (TP/TN/FP/FN)
4. **Compare**: Evaluate baseline CNN and VGG19 with same utilities
5. **Demo**: Create interactive Gradio app using both modules

## 🐛 Troubleshooting

### "Layer 'block5_conv3' not found"
- Model architecture mismatch
- Verify model was built with `build_vgg16()` from `src/model.py`

### "Heatmap is blank/all zeros"
- Model outputs softmax instead of logits
- Rebuild model ensuring NO activation on final Dense layer

### "Probabilities don't sum to 1"
- Normal: raw logits don't sum to 1
- Softmax is applied separately in `make_gradcam_heatmap()`

### "Generator shuffle warning"
- `predict_probs()` requires `shuffle=False`
- Reset generator before evaluation

## 📚 References

- **Grad-CAM Paper**: "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization" (Selvaraju et al., 2017)
- **VGG16**: "Very Deep Convolutional Networks for Large-Scale Image Recognition" (Simonyan & Zisserman, 2014)
- **Project Plan**: `PLAN_Pneumonia_GradCAM_A_B.md`
