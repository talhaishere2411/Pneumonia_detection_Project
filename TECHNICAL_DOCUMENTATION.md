# Technical Documentation - Complete Function Reference

## Overview

This document provides comprehensive technical documentation for all functions in the Pneumonia Detection project, including the technologies, libraries, and algorithms used.

---

## Table of Contents

1. [config.py - Configuration](#1-configpy---configuration)
2. [data.py - Data Loading](#2-datapy---data-loading)
3. [model.py - Neural Network](#3-modelpy---neural-network)
4. [gradcam.py - Explainability](#4-gradcampy---explainability)
5. [eval_utils.py - Evaluation](#5-eval_utilspy---evaluation)
6. [Technology Stack Summary](#technology-stack-summary)

---

# 1. config.py - Configuration

## Purpose
Centralized configuration management for the entire project.

## Libraries Used

### `pathlib.Path` (Python Standard Library)
- **What**: Cross-platform file path handling
- **Why**: Works on Windows (`\`), Linux/Mac (`/`) without modification
- **Used for**: Auto-detecting dataset locations

---

## Functions

### `find_data_root() -> Path`

**Purpose**: Automatically locate the dataset across different environments

**Technology Stack**:
- `pathlib.Path` - File system navigation
- `Path.exists()` - Check directory existence
- `Path.rglob()` - Recursive glob pattern matching

**How It Works**:
```
1. Check /kaggle/input (Kaggle notebooks)
2. Check /content (Google Colab)
3. Check data/ (local development)
4. For each location, search for:
   - train/ directory with NORMAL/ subdirectory
   - test/ directory in same parent
5. Return parent directory if structure valid
6. Raise error if not found
```

**Algorithm**:
```python
for base_path in [kaggle, colab, local]:
    if base_path exists:
        search recursively for "train" directories:
            if train/NORMAL exists AND parent/test exists:
                return parent directory
raise FileNotFoundError
```

**Use Case**: Seamless switching between Kaggle, Colab, and local environments

---

## Constants

### Image Configuration
- **`IMG_SIZE = (224, 224)`**: VGG16 input requirement
- **`INPUT_SHAPE = (224, 224, 3)`**: Height × Width × Channels (RGB)

### Training Configuration
- **`BATCH_SIZE = 16`**: Images processed simultaneously (memory vs speed trade-off)
- **`SEED = 42`**: Random seed for reproducibility
- **`VAL_SPLIT = 0.15`**: 15% of training data for validation

### Model Configuration
- **`CLASS_NAMES = ["NORMAL", "PNEUMONIA"]`**: Binary classification
- **`LAST_CONV_LAYER = "block5_conv3"`**: VGG16's final conv layer (for Grad-CAM)
- **`MODEL_FILE = "vgg16_pneumonia.keras"`**: Default save path

---

# 2. data.py - Data Loading

## Purpose
Load and preprocess chest X-ray images with data augmentation.

## Libraries Used

### 1. **NumPy** (`numpy as np`)
- **What**: Numerical computing library
- **Used for**: Array operations, mathematical computations
- **Specific use**: Creating class weight arrays

### 2. **scikit-learn** (`sklearn.utils.class_weight`)
- **What**: Machine learning utilities
- **Used for**: Handling imbalanced datasets
- **Algorithm**: Inversely proportional to class frequencies

### 3. **Keras ImageDataGenerator** (`tensorflow.keras.preprocessing.image`)
- **What**: Image data pipeline with augmentation
- **Used for**: Loading images + applying transformations

### 4. **VGG16 preprocess_input** (`tensorflow.keras.applications.vgg16`)
- **What**: ImageNet preprocessing function
- **Used for**: Normalizing images for VGG16 input

---

## Functions

### `get_generators(root=None) -> Tuple[Generator, Generator, Generator]`

**Purpose**: Create data generators for training, validation, and testing

**Parameters**:
- `root` (optional): Dataset root directory (auto-detected if None)

**Returns**: `(train_generator, val_generator, test_generator)`

**Technology Stack**:

#### ImageDataGenerator Configuration

**Training Generator (with augmentation)**:
```python
preprocessing_function=preprocess_input  # ImageNet normalization
validation_split=0.15                    # 85% train, 15% val
shear_range=0.2                          # Slant transformation
zoom_range=0.2                           # Random zoom in/out
horizontal_flip=True                     # Mirror images
```

**Augmentation Explained**:
- **Shear**: Slants the image (simulates viewing angle changes)
- **Zoom**: Random 0.8× to 1.2× scaling (simulates distance variation)
- **Horizontal flip**: Mirrors image (lungs are symmetric)
- **Why no vertical flip**: X-rays always oriented same way

**Validation/Test Generators (no augmentation)**:
```python
preprocessing_function=preprocess_input  # Same normalization
shuffle=False                            # Maintain order for evaluation
```

#### ImageNet Preprocessing

**What `preprocess_input` does**:
```python
# NOT simple: pixel / 255
# Instead:
BGR_mean = [103.939, 116.779, 123.68]
preprocessed = image - BGR_mean  # Mean subtraction (BGR order)
```

**Why this specific preprocessing**:
- VGG16 was trained on ImageNet with this exact normalization
- Using different preprocessing breaks transfer learning
- BGR order (not RGB) because OpenCV convention

#### flow_from_directory Explained

**What it does**:
```
data/
├── train/
│   ├── NORMAL/       → Automatically labeled as class 0
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── PNEUMONIA/    → Automatically labeled as class 1
│       ├── img3.jpg
│       └── img4.jpg
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

**Parameters**:
- `target_size=(224, 224)`: Resize all images
- `batch_size=16`: Load 16 images at a time
- `class_mode="categorical"`: One-hot encoding [1,0] or [0,1]
- `classes=["NORMAL", "PNEUMONIA"]`: Force order (0=NORMAL, 1=PNEUMONIA)
- `subset="training"`: Use 85% for training
- `shuffle=True`: Randomize order each epoch

**Memory Efficiency**: Loads images on-demand (not all at once)

---

### `get_class_weights(train_gen) -> Dict[int, float]`

**Purpose**: Compute weights to handle class imbalance

**Parameters**:
- `train_gen`: Training data generator

**Returns**: `{0: weight_normal, 1: weight_pneumonia}`

**Technology Stack**:

#### scikit-learn compute_class_weight

**Algorithm**:
```python
n_samples = total number of samples
n_classes = 2 (NORMAL, PNEUMONIA)
n_samples_per_class = [count_normal, count_pneumonia]

weight_for_class_i = n_samples / (n_classes × n_samples_per_class[i])
```

**Example**:
```
Total samples: 5216
NORMAL: 1341      → weight = 5216 / (2 × 1341) = 1.945
PNEUMONIA: 3875   → weight = 5216 / (2 × 3875) = 0.673
```

**Purpose**:
- Give more importance to minority class (NORMAL)
- Prevent model from just predicting majority class (PNEUMONIA)
- Loss = original_loss × class_weight[true_class]

**Why It Matters**:
Without class weights:
- Model learns "always predict PNEUMONIA" → 74% accuracy
- But misses all NORMAL cases (bad for diagnosis)

With class weights:
- Model learns both classes equally
- Balanced performance

---

# 3. model.py - Neural Network

## Purpose
Define and compile VGG16-based transfer learning model.

## Libraries Used

### 1. **TensorFlow/Keras** (`tensorflow as tf`)
- **What**: Deep learning framework
- **Used for**: Building, training, evaluating neural networks

### 2. **Keras Layers** (`tensorflow.keras.layers`)
- **What**: Building blocks of neural networks
- **Used for**: Flatten, Dropout, Dense layers

### 3. **Keras Models** (`tensorflow.keras.models.Model`)
- **What**: Connects layers into a network
- **Used for**: Defining model architecture

### 4. **VGG16** (`tensorflow.keras.applications.VGG16`)
- **What**: Pre-trained convolutional neural network
- **Used for**: Feature extraction via transfer learning

---

## Functions

### `build_vgg16(dropout=0.5) -> Model`

**Purpose**: Build VGG16 model with custom classification head

**Parameters**:
- `dropout` (float): Dropout rate for regularization (default: 0.5)

**Returns**: Uncompiled Keras Model

**Technology Stack**:

#### VGG16 Architecture

**Original VGG16**:
```
Input (224×224×3)
    ↓
Block 1: Conv64 → Conv64 → MaxPool
Block 2: Conv128 → Conv128 → MaxPool
Block 3: Conv256 → Conv256 → Conv256 → MaxPool
Block 4: Conv512 → Conv512 → Conv512 → MaxPool
Block 5: Conv512 → Conv512 → Conv512 → MaxPool  ← We use this!
    ↓
Flatten
    ↓
Dense(4096) → Dense(4096) → Dense(1000)  ← We replace this!
```

**Our Modified Architecture**:
```
VGG16 Base (frozen)
    ↓
Flatten           # Convert (7,7,512) → (25088,)
    ↓
Dropout(0.5)      # Randomly drop 50% of neurons
    ↓
Dense(2)          # 2 outputs (NORMAL, PNEUMONIA)
    ↓
Logits (no softmax!)  # Raw scores, not probabilities
```

#### VGG16 Parameters Explained

**`weights="imagenet"`**:
- Load pre-trained weights from ImageNet dataset
- ImageNet: 1.4 million images, 1000 categories
- VGG16 learned to detect edges, textures, shapes, objects

**`include_top=False`**:
- Remove original classification layers (Dense 4096 → 4096 → 1000)
- Keep only convolutional feature extractor
- Output: (7, 7, 512) feature maps

**`input_shape=(224, 224, 3)`**:
- Fixed input size required by VGG16
- 224×224 pixels, 3 channels (RGB)

#### Transfer Learning Concept

**What is Transfer Learning?**
```
ImageNet (millions of images) → VGG16 learns general features
    ↓
Transfer knowledge to medical imaging
    ↓
Fine-tune for pneumonia detection (5000 images)
```

**Benefits**:
- Need less data (5000 vs millions)
- Faster training (pre-trained features)
- Better accuracy (leverages general vision knowledge)

#### Freezing Base Model

**`base_model.trainable = False`**:
- Lock VGG16 weights (don't update during training)
- Only train the new classification head
- **Why**: Preserve learned ImageNet features

**Parameter Counts**:
- Total: 14.7M parameters
- Trainable (head only): 50K parameters
- Frozen (base): 14.65M parameters

#### Classification Head

**Flatten Layer**:
```python
Input: (None, 7, 7, 512)  # Batch × Height × Width × Channels
Output: (None, 25088)      # Batch × Flattened
# 7 × 7 × 512 = 25,088 features
```

**Dropout Layer** (dropout=0.5):
- **What**: Randomly set 50% of inputs to zero during training
- **Why**: Prevents overfitting (model memorizing training data)
- **How**: Forces network to learn robust features
- **During testing**: All neurons active (no dropout)

**Dense Output Layer** (2 units, no activation):
```python
Dense(2, activation=None, name="logits")
```
- **Why no softmax?**
  1. Better numerical stability
  2. Grad-CAM needs gradients (softmax saturates)
  3. Loss function handles it with `from_logits=True`

**Logits vs Probabilities**:
```
Logits: [-2.3, 1.7]           # Raw scores (can be any value)
    ↓ softmax
Probabilities: [0.08, 0.92]   # Sum to 1.0
```

#### Model Statistics

**Total Parameters**: 14,764,866
- VGG16 base: 14,714,688 (frozen)
- Flatten: 0 (just reshaping)
- Dropout: 0 (just masking)
- Dense: 50,178 (trainable)

**Memory**: ~60 MB for model weights

---

### `compile_model(model, lr=1e-4) -> Model`

**Purpose**: Configure model for training

**Parameters**:
- `model`: Keras model to compile
- `lr` (float): Learning rate (default: 0.0001)

**Returns**: Compiled model

**Technology Stack**:

#### Adam Optimizer

**What is Adam?**
- Adaptive Moment Estimation
- Combines best of AdaGrad + RMSProp
- Adapts learning rate per parameter

**How it works**:
```python
# Simplified algorithm:
for each parameter:
    gradient = calculate_gradient(parameter)
    momentum = 0.9 × old_momentum + 0.1 × gradient
    velocity = 0.999 × old_velocity + 0.001 × gradient²
    update = momentum / sqrt(velocity + ε)
    parameter -= learning_rate × update
```

**Learning Rate (lr=1e-4)**:
- 1e-4 = 0.0001 = small steps
- Too high: Overshoots minimum (unstable)
- Too low: Slow convergence
- 1e-4 is good default for fine-tuning

#### CategoricalCrossentropy Loss

**What is it?**
- Measures difference between predicted and true distribution
- Used for multi-class classification

**Formula**:
```
Loss = -Σ(y_true × log(y_pred))

Example:
True: [0, 1]  (PNEUMONIA)
Pred: [0.2, 0.8]
Loss = -(0×log(0.2) + 1×log(0.8)) = -log(0.8) = 0.223

If prediction was [0.01, 0.99]:
Loss = -log(0.99) = 0.010  # Better!
```

**`from_logits=True`**:
- Model outputs raw scores (logits)
- Loss function applies softmax internally
- **Why**: Numerically stable (avoids log(0) errors)

```python
# Without from_logits=True:
logits → softmax → log → loss  # Two operations

# With from_logits=True:
logits → log_softmax → loss    # Combined (stable)
```

#### Accuracy Metric

**How it's calculated**:
```python
predicted_class = argmax(model_output)
true_class = argmax(true_label)
accuracy = mean(predicted_class == true_class)
```

**Example**:
```
Batch of 32 images:
- 28 correct predictions
- 4 incorrect predictions
Batch accuracy = 28/32 = 87.5%
```

---

### `unfreeze_block5(model, lr=1e-5) -> Model`

**Purpose**: Enable fine-tuning of Block 5 for Stage 2 training

**Parameters**:
- `model`: Compiled model
- `lr` (float): Lower learning rate (default: 0.00001)

**Returns**: Recompiled model

**Technology Stack**:

#### Two-Stage Training Strategy

**Stage 1** (frozen base):
```
VGG16 Base [FROZEN] → Head [TRAINING]
Learning rate: 1e-4
Epochs: ~10
Goal: Train classification head
```

**Stage 2** (Block 5 unfrozen):
```
Blocks 1-4 [FROZEN] → Block 5 [TRAINING] → Head [TRAINING]
Learning rate: 1e-5 (10× lower!)
Epochs: ~5
Goal: Fine-tune deepest features
```

**Why Two Stages?**

1. **Stage 1**: Head needs to learn from scratch
   - Random initialization
   - Needs larger learning rate
   - Base features are good enough initially

2. **Stage 2**: Fine-tune for medical images
   - Head is now trained
   - Adjust deepest conv layers for X-rays
   - Low LR prevents breaking learned features

#### Selective Unfreezing

**Algorithm**:
```python
for each layer in model:
    if layer.name starts with "block5":
        layer.trainable = True     # Unfreeze
    elif layer.name == "logits":
        layer.trainable = True     # Keep trainable
    else:
        layer.trainable = False    # Keep frozen
```

**Trainable Parameters**:
- Stage 1: 50K parameters (head only)
- Stage 2: 7.1M parameters (Block 5 + head)

**Why Block 5?**
- Deepest layer = most abstract features
- Learns high-level patterns (organ structures)
- Earlier blocks = basic edges/textures (don't need adjustment)

#### Lower Learning Rate (1e-5)

**Why 10× lower?**
```
High LR (1e-4): [=====>] Fast but might overshoot
Low LR (1e-5):  [=>] Slow but precise adjustments
```

**Catastrophic Forgetting**:
- High LR during fine-tuning → destroys learned features
- Low LR → gentle adjustments, preserves knowledge

---

# 4. gradcam.py - Explainability

## Purpose
Visualize what the model is "looking at" using Gradient-weighted Class Activation Mapping.

## Libraries Used

### 1. **NumPy** (`numpy as np`)
- **What**: Numerical array operations
- **Used for**: Array manipulation, type conversions

### 2. **OpenCV** (`cv2`)
- **What**: Computer vision library
- **Used for**: Image resizing, color mapping, blending

### 3. **TensorFlow** (`tensorflow as tf`)
- **What**: Deep learning framework
- **Used for**: Gradient computation, model inference

### 4. **Matplotlib** (`matplotlib.pyplot as plt`)
- **What**: Plotting library
- **Used for**: Creating visualizations, saving figures

---

## Functions

### `load_and_preprocess_image(image_path) -> Tuple[Image, ndarray, ndarray]`

**Purpose**: Load and prepare a single image for model input

**Parameters**:
- `image_path` (str): Path to image file

**Returns**: 
- `img`: PIL Image object
- `img_array`: NumPy array (1, 224, 224, 3)
- `processed_img`: Preprocessed for VGG16

**Technology Stack**:

#### Keras Image Loading

**`tf.keras.utils.load_img()`**:
```python
# What it does:
1. Open image file (JPEG, PNG, etc.)
2. Resize to target_size=(224, 224)
3. Convert to RGB (if grayscale, duplicate to 3 channels)
4. Return PIL Image object
```

**`tf.keras.utils.img_to_array()`**:
```python
# Convert PIL Image → NumPy array
Input: PIL Image (224, 224, RGB)
Output: NumPy array (224, 224, 3) dtype=float32
```

#### Batch Dimension

**`np.expand_dims(img_array, axis=0)`**:
```python
# Add batch dimension
Input shape: (224, 224, 3)
Output shape: (1, 224, 224, 3)
# First dimension = batch size (1 image)
```

**Why needed?**
- Models expect batches: (batch_size, height, width, channels)
- Single image = batch of 1

#### VGG16 Preprocessing

**`preprocess_input(img_array.copy())`**:
```python
# ImageNet normalization
.copy() ensures original array unchanged
Subtracts mean: [103.939, 116.779, 123.68]
BGR order (OpenCV convention)
```

---

### `make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index) -> Tuple[ndarray, ndarray]`

**Purpose**: Generate Grad-CAM heatmap showing model focus areas

**Parameters**:
- `img_array`: Preprocessed image (1, 224, 224, 3)
- `model`: Trained Keras model
- `last_conv_layer_name`: Conv layer name (default: "block5_conv3")
- `pred_index`: Target class (None = predicted class)

**Returns**:
- `heatmap`: Normalized heatmap (H, W) in [0, 1]
- `probabilities`: Softmax probabilities (2,)

**Technology Stack**:

#### Grad-CAM Algorithm

**Step 1: Create Gradient Model**
```python
grad_model = Model(
    inputs=model.inputs,
    outputs=[conv_layer.output, model.output]
)
# Outputs both:
# 1. Last conv layer activations: (1, 14, 14, 512)
# 2. Final predictions: (1, 2)
```

**Step 2: Forward Pass with Gradient Recording**
```python
with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(img_array)
    # GradientTape records all operations
```

**What is GradientTape?**
- Records operations during forward pass
- Enables automatic differentiation (backprop)
- Computes gradients efficiently

**Step 3: Calculate Gradients**
```python
# Get score for target class
class_channel = predictions[:, pred_index]  # Single value (logit)

# Compute gradient: ∂(class_score)/∂(conv_output)
grads = tape.gradient(class_channel, conv_outputs)
# Shape: (1, 14, 14, 512)
```

**What does gradient mean?**
- How much each conv feature affects the prediction
- Large gradient = important for prediction
- Positive gradient = increases class score
- Negative gradient = decreases class score

**Step 4: Global Average Pooling**
```python
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
# Average over batch, height, width
# Shape: (512,) - one importance weight per channel
```

**Why average?**
- Each conv filter (512 total) has a 14×14 map
- Average gives overall importance of that filter
- Results in channel-wise importance weights

**Step 5: Weighted Combination**
```python
conv_outputs = conv_outputs[0]  # Remove batch dimension: (14, 14, 512)
heatmap = tf.reduce_sum(conv_outputs × pooled_grads, axis=-1)
# Multiply each channel by its importance
# Sum across channels
# Shape: (14, 14)
```

**Intuition**:
```
For each 14×14 location:
    weighted_activation = Σ(channel_activation × channel_importance)
    
Example:
Location (5, 7):
    Filter 1: activation=0.8, importance=0.3 → contributes 0.24
    Filter 2: activation=0.2, importance=0.9 → contributes 0.18
    ...
    Sum all 512 filters → heatmap value at (5, 7)
```

**Step 6: ReLU Activation**
```python
heatmap = tf.maximum(heatmap, 0)
# Keep only positive values
```

**Why ReLU?**
- Focus on features that INCREASE prediction
- Negative values = features that decrease score (ignore)
- Medical interpretation: "What supports pneumonia?"

**Step 7: Normalization**
```python
heatmap /= (tf.reduce_max(heatmap) + epsilon)
# Scale to [0, 1] range
# epsilon prevents division by zero
```

#### Mathematical Foundation

**Grad-CAM Formula**:
```
L^c_Grad-CAM = ReLU(Σ_k(α^c_k × A^k))

where:
α^c_k = (1/Z) × Σ_i Σ_j (∂y^c/∂A^k_ij)  # Importance weights
A^k = Activation map of kth filter
y^c = Score for class c
Z = Normalization factor
```

**In plain English**:
1. Take last conv layer activations (feature maps)
2. Weight each by its gradient (importance)
3. Sum weighted features
4. Apply ReLU (keep positive)
5. Normalize to [0, 1]

---

### `overlay_heatmap(img, heatmap, alpha=0.4) -> ndarray`

**Purpose**: Blend heatmap with original image for visualization

**Parameters**:
- `img`: Original image (uint8, RGB or grayscale)
- `heatmap`: Grad-CAM heatmap (H, W) in [0, 1]
- `alpha`: Blend factor (default: 0.4)

**Returns**: Overlayed image (uint8, RGB)

**Technology Stack**:

#### Image Format Handling

**Grayscale to RGB Conversion**:
```python
if img.ndim == 2:                    # Grayscale (H, W)
    img = np.stack([img] * 3, axis=-1)  # → (H, W, 3)
# Duplicate grayscale to R, G, B channels
```

#### OpenCV Operations

**1. Resize Heatmap**:
```python
heatmap_resized = cv2.resize(
    heatmap,
    (img.shape[1], img.shape[0])  # (width, height)
)
# Match heatmap size to original image
# cv2.resize uses (width, height) order!
```

**2. Apply Colormap**:
```python
heatmap_uint8 = np.uint8(255 * heatmap)  # [0,1] → [0,255]
heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
```

**JET Colormap**:
```
Value: 0.0 → 0.25 → 0.5 → 0.75 → 1.0
Color: Blue → Cyan → Green → Yellow → Red
Use: Medical convention (cold→hot)
```

**3. Color Space Conversion**:
```python
heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
# OpenCV uses BGR (Blue-Green-Red)
# Matplotlib/Display uses RGB (Red-Green-Blue)
```

**4. Alpha Blending**:
```python
overlay = cv2.addWeighted(
    img,           # Source 1
    1 - alpha,     # Weight 1 (0.6)
    heatmap,       # Source 2
    alpha,         # Weight 2 (0.4)
    0              # Gamma (brightness offset)
)
# Result = (0.6 × img) + (0.4 × heatmap)
```

**Alpha Values**:
- `alpha=0.0`: Only original image (no heatmap)
- `alpha=0.4`: Balanced overlay (recommended)
- `alpha=1.0`: Only heatmap (no original)

#### Blending Formula

**Mathematical operation**:
```
Output_pixel = (1 - α) × Original_pixel + α × Heatmap_pixel

Example with α=0.4:
Original RGB: (100, 120, 110)
Heatmap RGB:  (255, 0, 0) - red hot spot
Output: 0.6×(100,120,110) + 0.4×(255,0,0) = (162, 72, 66)
```

---

### `plot_gradcam_grid(model, paths, true_labels, ncols, save_path) -> None`

**Purpose**: Create multi-image visualization grid with Grad-CAM overlays

**Parameters**:
- `model`: Trained Keras model
- `paths`: List of image file paths
- `true_labels`: Ground truth labels (optional)
- `ncols`: Number of columns (default: 4)
- `save_path`: Save location (optional)

**Returns**: None (displays and optionally saves figure)

**Technology Stack**:

#### Matplotlib Grid Layout

**`plt.subplots(nrows, ncols*2)`**:
```python
# Example: 8 images, ncols=4
nrows = ceil(8 / 4) = 2
total_cols = 4 × 2 = 8  # (original + overlay) × 4
# Creates 2×8 grid of subplots
```

**Grid Structure**:
```
[Original 1] [Overlay 1] [Original 2] [Overlay 2]
[Original 3] [Overlay 3] [Original 4] [Overlay 4]
```

#### NumPy Operations

**`np.atleast_2d(axes)`**:
```python
# Ensures axes is 2D array
# Single row → converts to [[ax1, ax2, ...]]
# Multiple rows → already 2D
```

**`divmod(i, ncols)`**:
```python
# Divide and get quotient + remainder
i=5, ncols=4 → divmod(5, 4) = (1, 1)
# row=1, col=1 (second row, second column)
```

#### Image Processing Loop

**For each image**:
```python
1. Load and preprocess image
2. Generate Grad-CAM heatmap
3. Get prediction and confidence
4. Plot original in left subplot
5. Plot overlay in right subplot
6. Add title with prediction info
```

**Title Format**:
```
With true labels:
"true: NORMAL
pred: PNEUMONIA (0.87)"

Without true labels:
"pred: PNEUMONIA (0.87)"
```

#### Matplotlib Styling

**`ax.axis("off")`**: Remove x/y axis ticks and labels
**`ax.imshow()`**: Display image array
**`ax.set_title()`**: Add text above image
**`plt.tight_layout()`**: Auto-adjust spacing
**`plt.savefig(dpi=150)`**: High-resolution save

---

# 5. eval_utils.py - Evaluation

## Purpose
Comprehensive model evaluation with metrics, visualizations, and error analysis.

## Libraries Used

### 1. **NumPy** (`numpy as np`)
- **What**: Numerical computing
- **Used for**: Array operations, argmax, boolean indexing

### 2. **TensorFlow** (`tensorflow as tf`)
- **What**: Deep learning framework
- **Used for**: Softmax computation, batch prediction

### 3. **Matplotlib** (`matplotlib.pyplot as plt`)
- **What**: Visualization library
- **Used for**: Plotting confusion matrices, ROC curves, training history

### 4. **scikit-learn** (`sklearn.metrics`)
- **What**: Machine learning metrics
- **Used for**: Precision, recall, F1, ROC-AUC, confusion matrix

### 5. **JSON** (`json`)
- **What**: Data serialization
- **Used for**: Saving metrics to file

### 6. **Pathlib** (`pathlib.Path`)
- **What**: File system operations
- **Used for**: Creating directories, path manipulation

---

## Functions

### `get_model_prediction(model, processed_img) -> Dict[str, Any]`

**Purpose**: Single image inference with detailed results

**Parameters**:
- `model`: Trained Keras model (logits output)
- `processed_img`: Preprocessed image (1, H, W, 3)

**Returns**: Dictionary with prediction details

**Technology Stack**:

#### Model Prediction

**`model.predict(processed_img, verbose=0)`**:
```python
# Returns logits (raw scores)
Input: (1, 224, 224, 3)
Output: (1, 2) - two logits (one per class)

Example:
logits = [[-2.3, 1.7]]
# Negative for NORMAL, positive for PNEUMONIA
```

#### Softmax Conversion

**`tf.nn.softmax(predictions[0]).numpy()`**:
```python
# Convert logits to probabilities
Logits: [-2.3, 1.7]
    ↓ exp
    [0.100, 5.474]
    ↓ normalize
Probabilities: [0.018, 0.982]
# Sum to 1.0
```

**Softmax Formula**:
```
softmax(x_i) = exp(x_i) / Σ_j exp(x_j)

Example:
exp(-2.3) / (exp(-2.3) + exp(1.7)) = 0.018
exp(1.7) / (exp(-2.3) + exp(1.7)) = 0.982
```

#### Output Dictionary

```python
{
    "class_idx": 1,                    # 0=NORMAL, 1=PNEUMONIA
    "class_name": "PNEUMONIA",
    "confidence": 0.982,               # Max probability
    "probabilities": [0.018, 0.982],  # All class probs
    "raw_predictions": [-2.3, 1.7]    # Original logits
}
```

---

### `predict_probs(model, generator) -> Tuple[ndarray, ndarray]`

**Purpose**: Batch prediction on entire dataset

**Parameters**:
- `model`: Trained Keras model
- `generator`: ImageDataGenerator (MUST have shuffle=False)

**Returns**: `(y_true, probs)`

**Technology Stack**:

#### Generator Reset

**`generator.reset()`**:
- Resets internal counter to start
- Ensures full dataset coverage
- Prevents partial batch issues

#### Batch Prediction

**`model.predict(generator, verbose=0)`**:
```python
# Automatically:
1. Loop through all batches
2. Predict each batch
3. Concatenate results
4. Return full predictions array

Example with 624 images, batch_size=16:
- 39 full batches of 16 images
- Processes all automatically
- Returns: (624, 2) array of logits
```

#### Why shuffle=False?

**Order matters for evaluation**:
```python
generator.filepaths[i] corresponds to y_true[i] and probs[i]

With shuffle=True:
- Order randomized each epoch
- Can't match predictions to filenames
- Error analysis impossible

With shuffle=False:
- Consistent order
- Can trace back to specific images
- Enable misclassification analysis
```

#### Softmax Application

**`tf.nn.softmax(logits, axis=1).numpy()`**:
```python
# axis=1: Apply softmax across classes (not across samples)
Input: (624, 2) - 624 samples, 2 logits each
Output: (624, 2) - 624 samples, 2 probabilities each

Row sums to 1.0:
[0.2, 0.8] ✓
[0.95, 0.05] ✓
```

---

### `full_report(y_true, probs, tag, out_dir) -> Dict[str, Any]`

**Purpose**: Generate comprehensive evaluation report with metrics and visualizations

**Parameters**:
- `y_true`: True labels (N,)
- `probs`: Predicted probabilities (N, 2)
- `tag`: Report identifier
- `out_dir`: Output directory (default: "results")

**Returns**: Dictionary with all metrics

**Technology Stack**:

#### Classification Report

**`classification_report(y_true, y_pred, target_names, output_dict=True)`**:

**Metrics Computed**:

1. **Precision** = TP / (TP + FP)
   - "Of all PNEUMONIA predictions, how many were correct?"
   - High precision = few false alarms

2. **Recall** (Sensitivity) = TP / (TP + FN)
   - "Of all actual PNEUMONIA cases, how many did we catch?"
   - High recall = few missed cases
   - **Most critical for medical diagnosis!**

3. **F1-Score** = 2 × (Precision × Recall) / (Precision + Recall)
   - Harmonic mean of precision and recall
   - Balanced metric

4. **Support** = Number of samples in each class

**Example Output**:
```python
{
    "NORMAL": {
        "precision": 0.92,
        "recall": 0.88,
        "f1-score": 0.90,
        "support": 234
    },
    "PNEUMONIA": {
        "precision": 0.95,
        "recall": 0.97,
        "f1-score": 0.96,
        "support": 390
    },
    "accuracy": 0.94,
    "macro avg": {...},
    "weighted avg": {...}
}
```

#### ROC-AUC Score

**`roc_auc_score(y_true, probs[:, 1])`**:

**What is ROC-AUC?**
- ROC: Receiver Operating Characteristic curve
- AUC: Area Under the Curve
- Range: 0.5 (random) to 1.0 (perfect)

**Interpretation**:
- 0.50: Random guessing
- 0.70-0.80: Acceptable
- 0.80-0.90: Excellent
- 0.90-1.00: Outstanding

**Why use class index 1?**
```python
probs[:, 1]  # Probability of PNEUMONIA (positive class)
# ROC-AUC measures ability to rank positive class higher
```

#### Confusion Matrix

**`confusion_matrix(y_true, y_pred)`**:

**Structure**:
```
                Predicted
              NORMAL  PNEUMONIA
Actual NORMAL   TN      FP
     PNEUMONIA  FN      TP

Example:
[[210  24]    TN=210, FP=24
 [ 12 378]]   FN=12, TP=378
```

**ConfusionMatrixDisplay**:
- Visualizes matrix as heatmap
- Adds labels and colorbar
- Easy interpretation

#### ROC Curve

**`roc_curve(y_true, probs[:, 1])`**:

**What it shows**:
```
For different classification thresholds:
- X-axis: False Positive Rate = FP / (FP + TN)
- Y-axis: True Positive Rate = TP / (TP + FN)
```

**How it works**:
```python
Threshold = 0.9 → Very conservative
- Few predictions above threshold
- Low FPR (few false alarms)
- Low TPR (miss many cases)

Threshold = 0.5 → Balanced
- Moderate FPR and TPR

Threshold = 0.1 → Very liberal
- Many predictions above threshold
- High FPR (many false alarms)
- High TPR (catch most cases)
```

**Ideal ROC**:
```
TPR
 1.0|     ___
    |    |
    |    |
    |    |
 0.0|____|________FPR
    0.0        1.0
    
Perfect: Hugs top-left corner (100% TPR, 0% FPR)
Random: Diagonal line (AUC=0.5)
```

#### File Outputs

**JSON metrics** (`{tag}_metrics.json`):
```json
{
  "NORMAL": {"precision": 0.92, "recall": 0.88, ...},
  "PNEUMONIA": {"precision": 0.95, "recall": 0.97, ...},
  "accuracy": 0.94,
  "roc_auc": 0.96
}
```

**PNG visualizations**:
- `{tag}_confusion.png`: Confusion matrix heatmap
- `{tag}_roc.png`: ROC curve with AUC

---

### `plot_history(history, tag, out_dir) -> None`

**Purpose**: Visualize training progress over epochs

**Parameters**:
- `history`: Keras History object from model.fit()
- `tag`: Plot identifier
- `out_dir`: Output directory

**Returns**: None (displays and saves figure)

**Technology Stack**:

#### History Object

**What it contains**:
```python
history.history = {
    "loss": [0.45, 0.32, 0.24, ...],          # Training loss
    "val_loss": [0.52, 0.38, 0.31, ...],      # Validation loss
    "accuracy": [0.78, 0.86, 0.91, ...],      # Training accuracy
    "val_accuracy": [0.75, 0.83, 0.88, ...]   # Validation accuracy
}
# One value per epoch
```

#### Subplot Creation

**`plt.subplots(1, 2, figsize=(12, 4))`**:
```
Creates 1 row, 2 columns:
┌──────────────┬──────────────┐
│  Loss Plot   │ Accuracy Plot│
└──────────────┴──────────────┘
```

#### Loss Plot

**What it shows**:
```python
Epoch 0: Loss=0.45 → Random initialization
Epoch 5: Loss=0.24 → Learning patterns
Epoch 10: Loss=0.12 → Converged

Good pattern:
- Training loss decreases
- Validation loss decreases
- Small gap between them

Bad pattern (overfitting):
- Training loss → 0
- Validation loss increases
- Large gap (model memorizing)
```

#### Accuracy Plot

**What it shows**:
```python
Epoch 0: Acc=0.78 → Better than random (0.5)
Epoch 5: Acc=0.86 → Improving
Epoch 10: Acc=0.91 → Near plateau

Ideal pattern:
- Both training and validation increase
- Validation stays close to training
- Smooth curves (not erratic)
```

#### Matplotlib Styling

**Grid**: `plt.grid(alpha=0.3)` - Subtle gridlines
**Legend**: `plt.legend()` - Show train/val labels
**Labels**: `set_xlabel`, `set_ylabel` - Axis descriptions
**Title**: `set_title(fontweight="bold")` - Plot header

---

### `misclassified_indices(y_true, probs) -> Dict[str, ndarray]`

**Purpose**: Identify True Positives, True Negatives, False Positives, False Negatives

**Parameters**:
- `y_true`: True labels (N,)
- `probs`: Predicted probabilities (N, 2)

**Returns**: Dictionary with indices for each category

**Technology Stack**:

#### NumPy Boolean Indexing

**`np.where(condition)`**:
```python
# Find indices where condition is True
y_true = [0, 1, 1, 0, 1]
y_pred = [0, 1, 0, 1, 1]

tp_indices = np.where((y_true == 1) & (y_pred == 1))
# Returns: array([1, 4])
# Indices 1 and 4 have both true=1 and pred=1
```

#### Classification Categories

**Using PNEUMONIA (class 1) as positive**:

1. **True Positive (TP)**:
   - `(y_true == 1) & (y_pred == 1)`
   - Actually PNEUMONIA, predicted PNEUMONIA ✓
   - Best outcome for diagnosis

2. **True Negative (TN)**:
   - `(y_true == 0) & (y_pred == 0)`
   - Actually NORMAL, predicted NORMAL ✓
   - Correct rejection

3. **False Positive (FP)**:
   - `(y_true == 0) & (y_pred == 1)`
   - Actually NORMAL, predicted PNEUMONIA ✗
   - False alarm (not ideal but less critical)

4. **False Negative (FN)**:
   - `(y_true == 1) & (y_pred == 0)`
   - Actually PNEUMONIA, predicted NORMAL ✗
   - Missed diagnosis (CRITICAL! Most dangerous error)

#### Medical Interpretation

**Priority ranking**:
```
1. FN (False Negative) - CRITICAL
   Missing pneumonia → delayed treatment → serious consequences
   
2. TP (True Positive) - GOAL
   Correctly identifying pneumonia → proper treatment
   
3. FP (False Positive) - ACCEPTABLE
   False alarm → extra X-ray/test → better safe than sorry
   
4. TN (True Negative) - GOAL
   Correctly ruling out pneumonia → no unnecessary treatment
```

**Recall vs Precision trade-off**:
```
High threshold (0.9):
- Few FP (high precision)
- More FN (low recall) ← BAD for medical

Low threshold (0.3):
- More FP (lower precision) ← OK, just extra tests
- Few FN (high recall) ← GOOD for medical
```

---

### `print_misclassification_summary(y_true, probs) -> None`

**Purpose**: Print detailed breakdown of classification results

**Parameters**:
- `y_true`: True labels
- `probs`: Predicted probabilities

**Returns**: None (prints to console)

**Technology Stack**:

#### Summary Statistics

**Calculations**:
```python
total = len(y_true)
correct = len(tp) + len(tn)
incorrect = len(fp) + len(fn)

accuracy = correct / total
```

#### Console Output Format

**Example**:
```
============================================================
Classification Summary
============================================================
Total samples: 624
Correct predictions: 588 (94.2%)
Incorrect predictions: 36 (5.8%)

Breakdown:
  True Positives (TP):     378 - Correctly identified PNEUMONIA
  True Negatives (TN):     210 - Correctly identified NORMAL
  False Positives (FP):     24 - NORMAL incorrectly as PNEUMONIA
  False Negatives (FN):     12 - PNEUMONIA missed (critical!)
============================================================
```

#### Clinical Insights

**What to look for**:
1. **High TP count**: Model catches most pneumonia ✓
2. **Low FN count**: Few missed cases ✓
3. **FP count**: Acceptable if moderate
4. **Balanced TN**: Not just predicting all as positive

**Red flags**:
- FN > 5% of pneumonia cases
- TP very low (poor sensitivity)
- FP + FN > TN + TP (more errors than correct)

---

# Technology Stack Summary

## Core Libraries

### 1. **TensorFlow/Keras** (v2.15+)
**What**: Deep learning framework
**Used for**: 
- Model architecture (VGG16)
- Training and optimization
- Gradient computation
- Loss functions
**Why chosen**: Industry standard, extensive documentation

### 2. **NumPy**
**What**: Numerical computing library
**Used for**:
- Array operations
- Mathematical computations
- Data manipulation
**Why chosen**: Foundation of scientific Python

### 3. **scikit-learn**
**What**: Machine learning library
**Used for**:
- Class weight computation
- Evaluation metrics
- Classification reports
**Why chosen**: Comprehensive ML utilities

### 4. **OpenCV** (cv2)
**What**: Computer vision library
**Used for**:
- Image resizing
- Colormap application
- Image blending
**Why chosen**: Fast, optimized operations

### 5. **Matplotlib**
**What**: Plotting library
**Used for**:
- Training curves
- Confusion matrices
- ROC curves
- Grad-CAM grids
**Why chosen**: Publication-quality visualizations

### 6. **Pathlib**
**What**: File system library
**Used for**:
- Cross-platform paths
- Directory operations
**Why chosen**: Modern, Pythonic

---

## Key Algorithms

### 1. **Transfer Learning**
- Start with ImageNet-trained VGG16
- Freeze base, train classification head
- Fine-tune deepest layers

### 2. **Grad-CAM**
- Gradient-weighted Class Activation Mapping
- Visualizes model attention
- Uses gradients of predictions w.r.t. conv features

### 3. **Data Augmentation**
- Shear, zoom, horizontal flip
- Increases training data diversity
- Prevents overfitting

### 4. **Class Weighting**
- Handles imbalanced datasets
- Higher weight for minority class
- Balances training importance

### 5. **Two-Stage Training**
- Stage 1: Train head (frozen base)
- Stage 2: Fine-tune Block 5 (low LR)
- Prevents catastrophic forgetting

---

## Design Decisions

### Why VGG16?
- Simple architecture (easy to understand)
- Proven performance on medical images
- Well-established last conv layer for Grad-CAM

### Why Logits Output?
- Better numerical stability
- Grad-CAM needs raw gradients
- Loss function handles softmax

### Why ImageNet Preprocessing?
- VGG16 trained with this normalization
- Transfer learning requires same preprocessing
- Mean subtraction in BGR order

### Why Grad-CAM?
- Visual explanation (not just numbers)
- Verify model isn't using artifacts
- Clinical validation of attention

### Why These Metrics?
- **Accuracy**: Overall performance
- **Precision/Recall**: Class-specific quality
- **ROC-AUC**: Threshold-independent
- **Confusion Matrix**: Error breakdown

---

## Performance Characteristics

### Model
- **Parameters**: 14.7M total, 50K trainable (stage 1)
- **Memory**: ~60 MB
- **Inference**: ~10ms per image (GPU)
- **Training**: ~5 minutes/epoch (GPU)

### Data Pipeline
- **Batch size**: 16 images
- **Augmentation**: Real-time (not cached)
- **Loading**: On-demand (memory efficient)

### Grad-CAM
- **Resolution**: 14×14 heatmap (upsampled to 224×224)
- **Computation**: ~50ms per image
- **Memory**: Minimal (gradient tape only)

---

## Best Practices Implemented

### Code Quality
- Type hints for all functions
- Comprehensive docstrings
- Error handling and validation
- Consistent naming conventions

### Reproducibility
- Fixed random seed (SEED=42)
- Documented preprocessing
- Saved configurations
- Version-controlled code

### Medical AI Ethics
- High recall prioritized (don't miss disease)
- Explainable predictions (Grad-CAM)
- Error analysis tools
- Disclaimer: Research prototype only

---

## Usage Example

```python
# Complete workflow
from src.config import find_data_root
from src.data import get_generators, get_class_weights
from src.model import build_vgg16, compile_model, unfreeze_block5
from src.eval_utils import predict_probs, full_report
from src.gradcam import plot_gradcam_grid, make_gradcam_heatmap

# 1. Load data
data_root = find_data_root()
train_gen, val_gen, test_gen = get_generators(data_root)
class_weights = get_class_weights(train_gen)

# 2. Build and train (Stage 1)
model = build_vgg16(dropout=0.5)
compile_model(model, lr=1e-4)
history1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    class_weight=class_weights
)

# 3. Fine-tune (Stage 2)
unfreeze_block5(model, lr=1e-5)
history2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=5,
    class_weight=class_weights
)

# 4. Evaluate
y_true, probs = predict_probs(model, test_gen)
metrics = full_report(y_true, probs, "vgg16_final")

# 5. Grad-CAM analysis
from src.eval_utils import misclassified_indices
indices = misclassified_indices(y_true, probs)
fn_paths = [test_gen.filepaths[i] for i in indices['fn'][:8]]
plot_gradcam_grid(model, fn_paths, save_path="results/fn_gradcam.png")
```

---

## References

### Papers
- **VGG16**: Simonyan & Zisserman, "Very Deep Convolutional Networks" (2014)
- **Grad-CAM**: Selvaraju et al., "Grad-CAM: Visual Explanations" (2017)
- **Transfer Learning**: Pan & Yang, "A Survey on Transfer Learning" (2010)

### Libraries
- TensorFlow: https://tensorflow.org
- Keras: https://keras.io
- scikit-learn: https://scikit-learn.org
- OpenCV: https://opencv.org
- NumPy: https://numpy.org
- Matplotlib: https://matplotlib.org

### Dataset
- Chest X-Ray Images (Pneumonia): https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

---

**Last Updated**: 2024
**Version**: 1.0
**Author**: Pneumonia Detection Team
**License**: MIT
