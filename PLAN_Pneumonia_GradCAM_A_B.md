# Pneumonia Detection with VGG16 + Grad-CAM: Two-Person Implementation Plan

**Team:** Person **A** (Model lead) and Person **B** (Evaluation & Explainability lead)
**Platform:** Kaggle (primary), Colab (alternative; see Section 9)
**Dataset:** Kaggle "Chest X-Ray Images (Pneumonia)" by Paul Mooney (5,863 images, Normal / Pneumonia)

---

## 0. How the work is organised (read this first)

### 0.1 Notebooks vs `.py` files

| | Notebooks (`.ipynb`) | Python modules (`.py` in `src/`) |
|---|---|---|
| Purpose | Run experiments, train, plot, show results | Reusable functions shared by both people |
| Who edits | Each person edits their own notebooks | Whoever owns the file (see ownership below) |
| Where they run | Kaggle (GPU) | Imported *into* Kaggle notebooks |
| In Git | Committed after finishing (download from Kaggle) | Committed continuously |

Rule: **if both people need a function, it lives in `src/`, never copy-pasted between notebooks.**

### 0.2 Ownership

| Path | Owner |
|---|---|
| `src/config.py` | A (B reviews; any change needs both to agree) |
| `src/data.py`, `src/model.py` | A |
| `src/gradcam.py`, `src/eval_utils.py` | B |
| `notebooks/01_eda.ipynb`, `03_gradcam_dev.ipynb`, `04_evaluation.ipynb`, `06_gradcam_analysis.ipynb`, `07_demo.ipynb` | B |
| `notebooks/02_train_vgg16.ipynb`, `05_baseline_cnn.ipynb`, `08_vgg19_comparison.ipynb` | A |

### 0.2.1 Balance of work

- **A:** data pipeline, model, training (2 stages), baseline CNN, model comparison (VGG19), README.
- **B:** EDA, evaluation utilities, Grad-CAM (the unique part), error analysis, demo.
- Report writing is split in Section 8.

### 0.3 Sync points (who waits for whom)

| ID | What | From → To | Needed by |
|---|---|---|---|
| **S0** | Repo + `config.py` merged | A → B | Everything |
| **S1** | `data.py` + `model.py` pushed | A → B | B3 (evaluation test), B4 (needs loaders) |
| **S2** | Stage-1 trained model shared | A → B | B5 (real Grad-CAM run) |
| **S3** | Final fine-tuned model shared | A → B | B6, B7 |
| **S4** | `gradcam.py` + `eval_utils.py` pushed | B → A | A6 (reuse for VGG19 evaluation) |

Between sync points, **nobody is blocked**. B's early tasks (EDA, Grad-CAM on a sanity model) do not need A's trained model.

---

## 1. Setup checklist (each person does this BEFORE starting)

### 1.1 Both A and B

- [ ] **GitHub account** (free).
- [ ] **Kaggle account**, then verify your phone number (Profile → Settings → Phone verification). Without this you get **no GPU and no internet** in notebooks.
- [ ] Read the dataset page once: <https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia>
- [ ] Create a new Kaggle notebook (New Notebook), then:
  - Right panel → **Add Input** → search `chest-xray-pneumonia` → add the Paul Mooney dataset.
  - **Settings → Accelerator** → GPU T4 x2 (or P100 if offered).
  - **Settings → Internet** → On (needed for `git clone`).
  - Run `!nvidia-smi` and confirm a GPU shows up.
  - Run the path check:
    ```python
    import os
    for root, dirs, files in os.walk("/kaggle/input"):
        print(root, len(files)); 
        if root.count("/") > 5: break
    ```
    Note exactly where `train/`, `test/`, `val/` are (the dataset sometimes nests an extra `chest_xray/chest_xray/` folder and a `__MACOSX` folder; ignore `__MACOSX`).
- [ ] Pick one way to edit `.py` files (any one works):
  - **Easiest:** open the repo on GitHub and press the `.` key (opens github.dev, a VS Code editor in the browser; commit from there).
  - **Local:** install [Git](https://git-scm.com/) + [VS Code](https://code.visualstudio.com/), then `git clone` the repo.

### 1.2 Only A (repo owner), see Section 2

### 1.3 Only B

- [ ] Accept the GitHub collaborator invitation sent by A (check email).
- [ ] Share your **Kaggle username** with A (needed for sharing the model dataset in Section 7).
- [ ] Clone/open the repo and confirm you can create a branch.

---

## 2. Phase 0: Repo creation (Day 0, A does it, B reviews)

### A0.1: Create the repo

1. GitHub → **New repository** → name: `pneumonia-gradcam`.
2. Make it **Public** (simplest: Kaggle can `git clone` it without tokens; the repo contains code only, no patient data).
3. Tick: *Add a README*, *.gitignore → Python*.
4. **Settings → Collaborators → Add people** → invite B.

### A0.2: Create this folder structure

```
pneumonia-gradcam/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── __init__.py          (empty file)
│   ├── config.py
│   ├── data.py
│   ├── model.py
│   ├── gradcam.py
│   └── eval_utils.py
├── notebooks/
├── results/                 (figures, metrics JSON; committed)
└── models/                  (NOT committed; see .gitignore)
```

Append to `.gitignore`:

```
models/
*.keras
*.h5
.ipynb_checkpoints/
```

`requirements.txt` (Kaggle already has most of these; only needed for local/Colab):

```
tensorflow>=2.15
scikit-learn
matplotlib
opencv-python
pandas
numpy
gradio        # optional, for the demo
```

### A0.3: `src/config.py`, the contract (commit directly to `main`)

```python
"""Shared constants. Changing anything here needs BOTH people to agree."""
from pathlib import Path

IMG_SIZE = (224, 224)
INPUT_SHAPE = (224, 224, 3)
BATCH_SIZE = 32
SEED = 42
VAL_SPLIT = 0.15                      # carved out of train/, test/ stays untouched
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]  # index 0 = NORMAL, index 1 = PNEUMONIA
LAST_CONV_LAYER = "block5_conv3"       # VGG16. (VGG19 would be "block5_conv4")
MODEL_FILE = "vgg16_pneumonia.keras"


def find_data_root() -> Path:
    """Return the folder that contains train/ and test/ (Kaggle, Colab or local)."""
    for base in [Path("/kaggle/input"), Path("/content"), Path("data")]:
        if base.exists():
            for p in base.rglob("train"):
                if p.is_dir() and (p / "NORMAL").is_dir() and (p.parent / "test").is_dir():
                    return p.parent
    raise FileNotFoundError("Could not find chest_xray/train + test. Check dataset is added.")
```

### A0.4: Branch workflow (both follow)

- `main` is always working. Never push broken code to it.
- Create a branch per task: `git checkout -b a/data-pipeline`, `b/gradcam`, and so on.
- Open a Pull Request, the *other* person glances at it (5 minutes), then merge.
- Commit messages: short and specific (`add class weights to data.py`).

### B0.1: Review and verify (Day 0)

- [ ] Accept the invite, open `src/config.py`, comment on anything unclear.
- [ ] In your Kaggle notebook run the bootstrap cell below, then `print(config.find_data_root())`. It must print a valid path. If it fails, tell A and fix `find_data_root()` together.

### Bootstrap cell: first cell of EVERY notebook

```python
import os, sys
REPO = "pneumonia-gradcam"
URL  = "https://github.com/<A_USERNAME>/pneumonia-gradcam.git"   # <-- replace

if not os.path.exists(f"/kaggle/working/{REPO}"):
    !git clone {URL} /kaggle/working/{REPO}
else:
    !git -C /kaggle/working/{REPO} pull
sys.path.insert(0, f"/kaggle/working/{REPO}")

%load_ext autoreload
%autoreload 2
from src import config
print(config.find_data_root())
```

`%autoreload 2` means that after you re-run the cell above (to `git pull`), edited `.py` files are picked up without restarting the kernel.

**Done when (S0):** repo exists, B is a collaborator, `config.py` is on `main`, and B's bootstrap cell prints the data path.

---

## 3. Phase 1: Parallel work (Days 1–3)

### Person A tasks

#### A1: `src/data.py` (Day 1 morning → S1)

```python
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input
from src import config


def get_generators(root=None):
    """Returns (train_gen, val_gen, test_gen).
    - train: augmented (shear/zoom/flip), shuffled, 85% of train/
    - val:   NOT augmented, 15% of train/ (disjoint from train_gen)
    - test:  official test/ folder, NOT augmented, shuffle=False
    """
    root = root or config.find_data_root()
    # NOTE: preprocess_input (NOT rescale=1/255) -> VGG16 expects ImageNet preprocessing.
    aug = ImageDataGenerator(preprocessing_function=preprocess_input,
                             validation_split=config.VAL_SPLIT,
                             shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    plain = ImageDataGenerator(preprocessing_function=preprocess_input,
                               validation_split=config.VAL_SPLIT)
    test_dg = ImageDataGenerator(preprocessing_function=preprocess_input)

    kw = dict(target_size=config.IMG_SIZE, batch_size=config.BATCH_SIZE,
              class_mode="categorical", classes=config.CLASS_NAMES)

    train = aug.flow_from_directory(str(root / "train"), subset="training",
                                    shuffle=True, seed=config.SEED, **kw)
    val   = plain.flow_from_directory(str(root / "train"), subset="validation",
                                      shuffle=False, **kw)
    test  = test_dg.flow_from_directory(str(root / "test"), shuffle=False, **kw)
    return train, val, test


def get_class_weights(train_gen):
    w = compute_class_weight("balanced", classes=np.array([0, 1]), y=train_gen.classes)
    return {0: float(w[0]), 1: float(w[1])}
```

Checks before pushing:
- [ ] `train.class_indices` prints `{'NORMAL': 0, 'PNEUMONIA': 1}`.
- [ ] Image counts look right (train ≈ 4,434 / val ≈ 782 / test = 624).
- [ ] `get_class_weights(train)` gives NORMAL ≈ 1.9, PNEUMONIA ≈ 0.67.
- [ ] Optionally save a 3×3 grid of augmented images to `results/augmentation_samples.png`.

#### A2: `src/model.py` (Day 1 → S1)

```python
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import VGG16
from src import config


def build_vgg16(dropout=0.5):
    """VGG16 backbone (frozen) + Flatten + Dropout + linear Dense(2).
    The output is LOGITS (no softmax) so Grad-CAM gradients don't vanish.
    Always train with from_logits=True and apply softmax only when predicting."""
    base = VGG16(weights="imagenet", include_top=False, input_shape=config.INPUT_SHAPE)
    base.trainable = False
    x = layers.Flatten()(base.output)
    x = layers.Dropout(dropout)(x)
    logits = layers.Dense(2, name="logits")(x)          # linear, NOT softmax
    return Model(base.input, logits, name="vgg16_pneumonia")


def compile_model(model, lr=1e-4):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def unfreeze_block5(model, lr=1e-5):
    """Stage 2 fine-tuning: unfreeze only the last conv block, recompile with a low LR."""
    for layer in model.layers:
        layer.trainable = layer.name.startswith("block5")
    # Re-freeze the new head? No, keep head trainable too:
    for name in ("logits",):
        model.get_layer(name).trainable = True
    return compile_model(model, lr=lr)
```

Check: `model.get_layer("block5_conv3")` must work (the layers are flat inside the model, not nested). If it errors, tell B immediately.

#### A3: Notebook `02_train_vgg16.ipynb`, Stage 1 (Day 1–2 → S2)

1. Bootstrap cell, `get_generators()`, `get_class_weights()`.
2. `model = compile_model(build_vgg16())`, then `model.summary()`.
3. Callbacks:
   ```python
   cb = [
     tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
     tf.keras.callbacks.ModelCheckpoint("/kaggle/working/vgg16_stage1.keras",
                                        monitor="val_loss", save_best_only=True),
     tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2),
   ]
   ```
4. `model.fit(train, validation_data=val, epochs=10, class_weight=cw, callbacks=cb)`.
   (Do **not** pass `steps_per_epoch`; the generator already knows its length. That was the cause of the "ran out of data" warning in the GFG tutorial.)
5. Plot loss/accuracy curves and save to `results/stage1_curves.png`.
6. **Share the model with B (S2)**, see Section 7.

Long runs: use **Save Version → Save & Run All (Commit)** so training continues even if your browser closes. Outputs in `/kaggle/working` persist only after a saved version.

**Done when:** `vgg16_stage1.keras` is shared and val accuracy is meaningfully above 85%.

#### A4: Stage 2: fine-tuning (Day 2–3 → S3)

1. Load the stage-1 weights, then `model = unfreeze_block5(model, lr=1e-5)`.
2. Train 5–8 more epochs with the same callbacks (checkpoint → `vgg16_pneumonia.keras`).
3. Save curves to `results/stage2_curves.png`.
4. **Share the final model (S3).**
5. Write down in the notebook: epochs run, best val loss/accuracy, class weights used. B and the report need this.

Expected behaviour: the test set often scores *lower* than validation, because the Kaggle test distribution differs from train. This is normal and exactly the generalisation point in your literature review. Do not "fix" it by tuning on the test set.

#### A5: `05_baseline_cnn.ipynb` (Day 2–3, needs only S1)

A small CNN from scratch for the "model comparison" gap, for example 3× (Conv2D → MaxPool) with 32/64/128 filters, Flatten, Dense(128), Dropout, Dense(2). Use the same generators, class weights, and `from_logits=True`. Save as `baseline_cnn.keras`. B evaluates it with `eval_utils`.

#### A6: `08_vgg19_comparison.ipynb` (Day 4+, after S4)

Same pipeline with `VGG19`: set `LAST_CONV_LAYER = "block5_conv4"` for it, keep `preprocess_input` (identical to VGG16's). Evaluate it with B's `eval_utils`.

#### A7 (stretch): patient-level split

The Kaggle train folder has several X-rays per patient (e.g. `person1_bacteria_1.jpeg`, `person1_bacteria_2.jpeg`). A random split can put the same patient in train and validation and inflate val scores. If time permits, build a DataFrame of file paths with a patient ID parsed from the filename, split with `sklearn.model_selection.GroupShuffleSplit`, and use `flow_from_dataframe`. Mention this either way in the report as a limitation.

---

### Person B tasks

#### B1: `01_eda.ipynb` (Day 1, no dependency)

- [ ] Class counts for train / val / test (bar chart); mention the imbalance.
- [ ] 4×4 grid of sample images per class.
- [ ] Image size distribution (width/height histogram), showing why resizing to 224×224 is needed.
- [ ] Mean pixel intensity per class (boxplot). A large difference suggests the model could learn brightness or source artifacts instead of disease, which supports your bias gap.
- [ ] Save figures to `results/eda_*.png`.

#### B2: `src/gradcam.py` + `03_gradcam_dev.ipynb` (Day 1–2, **no trained model needed**)

```python
import numpy as np, cv2, tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import preprocess_input
from src import config


def load_image(path):
    """Returns (rgb_uint8 for display, preprocessed batch for the model)."""
    img = tf.keras.utils.load_img(path, target_size=config.IMG_SIZE)   # RGB
    rgb = np.array(img, dtype=np.uint8)
    batch = preprocess_input(np.expand_dims(rgb.astype("float32"), 0))  # astype -> copy
    return rgb, batch


def make_gradcam(model, batch, class_idx=None, layer=config.LAST_CONV_LAYER):
    """Grad-CAM. Returns (heatmap in [0,1] shape ~14x14, softmax probs)."""
    grad_model = tf.keras.Model(model.inputs,
                                [model.get_layer(layer).output, model.output])
    with tf.GradientTape() as tape:
        conv_out, logits = grad_model(batch, training=False)
        if class_idx is None:
            class_idx = int(tf.argmax(logits[0]))
        score = logits[:, class_idx]
    grads = tape.gradient(score, conv_out)
    weights = tf.reduce_mean(grads, axis=(1, 2))                 # (1, C)
    cam = tf.reduce_sum(conv_out * weights[:, None, None, :], axis=-1)[0]
    cam = tf.nn.relu(cam)
    cam = (cam / (tf.reduce_max(cam) + 1e-8)).numpy()
    return cam, tf.nn.softmax(logits[0]).numpy()


def overlay(cam, rgb_uint8, alpha=0.4):
    cam = cv2.resize(cam, (rgb_uint8.shape[1], rgb_uint8.shape[0]))
    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)[..., ::-1]  # BGR->RGB
    return np.uint8((1 - alpha) * rgb_uint8 + alpha * heat)


def plot_gradcam_grid(model, paths, true_labels=None, ncols=4, save_path=None):
    """For each image: original | overlay, titled with predicted class + confidence."""
    n = len(paths); nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols * 2, figsize=(4 * ncols * 2 / 2 * 1.0, 2.6 * nrows))
    axes = np.atleast_2d(axes)
    for i, p in enumerate(paths):
        r, c = divmod(i, ncols)
        rgb, batch = load_image(p)
        cam, probs = make_gradcam(model, batch)
        pred = int(probs.argmax())
        ax_o, ax_h = axes[r, 2 * c], axes[r, 2 * c + 1]
        ax_o.imshow(rgb); ax_o.axis("off")
        ax_h.imshow(overlay(cam, rgb)); ax_h.axis("off")
        t = f"pred: {config.CLASS_NAMES[pred]} ({probs[pred]:.2f})"
        if true_labels is not None:
            t = f"true: {config.CLASS_NAMES[true_labels[i]]}\n" + t
        ax_h.set_title(t, fontsize=8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
```

**Test before the trained model exists** (in `03_gradcam_dev.ipynb`):

```python
# ImageNet VGG16 with its original 1000-class head; same layer names as ours.
m = tf.keras.applications.VGG16(weights="imagenet")          # includes softmax head
rgb, batch = load_image("some_cat_or_dog_photo.jpg")         # upload any photo as a dataset/file
cam, _ = make_gradcam(m, batch)
plt.imshow(overlay(cam, rgb))
```

The heat should sit on the animal/object. (Here the model's output is already probabilities, so `make_gradcam`'s extra softmax only flattens the displayed numbers; the heatmap is still correct. Ignore the printed probs in this test.) When the heat lands on the object, the mechanics (layer name, gradients, resize, overlay colours) are correct.

Also test with a randomly initialised `build_vgg16()` once S1 lands, to confirm the code runs on the real architecture (heatmaps will be meaningless, but shapes must work).

**Done when:** the sanity test highlights the object, and `plot_gradcam_grid` saves an image. Push `gradcam.py` (S4).

#### B3: `src/eval_utils.py` (Day 2, writes against the contract; test once S1 lands)

```python
import json, numpy as np, tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, ConfusionMatrixDisplay)
from src import config


def predict_probs(model, gen):
    """gen MUST have shuffle=False. Returns (y_true, probs[N,2])."""
    gen.reset()
    logits = model.predict(gen, verbose=0)
    return gen.classes, tf.nn.softmax(logits, axis=1).numpy()


def full_report(y_true, probs, tag, out_dir="results"):
    """Prints + saves: classification report (JSON), confusion matrix PNG, ROC PNG."""
    y_pred = probs.argmax(1)
    rep = classification_report(y_true, y_pred, target_names=config.CLASS_NAMES, output_dict=True)
    rep["roc_auc"] = float(roc_auc_score(y_true, probs[:, 1]))
    json.dump(rep, open(f"{out_dir}/{tag}_metrics.json", "w"), indent=2)
    print(classification_report(y_true, y_pred, target_names=config.CLASS_NAMES), "ROC-AUC:", rep["roc_auc"])

    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred), display_labels=config.CLASS_NAMES).plot()
    plt.savefig(f"{out_dir}/{tag}_confusion.png", dpi=150, bbox_inches="tight"); plt.show()

    fpr, tpr, _ = roc_curve(y_true, probs[:, 1])
    plt.plot(fpr, tpr, label=f"AUC={rep['roc_auc']:.3f}"); plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.legend()
    plt.savefig(f"{out_dir}/{tag}_roc.png", dpi=150, bbox_inches="tight"); plt.show()
    return rep


def plot_history(history, tag, out_dir="results"):
    """Loss & accuracy curves from a Keras History object."""
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    for a, k in zip(ax, ["loss", "accuracy"]):
        a.plot(history.history[k], label="train"); a.plot(history.history["val_" + k], label="val")
        a.set_title(k); a.legend()
    plt.savefig(f"{out_dir}/{tag}_curves.png", dpi=150, bbox_inches="tight"); plt.show()


def misclassified_indices(y_true, probs):
    """Returns dict of index arrays: tp, tn, fp, fn (pneumonia = positive class)."""
    y_pred = probs.argmax(1)
    return {
        "tp": np.where((y_true == 1) & (y_pred == 1))[0],
        "tn": np.where((y_true == 0) & (y_pred == 0))[0],
        "fp": np.where((y_true == 0) & (y_pred == 1))[0],
        "fn": np.where((y_true == 1) & (y_pred == 0))[0],
    }
```

`test_gen.filepaths[i]` gives the file path of the i-th prediction (generators with `shuffle=False` keep order). B6 relies on this.

#### B4: `04_evaluation.ipynb`: first run (Day 3, after S2)

- Load `vgg16_stage1.keras`, `get_generators()`, then `predict_probs(model, test)` and `full_report(...)`.
- Report accuracy, precision, recall, F1 per class, ROC-AUC and the confusion matrix.
- Compare val vs test numbers; note the gap in text.
- Evaluate A's `baseline_cnn.keras` the same way (needs A5).
- Re-run later for the fine-tuned model (S3) and VGG19 (A6). Keep one results table: `results/comparison.csv` (columns: model, accuracy, precision_pneu, recall_pneu, f1_pneu, auc).

Pay special attention to **recall of PNEUMONIA** (false negatives = missed disease) and **recall of NORMAL** (specificity).

#### B5: Grad-CAM on the real model (Day 3, after S2)

In `03_gradcam_dev.ipynb` (or a new section): load the stage-1 model, run `plot_gradcam_grid` on about 8 test images (4 NORMAL + 4 PNEUMONIA). Check:
- Are heatmaps non-empty (not all-zero)? If blank, the logits trick is missing; tell A.
- Do they sit on the lung fields rather than corners, text markers, or the image border?

Report back to A. Poor localisation before fine-tuning is expected; if it stays bad after fine-tuning, that is a finding for the report.

#### B6: `06_gradcam_analysis.ipynb`, the core uniqueness deliverable (Day 4–5, after S3)

1. Using `misclassified_indices`, pick **6 examples each** of TP, TN, FP, FN (fixed seed, saved list of filenames in `results/gradcam_cases.json` so it is reproducible).
2. Produce one grid per category: `results/gradcam_tp.png`, `_tn.png`, `_fp.png`, `_fn.png`.
3. For each grid, write 3–5 sentences: where does the model look? Is it clinically plausible (lung fields) or suspicious (shoulders, corners, markers, heart border)?
4. Also show Grad-CAM for **both classes** on a few images (`class_idx=0` vs `class_idx=1`).
5. Summarise in a short table: category → typical focus region → comment.

Stretch ideas (pick one if time allows):
- Compare Grad-CAM before vs after fine-tuning.
- **Sanity check:** re-initialise `block5` weights randomly and show that the heatmaps change drastically (an explanation that does not depend on the learned weights would be untrustworthy).
- Compare VGG16 vs VGG19 heatmaps on the same images.

#### B7: `07_demo.ipynb`

One clean notebook: load the model, upload/select a single X-ray, print prediction + confidence, and show original | Grad-CAM overlay. Optional: wrap it with `gradio` (`gr.Interface(fn, gr.Image(), gr.Image())`).
Add the disclaimer: *"Research prototype, not a diagnostic tool."*

---

## 4. Phase 2: Integration (after S2/S3, about half a day, both)

1. A posts the model (Section 7). B pulls it and runs B4 + B5.
2. Joint 20-minute call:
   - Test metrics acceptable? (Pneumonia recall high; Normal recall not terrible.)
   - Do heatmaps look sensible?
3. If heatmaps or the test gap look bad, A tries (in this order): more fine-tuning epochs, stronger augmentation, lower LR, VGG19. **Never tune on the test set.**
4. Freeze the final model; note the exact notebook version that produced it.

---

## 5. Phase 3: Parallel analysis (Days 4–5)

| A | B |
|---|---|
| A6 VGG19 comparison | B6 Grad-CAM analysis |
| A7 patient-level split (stretch) | B4 final evaluation table (all models) |
| Draft Methodology + Training sections | Draft Dataset/EDA + Evaluation + Grad-CAM sections |

---

## 6. Phase 4: Wrap-up (Days 5–6, both)

- [ ] A: `README.md` with project overview, how to run (Kaggle steps), results table, folder map.
- [ ] B: `07_demo.ipynb` finished and tested from a fresh Kaggle session (clone → run all).
- [ ] Both: download all notebooks (File → Download Notebook) and commit them to `notebooks/` (clear heavy outputs if the files are large).
- [ ] Update your methodology document: add **"1.7 Explainability using Grad-CAM"** and update the abstract/gap table. Your literature doc's note says Grad-CAM should be called future work unless implemented; it is now implemented, so change this. Independent-dataset evaluation remains future work unless you do it.
- [ ] Final check: someone who has never seen the repo can follow the README and reproduce the numbers.

---

## 7. Sharing trained models between A and B

Model files (~60 MB) are **not** committed to Git. Recommended method:

1. After a notebook run is saved (**Save Version**), open the notebook's **Output** tab → **New Dataset** → name it `pneumonia-models` (private).
2. Dataset → **Share** → add B's Kaggle username as collaborator.
3. B: in their notebook → **Add Input** → search `pneumonia-models` under "Your Datasets" → load:
   ```python
   model = tf.keras.models.load_model("/kaggle/input/pneumonia-models/vgg16_pneumonia.keras")
   ```
4. For later versions, A uses the dataset's **New Version** so the same path keeps working.

Backup option: upload the `.keras` file to a shared Google Drive folder, or attach it to a GitHub Release.

---

## 8. Report split

| Section | Owner |
|---|---|
| Abstract, introduction, literature review (reuse existing docs) | A |
| Dataset + EDA + class imbalance + bias discussion | B |
| Methodology: preprocessing, augmentation, VGG16 transfer learning, two-stage training | A |
| Methodology: Grad-CAM | B |
| Results: training curves, baseline vs VGG16 vs VGG19 | A |
| Results: metrics, confusion matrices, ROC, Grad-CAM analysis (TP/TN/FP/FN) | B |
| Limitations, future work, conclusion | Both (review each other's text) |

Honest limitations to include: single dataset (no external validation), possible patient overlap in the random split (unless A7 done), test-set distribution shift, Grad-CAM is a coarse qualitative explanation and not proof the model reasons like a clinician, research prototype only.

---

## 9. If you use Colab instead of Kaggle

- Runtime → Change runtime type → GPU.
- Data: either upload `kaggle.json` (Kaggle → Settings → API → Create New Token) and run
  `!pip install -q kaggle && kaggle datasets download -d paultimothymooney/chest-xray-pneumonia && unzip -q chest-xray-pneumonia.zip -d /content/data`
  or copy the dataset to Google Drive once and mount it.
- Bootstrap cell: same `git clone`, but use `/content/` instead of `/kaggle/working/`.
- Save models to Google Drive (`/content/drive/MyDrive/...`) and share the folder with your teammate instead of using Kaggle Datasets.

---

## 10. Troubleshooting

| Problem | Fix |
|---|---|
| `git clone` fails in Kaggle | Settings → Internet **On** (needs phone-verified account); repo must be public |
| No GPU option | Verify phone number; check weekly GPU quota on your Kaggle profile |
| `find_data_root()` cannot find data | Dataset not added as Input, or extra nesting; print the tree from Section 1.1 |
| `get_layer("block5_conv3")` fails | The model was built with a nested backbone; rebuild with `Model(base.input, logits)` as in `model.py` |
| Grad-CAM heatmap is blank | The model output is softmax instead of logits, or the model is untrained/saturated; use the logits model |
| Heatmaps look "blocky" | Normal: block5 feature map is 14×14, upsampled to 224×224 |
| Test accuracy much lower than val | Known Kaggle distribution shift; report it honestly and rely on recall/F1/AUC |
| Edited `.py` change not picked up | Re-run the bootstrap cell (`git pull`) with `%autoreload 2` active, or restart the kernel |
| Training lost when the session ended | Use **Save Version → Save & Run All**; checkpoints go in `/kaggle/working` |
| `ImageDataGenerator` import/deprecation errors on a newer TF | Pin an older TensorFlow, or ask for the `image_dataset_from_directory` version of `data.py` |
| Merge conflict in a notebook | Avoid editing the same notebook; keep notebooks single-owner (Section 0.2) |

---

## 11. Master checklist

**Phase 0**
- [ ] A0.1–A0.3 Repo, structure, `config.py` (S0)
- [ ] B0.1 Setup verified, bootstrap cell works

**Phase 1**
- [ ] A1 `data.py`  · [ ] A2 `model.py` (S1)
- [ ] A3 Stage-1 training (S2)  · [ ] A4 Fine-tuning (S3)  · [ ] A5 Baseline CNN
- [ ] B1 EDA  · [ ] B2 `gradcam.py` + sanity test  · [ ] B3 `eval_utils.py` (S4)

**Phase 2**
- [ ] B4 Evaluation run  · [ ] B5 Grad-CAM on real model  · [ ] Joint review call

**Phase 3**
- [ ] A6 VGG19 comparison  · [ ] A7 patient-level split (stretch)
- [ ] B6 Grad-CAM TP/TN/FP/FN analysis  · [ ] Final comparison table

**Phase 4**
- [ ] B7 Demo  · [ ] README  · [ ] Notebooks committed  · [ ] Docs updated with Grad-CAM  · [ ] Report finished
