# Notebooks Directory

This directory contains Jupyter notebooks and templates for running experiments on Kaggle.

## 📁 Files

### `kaggle_test_template.py`
Quick-start template for testing your trained model on Kaggle.

**Usage:**
1. Create a new Kaggle notebook
2. Copy the entire contents of this file
3. Update `GITHUB_REPO_URL` and `MODEL_PATH`
4. Run the cell

**What it does:**
- Clones your GitHub repository
- Loads your trained model
- Evaluates on test set
- Generates Grad-CAM visualizations
- Saves all results to `/kaggle/working/`

## 🚀 Quick Start on Kaggle

### Prerequisites
1. **Upload your model** to Kaggle as a dataset
2. **Verify your phone** on Kaggle (required for GPU + Internet)
3. **Make your GitHub repo public** (or use authentication)

### Steps
1. **Create new Kaggle notebook**
2. **Add inputs**:
   - Chest X-Ray Pneumonia dataset (Paul Mooney)
   - Your model dataset
3. **Settings**:
   - Accelerator: GPU T4 x2
   - Internet: ON
4. **Copy code** from `kaggle_test_template.py`
5. **Update** the configuration variables
6. **Run all** and wait for results
7. **Download** outputs from the Output tab

## 📊 Expected Outputs

After running, you'll get:

```
/kaggle/working/
├── test_metrics.json           # Detailed metrics
├── test_confusion.png          # Confusion matrix
├── test_roc.png               # ROC curve
├── gradcam_tp.png             # True Positives
├── gradcam_tn.png             # True Negatives
├── gradcam_fp.png             # False Positives (if any)
└── gradcam_fn.png             # False Negatives (if any)
```

## 🔗 Resources

- **Full Guide**: See `../KAGGLE_TESTING_GUIDE.md`
- **Project Plan**: See `../PLAN_Pneumonia_GradCAM_A_B.md`
- **Phase 1 Guide**: See `../PHASE1_USAGE.md`
- **Phase 2 Guide**: See `../PHASE2_USAGE.md`

## 📝 Notes

### Your Notebooks (After Training)
Add your completed Kaggle notebooks here:

- `01_eda.ipynb` - Exploratory Data Analysis (Dev B)
- `02_train_vgg16.ipynb` - Training notebook (Dev A)
- `03_gradcam_dev.ipynb` - Grad-CAM development (Dev B)
- `04_evaluation.ipynb` - Model evaluation (Dev B)
- `05_baseline_cnn.ipynb` - Baseline model (Dev A)
- `06_gradcam_analysis.ipynb` - Error analysis (Dev B)
- `07_demo.ipynb` - Interactive demo (Dev B)
- `08_vgg19_comparison.ipynb` - Model comparison (Dev A)

To download from Kaggle:
1. Open your notebook
2. File → Download Notebook
3. Save to this directory
4. Commit to Git

### Notebook Best Practices

1. **Clear outputs** before committing (File → Clear All Outputs)
2. **Add markdown** explaining each section
3. **Include results** in markdown cells as text (not just images)
4. **Comment your code** for reproducibility
5. **Test on fresh kernel** before saving final version

## 🐛 Troubleshooting

### Model not loading?
```python
# Check available files
!ls -la /kaggle/input/your-model-dataset/

# Try different path
MODEL_PATH = "/kaggle/input/your-model-dataset/model.keras"
```

### Git clone failing?
```python
# Check internet is enabled
!curl -I https://github.com

# Try with explicit clone
!git clone https://github.com/username/repo.git /kaggle/working/repo
```

### Layer not found in Grad-CAM?
```python
# List all layers
for layer in model.layers:
    print(layer.name)

# Update config if needed
config.LAST_CONV_LAYER = "your_layer_name"
```

### Data not found?
```python
# Check data location
!find /kaggle/input -name "train" -type d
```

## 💡 Tips

- **Use GPU**: Set Accelerator to GPU T4 x2 for faster inference
- **Enable Internet**: Required to clone GitHub repo
- **Save often**: Kaggle can disconnect; use "Save & Run All"
- **Check quotas**: Kaggle has weekly GPU hours limit
- **Version notebooks**: Use "Save Version" to create checkpoints
