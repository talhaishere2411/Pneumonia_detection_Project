# Pneumonia Detection with Grad-CAM

A deep learning project for detecting pneumonia from chest X-ray images using VGG16 with Grad-CAM visualization.

## Project Structure

```
pneumonia-gradcam/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── config.py          # Shared configuration
│   ├── data.py            # Data loading utilities
│   ├── model.py           # Model training
│   ├── gradcam.py         # Grad-CAM visualization
│   └── eval_utils.py      # Evaluation utilities
├── notebooks/             # Jupyter notebooks
├── results/               # Output visualizations
└── models/                # Saved model files (gitignored)
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Place the Kaggle chest X-ray dataset in the `data/` directory or use the auto-detection in `config.py`.

## Dataset

This project uses the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset from Kaggle.

## Configuration

All shared configuration is centralized in `src/config.py`:
- Image size: 224x224
- Batch size: 32
- Validation split: 15% from training data
- Classes: NORMAL (0), PNEUMONIA (1)

## Usage

Training and evaluation scripts will be added in subsequent phases.

## License

MIT License
