"""
Test script for Phase 2 implementation (Dev B's modules).
Validates Grad-CAM and evaluation utilities without requiring actual data or trained model.
"""
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src import config
from src.gradcam import (
    load_and_preprocess_image,
    make_gradcam_heatmap,
    overlay_heatmap,
    plot_gradcam_grid
)
from src.eval_utils import (
    get_model_prediction,
    predict_probs,
    full_report,
    plot_history,
    misclassified_indices,
    print_misclassification_summary
)
from src.model import build_vgg16, compile_model


def test_gradcam_functions():
    """Test Grad-CAM functions with a dummy model."""
    print("\n" + "="*60)
    print("Testing Grad-CAM Functions (src/gradcam.py)")
    print("="*60)
    
    # Build a test model
    print("\n[1] Building test model...")
    model = build_vgg16(dropout=0.5)
    compile_model(model, lr=1e-4)
    print("✓ Model built successfully")
    
    # Verify layer exists
    print("\n[2] Verifying last conv layer...")
    try:
        layer = model.get_layer(config.LAST_CONV_LAYER)
        print(f"✓ Layer '{config.LAST_CONV_LAYER}' found: {layer}")
    except ValueError as e:
        print(f"✗ Layer not found: {e}")
        return False
    
    # Test with random input (simulating preprocessed image)
    print("\n[3] Testing Grad-CAM generation with random input...")
    dummy_input = np.random.randn(1, 224, 224, 3).astype(np.float32)
    
    try:
        heatmap, probs = make_gradcam_heatmap(dummy_input, model)
        print(f"✓ Heatmap shape: {heatmap.shape}")
        print(f"✓ Heatmap range: [{heatmap.min():.3f}, {heatmap.max():.3f}]")
        print(f"✓ Probabilities: {probs}")
        print(f"✓ Probabilities sum: {probs.sum():.3f} (should be ~1.0)")
        
        assert heatmap.ndim == 2, "Heatmap should be 2D"
        assert 0 <= heatmap.min() <= heatmap.max() <= 1, "Heatmap should be in [0, 1]"
        assert len(probs) == 2, "Should have 2 class probabilities"
        assert abs(probs.sum() - 1.0) < 0.01, "Probabilities should sum to 1"
        
    except Exception as e:
        print(f"✗ Grad-CAM generation failed: {e}")
        return False
    
    # Test overlay function
    print("\n[4] Testing heatmap overlay...")
    try:
        dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        overlay_img = overlay_heatmap(dummy_img, heatmap, alpha=0.4)
        
        print(f"✓ Overlay shape: {overlay_img.shape}")
        print(f"✓ Overlay dtype: {overlay_img.dtype}")
        assert overlay_img.shape == (224, 224, 3), "Overlay should match input shape"
        assert overlay_img.dtype == np.uint8, "Overlay should be uint8"
        
    except Exception as e:
        print(f"✗ Overlay generation failed: {e}")
        return False
    
    # Test with specific class index
    print("\n[5] Testing Grad-CAM with specific class index...")
    try:
        heatmap_class0, _ = make_gradcam_heatmap(dummy_input, model, pred_index=0)
        heatmap_class1, _ = make_gradcam_heatmap(dummy_input, model, pred_index=1)
        
        print(f"✓ Class 0 heatmap generated: {heatmap_class0.shape}")
        print(f"✓ Class 1 heatmap generated: {heatmap_class1.shape}")
        
    except Exception as e:
        print(f"✗ Class-specific Grad-CAM failed: {e}")
        return False
    
    print("\n✓ All Grad-CAM tests passed!")
    return True


def test_eval_functions():
    """Test evaluation utility functions."""
    print("\n" + "="*60)
    print("Testing Evaluation Functions (src/eval_utils.py)")
    print("="*60)
    
    # Build test model
    print("\n[1] Building test model...")
    model = build_vgg16(dropout=0.5)
    compile_model(model, lr=1e-4)
    print("✓ Model built")
    
    # Test get_model_prediction
    print("\n[2] Testing get_model_prediction...")
    try:
        dummy_input = np.random.randn(1, 224, 224, 3).astype(np.float32)
        result = get_model_prediction(model, dummy_input)
        
        print(f"✓ Prediction: {result['class_name']} (idx: {result['class_idx']})")
        print(f"✓ Confidence: {result['confidence']:.2%}")
        print(f"✓ Probabilities: {result['probabilities']}")
        
        assert 'class_idx' in result, "Missing class_idx"
        assert 'class_name' in result, "Missing class_name"
        assert 'confidence' in result, "Missing confidence"
        assert 0 <= result['confidence'] <= 1, "Confidence should be in [0, 1]"
        
    except Exception as e:
        print(f"✗ get_model_prediction failed: {e}")
        return False
    
    # Test misclassified_indices
    print("\n[3] Testing misclassified_indices...")
    try:
        # Create dummy predictions
        y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0])
        probs = np.array([
            [0.9, 0.1],  # TN
            [0.3, 0.7],  # FP
            [0.2, 0.8],  # TP
            [0.7, 0.3],  # FN
            [0.8, 0.2],  # TN
            [0.1, 0.9],  # TP
            [0.3, 0.7],  # TP
            [0.6, 0.4],  # TN
        ])
        
        indices = misclassified_indices(y_true, probs)
        
        print(f"✓ True Positives: {len(indices['tp'])} indices")
        print(f"✓ True Negatives: {len(indices['tn'])} indices")
        print(f"✓ False Positives: {len(indices['fp'])} indices")
        print(f"✓ False Negatives: {len(indices['fn'])} indices")
        
        assert len(indices['tp']) == 3, f"Expected 3 TPs, got {len(indices['tp'])}"
        assert len(indices['tn']) == 3, f"Expected 3 TNs, got {len(indices['tn'])}"
        assert len(indices['fp']) == 1, f"Expected 1 FP, got {len(indices['fp'])}"
        assert len(indices['fn']) == 1, f"Expected 1 FN, got {len(indices['fn'])}"
        
        print_misclassification_summary(y_true, probs)
        
    except Exception as e:
        print(f"✗ misclassified_indices failed: {e}")
        return False
    
    # Test plot_history (without actual training)
    print("\n[4] Testing plot_history...")
    try:
        # Create dummy history object
        class DummyHistory:
            def __init__(self):
                self.history = {
                    'loss': [0.5, 0.4, 0.3, 0.2],
                    'val_loss': [0.6, 0.5, 0.4, 0.35],
                    'accuracy': [0.7, 0.75, 0.8, 0.85],
                    'val_accuracy': [0.65, 0.7, 0.75, 0.8]
                }
        
        history = DummyHistory()
        
        # Create results directory if it doesn't exist
        Path("results").mkdir(exist_ok=True)
        
        plot_history(history, "test_curves")
        print("✓ Training curves plotted and saved")
        
    except Exception as e:
        print(f"✗ plot_history failed: {e}")
        return False
    
    # Test full_report
    print("\n[5] Testing full_report...")
    try:
        # Generate larger dummy dataset
        np.random.seed(42)
        n_samples = 100
        y_true = np.random.randint(0, 2, n_samples)
        
        # Generate realistic-looking probabilities
        probs = np.zeros((n_samples, 2))
        for i in range(n_samples):
            if y_true[i] == 0:
                # NORMAL: higher prob for class 0
                probs[i, 0] = np.random.uniform(0.4, 0.95)
            else:
                # PNEUMONIA: higher prob for class 1
                probs[i, 1] = np.random.uniform(0.4, 0.95)
            probs[i] = probs[i] / probs[i].sum()  # Normalize
        
        report = full_report(y_true, probs, "test_full_report")
        
        print("\n✓ Full report generated successfully!")
        print(f"✓ ROC-AUC: {report['roc_auc']:.3f}")
        
    except Exception as e:
        print(f"✗ full_report failed: {e}")
        return False
    
    print("\n✓ All evaluation tests passed!")
    return True


def test_integration():
    """Test integration between modules."""
    print("\n" + "="*60)
    print("Testing Module Integration")
    print("="*60)
    
    print("\n[1] Testing config imports...")
    assert config.IMG_SIZE == (224, 224), "IMG_SIZE mismatch"
    assert config.CLASS_NAMES == ["NORMAL", "PNEUMONIA"], "CLASS_NAMES mismatch"
    assert config.LAST_CONV_LAYER == "block5_conv3", "LAST_CONV_LAYER mismatch"
    print("✓ Config values correct")
    
    print("\n[2] Testing model + gradcam integration...")
    model = build_vgg16()
    compile_model(model)
    dummy_input = np.random.randn(1, 224, 224, 3).astype(np.float32)
    
    # Get prediction
    pred = get_model_prediction(model, dummy_input)
    
    # Generate Grad-CAM for predicted class
    heatmap, probs = make_gradcam_heatmap(dummy_input, model, pred_index=pred['class_idx'])
    
    print(f"✓ Predicted: {pred['class_name']} with confidence {pred['confidence']:.2%}")
    print(f"✓ Grad-CAM heatmap shape: {heatmap.shape}")
    
    print("\n✓ Integration tests passed!")
    return True


def main():
    """Run all tests."""
    print("="*60)
    print("Phase 2 Implementation Test Suite")
    print("Dev B: Grad-CAM & Evaluation Utilities")
    print("="*60)
    
    # Suppress TensorFlow warnings for cleaner output
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    results = {
        "Grad-CAM Functions": test_gradcam_functions(),
        "Evaluation Functions": test_eval_functions(),
        "Module Integration": test_integration()
    }
    
    # Final summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*60)
    if all_passed:
        print("Phase 2 Implementation: COMPLETE ✓")
    else:
        print("Phase 2 Implementation: INCOMPLETE ✗")
    print("="*60)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
