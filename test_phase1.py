"""
Test script for Phase 1 implementation (Dev A's modules).
Validates data loading and model building without requiring actual data.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src import config
from src.data import get_generators, get_class_weights
from src.model import build_vgg16, compile_model, unfreeze_block5


def test_config():
    """Test configuration constants."""
    print("\n" + "="*60)
    print("Testing Configuration (src/config.py)")
    print("="*60)
    
    print(f"✓ IMG_SIZE: {config.IMG_SIZE}")
    print(f"✓ INPUT_SHAPE: {config.INPUT_SHAPE}")
    print(f"✓ BATCH_SIZE: {config.BATCH_SIZE}")
    print(f"✓ SEED: {config.SEED}")
    print(f"✓ VAL_SPLIT: {config.VAL_SPLIT}")
    print(f"✓ CLASS_NAMES: {config.CLASS_NAMES}")
    print(f"✓ LAST_CONV_LAYER: {config.LAST_CONV_LAYER}")
    print(f"✓ MODEL_FILE: {config.MODEL_FILE}")
    
    # Test find_data_root (will fail gracefully if no data)
    try:
        data_root = config.find_data_root()
        print(f"✓ DATA_ROOT found: {data_root}")
    except FileNotFoundError as e:
        print(f"ℹ DATA_ROOT not found (expected if data not downloaded): {e}")


def test_model_building():
    """Test model building and compilation."""
    print("\n" + "="*60)
    print("Testing Model Building (src/model.py)")
    print("="*60)
    
    # Build model
    print("\n[1] Building VGG16 model...")
    model = build_vgg16(dropout=0.5)
    
    # Verify architecture
    assert model.name == "vgg16_pneumonia", "Model name mismatch"
    assert model.input_shape == (None, 224, 224, 3), "Input shape mismatch"
    assert model.output_shape == (None, 2), "Output shape mismatch"
    print("✓ Architecture verified")
    
    # Compile model
    print("\n[2] Compiling model (Stage 1)...")
    compile_model(model, lr=1e-4)
    assert model.optimizer is not None, "Model not compiled"
    print("✓ Model compiled successfully")
    
    # Test unfreezing block5
    print("\n[3] Unfreezing Block 5 (Stage 2)...")
    initial_trainable = sum([w.shape.num_elements() for w in model.trainable_weights])
    unfreeze_block5(model, lr=1e-5)
    final_trainable = sum([w.shape.num_elements() for w in model.trainable_weights])
    
    assert final_trainable > initial_trainable, "Block5 unfreezing failed"
    print(f"✓ Trainable params increased: {initial_trainable:,} → {final_trainable:,}")
    
    # Display model summary
    print("\n[4] Model Summary:")
    print("-" * 60)
    model.summary()
    
    return model


def test_data_loading_mock():
    """Test data loading functions (without actual data)."""
    print("\n" + "="*60)
    print("Testing Data Loading (src/data.py)")
    print("="*60)
    
    print("\nℹ Data loading requires the Kaggle chest X-ray dataset.")
    print("ℹ To test with real data:")
    print("  1. Download: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")
    print("  2. Extract to: data/chest_xray/")
    print("  3. Run: python test_phase1.py")
    
    # Try to load if data exists
    try:
        print("\n[Attempting to load data...]")
        train_gen, val_gen, test_gen = get_generators()
        
        print("\n✓ Data generators created successfully!")
        
        # Test class weights
        print("\n[Computing class weights...]")
        class_weights = get_class_weights(train_gen)
        print("✓ Class weights computed successfully!")
        
        return train_gen, val_gen, test_gen, class_weights
        
    except FileNotFoundError as e:
        print(f"\n✗ Data not found: {e}")
        print("✓ Data loading code is ready (waiting for dataset)")
        return None, None, None, None


def main():
    """Run all tests."""
    print("="*60)
    print("Phase 1 Implementation Test Suite")
    print("Dev A: Data Loading & Model Building")
    print("="*60)
    
    # Test configuration
    test_config()
    
    # Test model building
    model = test_model_building()
    
    # Test data loading
    generators = test_data_loading_mock()
    
    # Final summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    print("✓ Configuration: PASSED")
    print("✓ Model Building: PASSED")
    print("✓ Model Compilation: PASSED")
    print("✓ Block5 Unfreezing: PASSED")
    
    if generators[0] is not None:
        print("✓ Data Loading: PASSED")
        print("✓ Class Weights: PASSED")
    else:
        print("ℹ Data Loading: READY (awaiting dataset)")
    
    print("\n" + "="*60)
    print("Phase 1 Implementation: COMPLETE ✓")
    print("="*60)


if __name__ == "__main__":
    main()
