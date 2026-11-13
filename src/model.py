"""
Model definition and training utilities (Dev A's responsibility).
Implements VGG16 transfer learning with two-stage training support.
"""
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16

from . import config


def build_vgg16(dropout: float = 0.5) -> Model:
    """
    Build VGG16 model with transfer learning for binary classification.
    
    Architecture:
    - VGG16 base (pretrained on ImageNet, frozen initially)
    - Flatten layer
    - Dropout layer
    - Dense layer with 2 units (logits, no softmax)
    
    The final dense layer outputs raw logits (no activation) to avoid
    gradient vanishing during training with from_logits=True loss.
    
    Args:
        dropout: Dropout rate for regularization (default: 0.5)
        
    Returns:
        Uncompiled Keras Model
    """
    # Load pretrained VGG16 without top classification layer
    base_model = VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=config.INPUT_SHAPE
    )
    
    # Freeze the base model for transfer learning
    base_model.trainable = False
    
    # Build classification head
    x = base_model.output
    x = Flatten(name="flatten")(x)
    x = Dropout(dropout, name="dropout")(x)
    
    # Output logits (NO softmax activation)
    # This prevents gradient vanishing when using from_logits=True
    outputs = Dense(2, activation=None, name="logits")(x)
    
    # Create the full model
    model = Model(
        inputs=base_model.input,
        outputs=outputs,
        name="vgg16_pneumonia"
    )
    
    print(f"✓ Model built: {model.name}")
    print(f"✓ Total params: {model.count_params():,}")
    print(f"✓ Trainable params: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
    
    return model


def compile_model(model: Model, lr: float = 1e-4) -> None:
    """
    Compile model with optimizer, loss, and metrics.
    
    Uses CategoricalCrossentropy with from_logits=True since the model
    outputs raw logits without softmax activation.
    
    Args:
        model: Keras model to compile
        lr: Learning rate for Adam optimizer (default: 1e-4)
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )
    
    print(f"✓ Model compiled with lr={lr}")


def unfreeze_block5(model: Model, lr: float = 1e-5) -> None:
    """
    Unfreeze Block 5 of VGG16 for fine-tuning (Stage 2 training).
    
    This enables fine-tuning of the deepest convolutional block while
    keeping earlier layers frozen. Uses a lower learning rate to prevent
    catastrophic forgetting of pretrained features.
    
    Args:
        model: Compiled Keras model with VGG16 base
        lr: Fine-tuning learning rate (default: 1e-5, lower than stage 1)
    """
    # Unfreeze layers whose names start with "block5"
    for layer in model.layers:
        if layer.name.startswith("block5"):
            layer.trainable = True
        elif layer.name == "logits":
            # Keep the output layer trainable
            layer.trainable = True
        else:
            # Keep all other layers frozen
            layer.trainable = False
    
    # Recompile with lower learning rate for fine-tuning
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )
    
    trainable_count = sum([tf.size(w).numpy() for w in model.trainable_weights])
    print(f"✓ Block5 unfrozen for fine-tuning")
    print(f"✓ Trainable params: {trainable_count:,}")
    print(f"✓ Model recompiled with lr={lr}")
