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
    """
    base_model = VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=config.INPUT_SHAPE
    )
    
    base_model.trainable = False
    
    x = base_model.output
    x = Flatten(name="flatten")(x)
    x = Dropout(dropout, name="dropout")(x)
    
    outputs = Dense(2, activation=None, name="logits")(x)
    
    model = Model(
        inputs=base_model.input,
        outputs=outputs,
        name="vgg16_pneumonia"
    )
    
    print(f"✓ Model built: {model.name}")
    print(f"✓ Total params: {model.count_params():,}")
    print(f"✓ Trainable params: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
    
    return model


def compile_model(model: Model, lr: float = 1e-4) -> Model:
    """
    Compile model with optimizer, loss, and metrics, and return it.
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )
    
    print(f"✓ Model compiled with lr={lr}")
    return model  # <--- Added return statement


def unfreeze_block5(model: Model, lr: float = 1e-5) -> Model:
    """
    Unfreeze Block 5 of VGG16 for fine-tuning (Stage 2 training), recompile, and return.
    """
    for layer in model.layers:
        if layer.name.startswith("block5"):
            layer.trainable = True
        elif layer.name == "logits":
            layer.trainable = True
        else:
            layer.trainable = False
            
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )
    
    trainable_count = sum([tf.size(w).numpy() for w in model.trainable_weights])
    print(f"✓ Block5 unfrozen for fine-tuning")
    print(f"✓ Trainable params: {trainable_count:,}")
    print(f"✓ Model recompiled with lr={lr}")
    
    return model  # <--- Added return statement