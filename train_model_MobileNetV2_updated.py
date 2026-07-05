# ==========================================================
# PART 1: IMPORTS, CONFIGURATION, DATA, CLASS WEIGHTS
# ==========================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    BatchNormalization,
    GlobalAveragePooling2D
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# -----------------------------
# CONFIG
# -----------------------------
TRAIN_DIR = "data_updated/train"
TEST_DIR = "data_updated/test"

IMG_SIZE = 96
BATCH_SIZE = 32
EPOCHS = 20

NUM_CLASSES = 7

# -----------------------------
# AUGMENTATION (REDUCED + SAFE)
# -----------------------------
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="rgb",
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    shuffle=True
)

validation_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="rgb",
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ==========================================================
# PART 2: MODEL + HEAD + COMPILE + CALLBACKS
# ==========================================================

# -----------------------------
# BASE MODEL
# -----------------------------
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False

# -----------------------------
# IMPROVED CLASSIFIER HEAD
# -----------------------------
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.Dense(
    512,
    activation="relu",
    kernel_regularizer=tf.keras.regularizers.l2(0.001)
)(x)
x = tf.keras.layers.Dropout(0.5)(x)

x = tf.keras.layers.Dense(
    256,
    activation="relu",
    kernel_regularizer=tf.keras.regularizers.l2(0.001)
)(x)
x = tf.keras.layers.Dropout(0.3)(x)

outputs = tf.keras.layers.Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)

model = tf.keras.Model(inputs=base_model.input, outputs=outputs)

# -----------------------------
# COMPILE (STAGE 1)
# -----------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# CALLBACKS
# -----------------------------
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        "best_emotion_model.keras",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=6,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=3,
        min_lr=5e-6,
        verbose=1
    )
]


# ==========================================================
# PART 3: TRAINING + FINE TUNING + EVALUATION + PLOTS
# ==========================================================

# -----------------------------
# STAGE 1: TRAIN CLASSIFIER HEAD
# -----------------------------
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    callbacks=callbacks
)

# -----------------------------
# STAGE 2: FINE TUNING
# -----------------------------
base_model.trainable = True

# Unfreeze last 50 layers only
for layer in base_model.layers[:-50]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-6),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=["accuracy"]
)

history_fine = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=30,
    callbacks=callbacks
)

# -----------------------------
# COMBINE HISTORY
# -----------------------------
acc = history.history["accuracy"] + history_fine.history["accuracy"]
val_acc = history.history["val_accuracy"] + history_fine.history["val_accuracy"]

loss = history.history["loss"] + history_fine.history["loss"]
val_loss = history.history["val_loss"] + history_fine.history["val_loss"]

# -----------------------------
# PLOTS
# -----------------------------
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(acc, label="Train Accuracy")
plt.plot(val_acc, label="Val Accuracy")
plt.title("Accuracy Curve")
plt.legend()

plt.subplot(1,2,2)
plt.plot(loss, label="Train Loss")
plt.plot(val_loss, label="Val Loss")
plt.title("Loss Curve")
plt.legend()

plt.tight_layout()
plt.savefig("training_results.png")

# -----------------------------
# SAVE FINAL MODEL
# -----------------------------
model.save("final_emotion_model.keras")

print("\nTraining Complete!")
print("Model saved as final_emotion_model.keras")