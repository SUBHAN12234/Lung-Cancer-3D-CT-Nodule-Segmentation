# train.py

import os
import tensorflow as tf
from data_loader import DataLoader
from model import build_3d_unet  # ✅ match your actual function name

# -----------------------------
# Config
# -----------------------------
DATA_DIR = "data/processed"
SPLITS_DIR = "data/splits"
BATCH_SIZE = 2
EPOCHS = 1
MODEL_DIR = "models"
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "unet_3d.h5")

# Create output directory
os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------
# Data
# -----------------------------
loader = DataLoader(DATA_DIR, SPLITS_DIR, batch_size=BATCH_SIZE)
train_ds = loader.get_dataset("train")
val_ds = loader.get_dataset("val")

# -----------------------------
# Model
# -----------------------------
model = build_3d_unet(input_shape=(32, 128, 128, 1))
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.MeanIoU(num_classes=2)]
)

# -----------------------------
# Callbacks
# -----------------------------
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_SAVE_PATH,
        save_best_only=True,
        monitor="val_loss",
        mode="min"
    ),
    tf.keras.callbacks.EarlyStopping(
        patience=5,
        monitor="val_loss",
        restore_best_weights=True
    )
]

# -----------------------------
# Training
# -----------------------------
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1 
)
