# train.py

import os
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import mixed_precision # For optional mixed precision
from data_loader import DataLoader
from model import build_3d_unet
import re # Added for parsing epoch from filename

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DATA_DIR = "data/processed"
SPLITS_DIR = "data/splits"
BATCH_SIZE = 1 # Keep this at 1 due to VRAM limitations
EPOCHS_PER_RUN = 20 # This defines how many epochs will be run *in each session*
TOTAL_TARGET_EPOCHS = 40 # <--- Changed: 20 initial + 20 more = 40 total epochs

MODEL_DIR = "models" # Directory for final saved model
CHECKPOINT_DIR = "checkpoints" # Directory for epoch-wise checkpoints

# NEW: Path for the absolute best model across all runs
BEST_OVERALL_MODEL_PATH = os.path.join(MODEL_DIR, "best_overall_3d_unet_model.keras")

# --- Resume Training Configuration ---
# Set to True if you want to resume training from a specific checkpoint.
# Set to False to start training from scratch.
RESUME_TRAINING = True # <--- Changed: Set to True to resume
# If RESUME_TRAINING is True, specify the path to the checkpoint file to load.
# Example: './checkpoints/model_epoch_05.keras'
LAST_CHECKPOINT_PATH = './checkpoints/model_epoch_20.keras' # <--- Changed: Point to Epoch 20
INITIAL_EPOCH = 0 # This will be 0 for a new run, or loaded from checkpoint for resume

# Create output directories if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Optional: Enable mixed precision for potential speedup and memory saving
# Your RTX 2050 (Compute Capability 8.6) supports mixed_float16.
# Uncomment the line below to enable it. Test carefully for numerical stability.
# mixed_precision.set_global_policy('mixed_float16')

# -----------------------------------------------------------------------------
# Custom Loss Functions and Metrics
# -----------------------------------------------------------------------------

def dice_coef(y_true, y_pred, smooth=1e-7):
    """
    Dice coefficient for 3D volumes.
    Args:
        y_true: Ground truth mask (binary).
        y_pred: Predicted mask (probabilities).
        smooth: Smoothing factor to prevent division by zero.
    Returns:
        Dice coefficient.
    """
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.reshape(y_pred, [-1])
    
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    """
    Dice Loss function.
    Args:
        y_true: Ground truth mask.
        y_pred: Predicted mask.
    Returns:
        Dice Loss.
    """
    return 1 - dice_coef(y_true, y_pred)


# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------
loader = DataLoader(DATA_DIR, SPLITS_DIR, batch_size=BATCH_SIZE)
train_ds = loader.get_dataset("train")
val_ds = loader.get_dataset("val")

# -----------------------------------------------------------------------------
# Model Definition or Loading
# -----------------------------------------------------------------------------
model = None # Initialize model to None
if RESUME_TRAINING and os.path.exists(LAST_CHECKPOINT_PATH):
    print(f"Resuming training from checkpoint: {LAST_CHECKPOINT_PATH}")
    # When loading a model with custom objects (like dice_loss), you need to provide them
    model = tf.keras.models.load_model(LAST_CHECKPOINT_PATH, custom_objects={'dice_loss': dice_loss, 'dice_coef': dice_coef})
    try:
        match = re.search(r'model_epoch_(\d+)\.keras', LAST_CHECKPOINT_PATH)
        if match:
            INITIAL_EPOCH = int(match.group(1))
            print(f"Training will continue from epoch {INITIAL_EPOCH + 1} (0-indexed: {INITIAL_EPOCH})")
        else:
            print("Could not parse epoch number from checkpoint path. Starting from epoch 0.")
            INITIAL_EPOCH = 0
    except Exception as e:
        print(f"Error parsing epoch from checkpoint path: {e}. Starting from epoch 0.")
        INITIAL_EPOCH = 0
else:
    print("Starting new training run.")
    model = build_3d_unet(input_shape=(32, 128, 128, 1))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4), # <--- Learning rate remains 1e-4
        loss=dice_loss,
        metrics=["accuracy", tf.keras.metrics.MeanIoU(num_classes=2), dice_coef]
    )

if model is None:
    raise ValueError("Model could not be loaded or created. Please check paths and configurations.")

model.summary()

# -----------------------------------------------------------------------------
# Callbacks
# -----------------------------------------------------------------------------
# 1. Checkpoint callback: Saves the model after *every* epoch for easy resumption
epoch_checkpoint_filepath = os.path.join(CHECKPOINT_DIR, 'model_epoch_{epoch:02d}.keras')
epoch_checkpoint_callback = ModelCheckpoint(
    filepath=epoch_checkpoint_filepath,
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=False, # Set to False to save a checkpoint after *every* epoch
    verbose=0 # Changed to 0 to reduce clutter, as the overall best one will be verbose
)

# 2. NEW: Best Overall Model Checkpoint: Saves ONLY the best model based on val_loss
best_overall_model_callback = ModelCheckpoint(
    filepath=BEST_OVERALL_MODEL_PATH,
    save_weights_only=False,
    monitor='val_loss', # Monitor validation loss
    mode='min',         # Save when val_loss is minimized
    save_best_only=True, # IMPORTANT: Only save if this is the best so far
    verbose=1           # Print messages when this model is saved
)

# Early Stopping callback: Stops training if val_loss doesn't improve for 'patience' epochs
early_stopping_callback = EarlyStopping(
    patience=5, # <--- Patience remains 5
    monitor="val_loss",
    restore_best_weights=True
)

callbacks = [
    epoch_checkpoint_callback,
    best_overall_model_callback, # ADDED: This will save the best model across all runs
    early_stopping_callback
]

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
target_epoch_for_this_run = INITIAL_EPOCH + EPOCHS_PER_RUN
final_epoch_for_fit_call = min(TOTAL_TARGET_EPOCHS, target_epoch_for_this_run)

print(f"\nStarting model training from epoch {INITIAL_EPOCH + 1} (0-indexed: {INITIAL_EPOCH})")
print(f"This run will train for {EPOCHS_PER_RUN} epochs, up to epoch {final_epoch_for_fit_call} (0-indexed: {final_epoch_for_fit_call - 1}).")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=final_epoch_for_fit_call,
    initial_epoch=INITIAL_EPOCH,
    callbacks=callbacks,
    verbose=1
)

# -----------------------------------------------------------------------------
# Final Model Save (This will save the best model from *this specific run*)
# -----------------------------------------------------------------------------
final_model_path = os.path.join(MODEL_DIR, "final_3d_unet_model.keras")
model.save(final_model_path)
print(f"\nFinal trained model from this run saved to: {final_model_path}")
print(f"The absolute best model across all runs is saved to: {BEST_OVERALL_MODEL_PATH}")

# Optional: Plotting Training History (as discussed before)
# import matplotlib.pyplot as plt
# ... (plotting code) ...
