# evaluate_model.py

import tensorflow as tf
import os
from data_loader import DataLoader # Assuming data_loader.py is in the same directory

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DATA_DIR = "data/processed"
SPLITS_DIR = "data/splits"
BATCH_SIZE = 1 # Use the same batch size as training for consistency, or adjust based on memory
MODEL_TO_EVALUATE_PATH = os.path.join("models", "best_overall_3d_unet_model.keras") # Path to your best model

# -----------------------------------------------------------------------------
# Custom Loss Functions and Metrics (MUST be defined for model loading)
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
# Main Evaluation Logic
# -----------------------------------------------------------------------------

def main():
    # Load the model
    print(f"Loading model from: {MODEL_TO_EVALUATE_PATH}")
    if not os.path.exists(MODEL_TO_EVALUATE_PATH):
        print(f"Error: Model file not found at {MODEL_TO_EVALUATE_PATH}")
        print("Please ensure the model exists or update the MODEL_TO_EVALUATE_PATH.")
        return

    try:
        model = tf.keras.models.load_model(
            MODEL_TO_EVALUATE_PATH,
            custom_objects={'dice_loss': dice_loss, 'dice_coef': dice_coef, 'MeanIoU': tf.keras.metrics.MeanIoU}
            # Note: MeanIoU needs to be passed if it's part of the saved model's metrics
        )
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Ensure custom objects (dice_loss, dice_coef) are correctly defined and passed.")
        return

    # Load the test dataset
    loader = DataLoader(DATA_DIR, SPLITS_DIR, batch_size=BATCH_SIZE)
    test_ds = loader.get_dataset("test") # Get the test dataset
    print("Test dataset loaded.")

    # Evaluate the model
    print("\nEvaluating model on test dataset...")
    # The verbose=1 will show a progress bar
    results = model.evaluate(test_ds, verbose=1)

    # Print the results
    print("\n--- Test Evaluation Results ---")
    for name, value in zip(model.metrics_names, results):
        print(f"{name}: {value:.4f}")

if __name__ == "__main__":
    main()
