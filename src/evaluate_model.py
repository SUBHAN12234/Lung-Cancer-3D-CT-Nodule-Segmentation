# evaluate_model.py

import tensorflow as tf
import os
from data_loader import DataLoader
import datetime # Added for timestamping output file

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DATA_DIR = "data/processed"
SPLITS_DIR = "data/splits"
BATCH_SIZE = 1 # Use the same batch size as training for consistency, or adjust based on memory
MODEL_TO_EVALUATE_PATH = os.path.join("models", "best_overall_3d_unet_model.keras") # Path to your best model
RESULTS_DIR = "evaluation_results" # Directory to save evaluation results
os.makedirs(RESULTS_DIR, exist_ok=True) # Create results directory if it doesn't exist

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
        # When loading, custom_objects should map names to the functions/classes
        # that Keras needs to reconstruct the model.
        # For MeanIoU, passing the class itself is usually sufficient for Keras to re-instantiate it.
        model = tf.keras.models.load_model(
            MODEL_TO_EVALUATE_PATH,
            custom_objects={
                'dice_loss': dice_loss,
                'dice_coef': dice_coef,
                'MeanIoU': tf.keras.metrics.MeanIoU # Pass the class, not an instance
            }
        )
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Ensure custom objects (dice_loss, dice_coef, MeanIoU) are correctly defined and passed.")
        return

    # Load the test dataset
    loader = DataLoader(DATA_DIR, SPLITS_DIR, batch_size=BATCH_SIZE)
    test_ds = loader.get_dataset("test") # Get the test dataset
    print("Test dataset loaded.")

    # Evaluate the model
    print("\nEvaluating model on test dataset...")
    results = model.evaluate(test_ds, verbose=1)

    # Prepare results for saving
    results_output = []
    results_output.append("\n--- Test Evaluation Results ---")

    # Attempt to get metric names from the model, but provide a fallback if needed
    # The order of metrics in `results` should typically match the order in `model.metrics_names`
    # and the order they were compiled in train.py: loss, accuracy, mean_io_u, dice_coef
    metric_names = model.metrics_names
    
    # Check if the number of names matches the number of results
    if len(metric_names) == len(results):
        for name, value in zip(metric_names, results):
            results_output.append(f"{name}: {value:.4f}")
    else:
        # Fallback for unexpected number of results or names (like the 'compile_metrics' issue)
        # We'll assume the standard order if we have 4 results: loss, accuracy, mean_io_u, dice_coef
        if len(results) == 4:
            results_output.append(f"loss: {results[0]:.4f}")
            results_output.append(f"accuracy: {results[1]:.4f}")
            results_output.append(f"mean_io_u: {results[2]:.4f}")
            results_output.append(f"dice_coef: {results[3]:.4f}")
            results_output.append("\nNote: Model's reported metrics_names were unexpected, assumed standard order for output.")
            results_output.append(f"Model's reported metrics_names: {metric_names}")
        else:
            # General fallback if structure is completely unexpected
            results_output.append("Could not parse all metrics as expected. Raw output:")
            for i, value in enumerate(results):
                results_output.append(f"Metric {i} (name: {metric_names[i] if i < len(metric_names) else 'N/A'}): {value:.4f}")
            results_output.append(f"Model's reported metrics_names: {metric_names}")
            results_output.append(f"Raw results array: {results}")

    # Generate a timestamped filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = os.path.join(RESULTS_DIR, f"test_results_{timestamp}.txt")

    # Save results to file
    with open(output_filename, 'w') as f:
        for line in results_output:
            f.write(line + "\n")
    
    print(f"\nResults saved to: {output_filename}")

    # Also print to terminal for immediate viewing
    for line in results_output:
        print(line)

if __name__ == "__main__":
    main()
