

```markdown
# Lung-Cancer-3D-CT-Nodule-Segmentation

3D Lung Nodule Segmentation using 3D U-Net

This project focuses on developing a deep learning model for the **accurate segmentation of lung nodules from 3D Computed Tomography (CT) scans**. Early and precise detection of lung nodules is crucial for the timely diagnosis of lung cancer, significantly improving patient outcomes. This repository provides the code, trained models, and documentation for a 3D U-Net architecture implemented using **TensorFlow and Keras**.

---

## Overview & Problem Statement

Lung cancer remains one of the leading causes of cancer-related deaths worldwide. CT scans are widely used for screening and diagnosis, but manually identifying and delineating small lung nodules is a time-consuming and error-prone task for radiologists.

This project addresses this by providing **automated segmentation methods** that can significantly assist in this process. Our approach offers consistent and rapid analysis, thereby enhancing diagnostic accuracy and efficiency.

---

## Model Architecture: 3D U-Net

The core of this project is a **3D U-Net**, a convolutional neural network architecture renowned for its effectiveness in biomedical image segmentation. The 3D U-Net extends the original 2D U-Net by processing volumetric data directly, allowing it to capture spatial context across slices, which is vital for 3D medical images like CT scans.

The architecture consists of:

* **Encoder (Contracting Path):** Downsamples the input CT volume through a series of 3D convolutional layers and max-pooling operations, extracting hierarchical features.
* **Decoder (Expansive Path):** Upsamples the feature maps, combining them with high-resolution features from the encoder via skip connections. These skip connections are crucial for propagating fine-grained spatial information, enabling precise localization of nodules.
* **Output Layer:** A final 3D convolutional layer with a sigmoid activation function outputs a probability map, indicating the likelihood of each voxel belonging to a lung nodule.

---

## Dataset

This project utilizes a subset of the **Lung Image Database Consortium (LIDC-IDRI) dataset**. The LIDC-IDRI dataset is a publicly available resource containing thoracic CT scans with marked-up annotated lesions by multiple radiologists. The data undergoes preprocessing steps to ensure consistency and suitability for 3D deep learning models.

---

## Key Technologies Used

* **Python:** The primary programming language.
* **TensorFlow & Keras:** Deep learning framework for building, training, and evaluating the 3D U-Net model.
* **NumPy:** For numerical operations and data handling.
* **WSL2 (Windows Subsystem for Linux 2) & NVIDIA GPU:** For accelerated training and inference.
* **Git & GitHub:** For version control and project hosting.

---

## Performance

The model was trained for a total of **28 epochs** (initially 20, then resumed for 8 more until EarlyStopping triggered). The best performing model, identified at Epoch 23, demonstrated strong generalization capabilities on the unseen test set.

Here are the final evaluation metrics on the test dataset for the best overall model (Epoch 23):

```

loss: 0.3272
accuracy: 0.9993
mean\_io\_u: 0.6926
dice\_coef: 0.6728

```

These results indicate a robust performance in segmenting lung nodules, achieving a high degree of overlap with the ground truth annotations.

---

## Project Structure

The repository is organized as follows:

```

Lung-Cancer-3D-CT-Nodule-Segmentation/
├── data/
│   ├── processed/          \# Preprocessed CT scans and masks (e.g., .npy files)
│   └── splits/             \# Files defining train, validation, and test splits
├── models/
│   ├── best\_overall\_3d\_unet\_model.keras  \# The absolute best model saved across all runs (Epoch 23)
│   ├── final\_3d\_unet\_model.keras         \# Best model from the last training session (also Epoch 23)
│   └── final\_3d\_unet\_model\_epoch\_20\_best.keras \# (Your manual backup of Epoch 20)
├── checkpoints/            \# Saved model checkpoints from each epoch during training
│   ├── model\_epoch\_01.keras
│   ├── ...
│   └── model\_epoch\_28.keras
├── evaluation\_results/     \# Directory to store test evaluation results
│   └── test\_results\_YYYYMMDD\_HHMMSS.txt
├── src/
│   ├── data\_loader.py      \# Script for loading and preprocessing data
│   ├── model.py            \# Defines the 3D U-Net model architecture
│   ├── train.py            \# Script for training the model
│   └── evaluate\_model.py   \# Script for evaluating the trained model on test data
├── .gitignore              \# Specifies files/directories to ignore in Git (e.g., checkpoints/)
└── README.md               \# This file

````

---

## Setup and Installation

### 1. Clone the Repository:

```bash
git clone [https://github.com/SUBHAN12234/Lung-Cancer-3D-CT-Nodule-Segmentation.git](https://github.com/SUBHAN12234/Lung-Cancer-3D-CT-Nodule-Segmentation.git)
cd Lung-Cancer-3D-CT-Nodule-Segmentation
````

### 2\. Set up Conda Environment (Recommended for GPU support):

Ensure you have Miniconda or Anaconda installed.

```bash
conda create -n tf-gpu python=3.9
conda activate tf-gpu
pip install tensorflow[and-cuda]==2.15.0 # Or your specific TF/CUDA version
pip install numpy pandas scikit-learn matplotlib
```

**Note:** Adjust `tensorflow[and-cuda]==2.15.0` based on your CUDA/cuDNN setup if necessary.

### 3\. Prepare Data:

  * Obtain the LIDC-IDRI dataset.
  * Run your data preprocessing script (if you have one, or manually place processed `.npy` files into `data/processed` and split definitions into `data/splits` as per your `data_loader.py`).

-----

## Usage

### Training the Model

To start or resume training the 3D U-Net model:

1.  **Configure `src/train.py`:**

      * Adjust `TOTAL_TARGET_EPOCHS` as desired.
      * Set `RESUME_TRAINING = True` and `LAST_CHECKPOINT_PATH` to the `.keras` file of the epoch you wish to resume from (e.g., `./checkpoints/model_epoch_20.keras`).

2.  **Run Training:**

    ```bash
    conda activate tf-gpu
    cd src/
    python train.py
    ```

The training process will save epoch checkpoints in `checkpoints/` and the best model from the run (based on `val_loss`) to `models/final_3d_unet_model.keras`. The `models/best_overall_3d_unet_model.keras` will also be updated if a new all-time best `val_loss` is achieved.

### Evaluating the Model

To evaluate the best trained model on the unseen test dataset:

1.  **Configure `src/evaluate_model.py`:**

      * Ensure `MODEL_TO_EVALUATE_PATH` points to the model you want to test (e.g., `models/best_overall_3d_unet_model.keras`).

2.  **Run Evaluation:**

    ```bash
    conda activate tf-gpu
    cd src/
    python evaluate_model.py
    ```

The results will be printed to the console and saved to a timestamped text file in the `evaluation_results/` directory.

-----

## Future Work

  * **Advanced Data Augmentation:** Implement more sophisticated 3D data augmentation techniques (e.g., elastic deformations, intensity shifts) to improve model robustness.
  * **Hyperparameter Optimization:** Conduct a more thorough search for optimal learning rates, batch sizes, and network configurations.
  * **Loss Function Exploration:** Experiment with hybrid loss functions (e.g., Dice + BCE) or focal loss to address class imbalance more effectively.
  * **Model Ensembling:** Combine predictions from multiple trained models to potentially achieve higher overall performance.
  * **Post-processing:** Implement post-processing techniques on the segmentation masks (e.g., connected component analysis, morphological operations) to refine predictions.
  * **Quantification & Visualization:** Develop tools for quantitative analysis of nodule characteristics and 3D visualization of segmentation results.

-----

## Acknowledgements

  * **LIDC-IDRI Dataset:** This project utilizes data from the Lung Image Database Consortium and Image Database Resource Initiative (LIDC-IDRI) public database.
  * **TensorFlow & Keras Teams:** For providing powerful and flexible deep learning frameworks.

<!-- end list -->

```
```
