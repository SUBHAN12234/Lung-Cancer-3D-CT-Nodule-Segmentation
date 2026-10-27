# 3D Lung Nodule Segmentation with a 3D U-Net

A self-directed deep learning project that segments lung nodules in 3D CT scans. Given a CT volume around a nodule, the model predicts, voxel by voxel, which parts of the scan belong to the nodule.

Built end to end in Python with TensorFlow/Keras: data preprocessing, a patient-level train/val/test split, a `tf.data` input pipeline, a 3D U-Net, training with checkpointing and resumable runs, and evaluation on a held-out test set. Trained locally on a laptop GPU (NVIDIA RTX 2050) under WSL2.

## Why this problem

Radiologists outline lung nodules in CT scans by hand, slice by slice, which is slow and varies between readers. I chose segmentation over a yes/no classifier because it shows where the nodule is and its shape and size, taking more of that manual work off them. The model is 3D because nodules are 3D objects, and neighbouring slices help tell a nodule apart from a blood vessel.

## Results

Best model (epoch 23), evaluated on the unseen test set:

| Metric | Test score |
|---|---|
| Dice coefficient | **0.673** |
| Mean IoU | 0.693 |
| Dice loss | 0.327 |
| Voxel accuracy | 0.999 |

**A note on accuracy:** nodules take up a tiny fraction of each volume, so a model that predicts "no nodule" everywhere would already score above 99% accuracy. That's why I trained on **Dice loss** and treat the **Dice coefficient** as the main metric, since it only measures overlap with the nodule itself.

Training ran for 28 epochs. Validation Dice peaked at epoch 23 (0.680), and early stopping ended training five epochs later. The full per-epoch log, with notes, is in `All epochs of training.xlsx`.

## How it works

```
LIDC-IDRI scans ──► data_processing.py ──► generate_splits.py ──► data_loader.py ──► train.py ──► evaluate_model.py
 (PNG slices +      (3D volumes + binary     (70/20/10 split        (tf.data pipeline,   (3D U-Net,     (test-set metrics,
  radiologist        masks saved as .npy)     by patient)            fixed depth of 32)   Dice loss)      timestamped report)
  masks)
```

### 1. Data preprocessing (`src/data_processing.py`)
- **Dataset:** the public [LIDC-IDRI](https://www.cancerimagingarchive.net/collection/lidc-idri/) collection of thoracic CT scans, using per-nodule slice crops.
- Each nodule's 2D slices are resized to 128×128, normalised to [0, 1] and stacked into a 3D volume.
- Each nodule has outlines from up to **four radiologists**. I combine them (a voxel counts as nodule if any radiologist marked it) into one binary mask.
- Volumes and masks are saved as `.npy` files so training doesn't re-decode images every epoch.

### 2. Train / validation / test split (`src/generate_splits.py`)
- 70% train, 20% validation, 10% test, with a fixed random seed for reproducibility.
- The split is done **by patient, not by nodule**. One patient can have several nodules, and splitting by nodule would let the model see the same patient's scans in both training and testing, which inflates results.

### 3. Input pipeline (`src/data_loader.py`)
- Built on `tf.data`: shuffling for training, parallel loading (`AUTOTUNE`) and prefetching so the GPU isn't left waiting on disk.
- Nodules span different numbers of slices, but the model needs a fixed input size. Volumes are **padded** to 32 slices if shorter, or **centre-cropped** to 32 if longer, keeping the middle of the nodule.

### 4. Model (`src/model.py`)
A 3D U-Net with input shape `32 × 128 × 128 × 1`:
- **Encoder:** three levels (32 → 64 → 128 filters). Each level has two 3×3×3 convolutions with batch normalisation and ReLU, then max pooling to halve the resolution.
- **Bottleneck:** 256 filters.
- **Decoder:** transposed convolutions upsample back to full resolution. **Skip connections** concatenate the matching encoder features, which brings back the fine spatial detail lost during downsampling. That detail is what makes the boundaries sharp.
- **Output:** a 1×1×1 convolution with sigmoid, giving a per-voxel probability of being nodule.

### 5. Training (`src/train.py`)
- **Loss:** Dice loss, to handle the extreme class imbalance described above.
- **Optimiser:** Adam, learning rate 1e-4.
- **Batch size 1**, because 3D volumes are memory-heavy and the laptop GPU has limited VRAM.
- **Checkpointing:** the model is saved after every epoch, and separately whenever validation loss hits a new all-time best.
- **Resumable training:** a run can pick up from any saved checkpoint, so long training could be split across several sessions.
- **Early stopping** (patience 5) on validation loss, restoring the best weights.

### 6. Evaluation (`src/evaluate_model.py`)
Loads the best model, scores it on the test set, and writes the metrics to a timestamped file in `evaluation_results/`.

## What I learned

- **Pick the metric that matches the problem.** 99.9% accuracy looked great but meant very little. Learning why, and switching to Dice, was one of the biggest lessons of the project.
- **Avoid data leakage.** Splitting by patient rather than by nodule is a small detail that decides whether the test score can be trusted.
- **Engineer around hardware limits.** Batch size 1, a fixed input depth, an efficient input pipeline and resumable training were all needed to train a 3D model on a laptop GPU.
- **Read the training curves, not just the final number.** Validation Dice swung a lot between epochs (for example, it dropped to 0.11 at epoch 18 and recovered to 0.67 at epoch 20), and training Dice reached about 0.82 while validation stayed around 0.6–0.68. That points to overfitting and noise from a small validation set, which shaped the improvements below.

## Limitations and next steps

- **Overfitting:** add 3D data augmentation (flips, rotations, intensity changes, elastic deformation) and more dropout or regularisation.
- **Loss function:** try Dice + binary cross-entropy, or focal loss, which often train more stably than Dice alone.
- **Better evaluation:** report Dice and IoU per nodule on thresholded predictions, plus precision and recall, instead of only averaged voxel metrics.
- **Input handling:** resample scans to a consistent physical voxel spacing, and use sliding-window inference instead of cropping to 32 slices so large nodules aren't cut off.
- **Post-processing:** remove small false-positive blobs with connected-component analysis.
- **Engineering:** move settings out of the scripts into a config file or command-line arguments, share the duplicated Dice code in one module, and add unit tests for the preprocessing and loss functions.

## Project structure

```
├── src/
│   ├── data_processing.py   # PNG slices + radiologist masks → 3D .npy volumes and binary masks
│   ├── generate_splits.py   # patient-level 70/20/10 train/val/test split
│   ├── data_loader.py       # tf.data pipeline with depth padding/cropping
│   ├── model.py             # 3D U-Net architecture
│   ├── train.py             # training, checkpointing, resume and early stopping
│   └── evaluate_model.py    # test-set evaluation and results report
├── models/                  # trained models, including best_overall_3d_unet_model.keras (epoch 23)
├── evaluation_results/      # timestamped test-set results
├── All epochs of training.xlsx  # per-epoch training and validation metrics with notes
├── gpu.py                   # quick check that TensorFlow can see the GPU
└── requirements.txt
```

The `data/` folder is not included in the repository because of the dataset's size. Download LIDC-IDRI separately and run the preprocessing steps below.

## Running it

```bash
git clone https://github.com/SUBHAN12234/Lung-Cancer-3D-CT-Nodule-Segmentation.git
cd Lung-Cancer-3D-CT-Nodule-Segmentation
pip install -r requirements.txt
python gpu.py                      # confirm the GPU is visible

cd src
python data_processing.py          # expects raw data in data/raw/
cd ..
python src/generate_splits.py      # writes data/splits/{train,val,test}.txt
python src/train.py                # set RESUME_TRAINING in train.py to resume from a checkpoint
python src/evaluate_model.py       # scores models/best_overall_3d_unet_model.keras on the test set
```

## Tech stack

Python · TensorFlow / Keras · NumPy · Pillow · WSL2 · NVIDIA GPU (CUDA)

## Acknowledgements

CT data from the Lung Image Database Consortium and Image Database Resource Initiative (LIDC-IDRI), hosted by The Cancer Imaging Archive.
