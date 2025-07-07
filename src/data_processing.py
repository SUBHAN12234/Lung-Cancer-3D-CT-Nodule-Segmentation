import os
import numpy as np
from PIL import Image

IMG_SIZE = 128  # Target size for each slice

def load_slices_as_3d_volume(images_dir):
    slice_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])
    slices = []
    for slice_file in slice_files:
        img_path = os.path.join(images_dir, slice_file)
        img = Image.open(img_path).convert('L').resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0, 1]
        slices.append(img_array)
    volume_3d = np.stack(slices, axis=0)  # Shape: (depth, height, width)
    return volume_3d

def load_masks_as_3d_volume(masks_dirs, num_slices):
    combined_masks = np.zeros((num_slices, IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    for mask_dir in masks_dirs:
        mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])
        for idx, mask_file in enumerate(mask_files):
            mask_path = os.path.join(mask_dir, mask_file)
            mask_img = Image.open(mask_path).convert('L').resize((IMG_SIZE, IMG_SIZE))
            mask_array = np.array(mask_img)
            combined_masks[idx] = np.maximum(combined_masks[idx], mask_array)
    combined_masks = (combined_masks > 0).astype(np.uint8)  # Binary mask
    return combined_masks

def process_patient(raw_patient_path, processed_root_dir):
    """
    Process all nodules for a single patient and save their volume and mask as .npy files.

    Parameters:
    - raw_patient_path: Path to raw patient folder (e.g., raw_data/LIDC-IDRI-0001)
    - processed_root_dir: Path to processed folder root (e.g., processed/)
    """
    patient_id = os.path.basename(raw_patient_path)
    output_patient_dir = os.path.join(processed_root_dir, patient_id)
    os.makedirs(output_patient_dir, exist_ok=True)

    nodule_folders = [f for f in os.listdir(raw_patient_path) if f.startswith('nodule-')]

    for nodule in nodule_folders:
        nodule_path = os.path.join(raw_patient_path, nodule)
        images_dir = os.path.join(nodule_path, 'images')
        masks_dirs = [os.path.join(nodule_path, f"mask-{i}") for i in range(4)]

        volume = load_slices_as_3d_volume(images_dir)
        mask = load_masks_as_3d_volume(masks_dirs, volume.shape[0])

        np.save(os.path.join(output_patient_dir, f"{nodule}_volume.npy"), volume)
        np.save(os.path.join(output_patient_dir, f"{nodule}_mask.npy"), mask)
