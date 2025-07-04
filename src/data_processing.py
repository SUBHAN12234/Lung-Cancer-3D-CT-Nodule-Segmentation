import os
import numpy as np
from PIL import Image
import tensorflow as tf

IMG_SIZE = 128  # Target size for each slice

def load_slices_as_3d_volume(images_dir):
    slice_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])
    slices = []
    for slice_file in slice_files:
        img_path = os.path.join(images_dir, slice_file)
        img = Image.open(img_path).convert('L').resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0, 1]
        slices.append(img_array)
    volume_3d = np.stack(slices, axis=0)  # Shape: (num_slices, 128, 128)
    return tf.convert_to_tensor(volume_3d, dtype=tf.float32)

def load_masks_as_3d_volume(masks_dirs, num_slices):
    combined_masks = np.zeros((num_slices, IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    for mask_dir in masks_dirs:
        mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])
        for idx, mask_file in enumerate(mask_files):
            mask_path = os.path.join(mask_dir, mask_file)
            mask_img = Image.open(mask_path).convert('L').resize((IMG_SIZE, IMG_SIZE))
            mask_array = np.array(mask_img)
            combined_masks[idx] = np.maximum(combined_masks[idx], mask_array)
    combined_masks = (combined_masks > 0).astype(np.uint8)
    return tf.convert_to_tensor(combined_masks, dtype=tf.uint8)

def process_patient(patient_dir):
    nodule_folders = [f for f in os.listdir(patient_dir) if f.startswith('nodule-')]
    data = []
    labels = []
    for nodule in nodule_folders:
        nodule_path = os.path.join(patient_dir, nodule)
        images_dir = os.path.join(nodule_path, 'images')
        masks_dirs = [os.path.join(nodule_path, f"mask-{i}") for i in range(4)]

        volume = load_slices_as_3d_volume(images_dir)
        masks = load_masks_as_3d_volume(masks_dirs, volume.shape[0])

        label = 0 if nodule == 'nodule-0' else 1

        data.append((volume, masks, label))
    return data

