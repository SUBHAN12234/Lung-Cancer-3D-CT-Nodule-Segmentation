import os
import numpy as np
import tensorflow as tf

class DataLoader:
    def __init__(self, data_dir, splits_dir, batch_size=4, shuffle_buffer=100):
        self.data_dir = data_dir
        self.splits_dir = splits_dir
        self.batch_size = batch_size
        self.shuffle_buffer = shuffle_buffer
        self.target_depth = 32  # Fixed depth for 3D volumes

    def read_split_file(self, split_name):
        split_path = os.path.join(self.splits_dir, f"{split_name}.txt")
        with open(split_path, 'r') as f:
            patient_ids = [line.strip() for line in f.readlines()]
        return patient_ids

    def get_patient_paths(self, patient_id):
        patient_dir = os.path.join(self.data_dir, patient_id)
        pairs = []
        for fname in os.listdir(patient_dir):
            if fname.endswith("_volume.npy"):
                base = fname.replace("_volume.npy", "")
                volume_path = os.path.join(patient_dir, f"{base}_volume.npy")
                mask_path = os.path.join(patient_dir, f"{base}_mask.npy")
                if os.path.exists(mask_path):
                    pairs.append((volume_path, mask_path))
        return pairs

    def gather_all_samples(self, split_name):
        volume_paths = []
        mask_paths = []
        for pid in self.read_split_file(split_name):
            pairs = self.get_patient_paths(pid)
            for vol_path, mask_path in pairs:
                volume_paths.append(vol_path)
                mask_paths.append(mask_path)
        return volume_paths, mask_paths

    def fix_depth(self, volume, mask):
        current_depth = volume.shape[0]
        target = self.target_depth
        if current_depth < target:
            pad_amt = target - current_depth
            pad_width = ((0, pad_amt), (0, 0), (0, 0))
            volume = np.pad(volume, pad_width, mode='constant')
            mask = np.pad(mask, pad_width, mode='constant')
        elif current_depth > target:
            start = (current_depth - target) // 2
            volume = volume[start:start + target]
            mask = mask[start:start + target]
        return volume, mask

    def load_volume_and_mask(self, volume_path, mask_path):
        volume = np.load(volume_path.decode('utf-8'))
        mask = np.load(mask_path.decode('utf-8'))
        volume, mask = self.fix_depth(volume, mask)
        return volume.astype(np.float32), mask.astype(np.uint8)

    def tf_load_volume_and_mask(self, volume_path, mask_path):
        volume, mask = tf.numpy_function(
            self.load_volume_and_mask, [volume_path, mask_path], [tf.float32, tf.uint8]
        )
        volume.set_shape([self.target_depth, 128, 128])
        mask.set_shape([self.target_depth, 128, 128])
        return volume, mask

    def get_dataset(self, split_name):
        volume_paths, mask_paths = self.gather_all_samples(split_name)
        volume_ds = tf.data.Dataset.from_tensor_slices(volume_paths)
        mask_ds = tf.data.Dataset.from_tensor_slices(mask_paths)
        dataset = tf.data.Dataset.zip((volume_ds, mask_ds))
        if split_name == 'train':
            dataset = dataset.shuffle(self.shuffle_buffer)
        dataset = dataset.map(self.tf_load_volume_and_mask, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset
