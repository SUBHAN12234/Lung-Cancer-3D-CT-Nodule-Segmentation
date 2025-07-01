import os
import random

RAW_DATA_DIR = "data/raw"
SPLIT_DIR = "data/splits"

os.makedirs(SPLIT_DIR, exist_ok=True)

all_patients = sorted([d for d in os.listdir(RAW_DATA_DIR) if d.startswith("LIDC-IDRI")])
random.seed(42)
random.shuffle(all_patients)

num_total = len(all_patients)
num_train = int(0.7 * num_total)
num_val = int(0.2 * num_total)

train = all_patients[:num_train]
val = all_patients[num_train:num_train + num_val]
test = all_patients[num_train + num_val:]

with open(os.path.join(SPLIT_DIR, "train.txt"), "w") as f:
    f.write("\n".join(train))

with open(os.path.join(SPLIT_DIR, "val.txt"), "w") as f:
    f.write("\n".join(val))

with open(os.path.join(SPLIT_DIR, "test.txt"), "w") as f:
    f.write("\n".join(test))
