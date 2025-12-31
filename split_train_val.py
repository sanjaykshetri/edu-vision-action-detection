import os
import random
import shutil

CLASSES = ["hand_raised", "writing", "looking_board", "device_use"]
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
VAL_RATIO = 0.2

random.seed(42)

def split_class(class_name):
    train_path = os.path.join(TRAIN_DIR, class_name)
    val_path = os.path.join(VAL_DIR, class_name)
    os.makedirs(val_path, exist_ok=True)
    images = [f for f in os.listdir(train_path) if f.lower().endswith('.jpg')]
    n_val = int(len(images) * VAL_RATIO)
    val_images = random.sample(images, n_val)
    for img in val_images:
        src = os.path.join(train_path, img)
        dst = os.path.join(val_path, img)
        shutil.move(src, dst)
        print(f"Moved {img} to {val_path}")

def main():
    for class_name in CLASSES:
        split_class(class_name)
    print("Train/val split complete.")

if __name__ == "__main__":
    main()
