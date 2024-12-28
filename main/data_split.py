import os
import numpy as np
from sklearn.model_selection import train_test_split

def generate_labels(input_dir, output_file):
    """라벨 생성 및 저장"""
    if not os.path.exists(input_dir) or not os.listdir(input_dir):
        print(f"Error: Normalized images directory '{input_dir}' does not exist or is empty.")
        return

    labels = []
    for file_name in sorted(os.listdir(input_dir)):
        if file_name.lower().endswith(".npy"):
            try:
                angle = int(file_name.split("_")[2].split(".")[0])  # 예: *_*_angle.npy → angle
                labels.append(angle)
            except (IndexError, ValueError):
                print(f"Invalid file format: {file_name}")

    if labels:
        np.save(output_file, np.array(labels))
        print(f"Saved labels to: {output_file}")
    else:
        print(f"No valid labels were generated from '{input_dir}'.")

def split_data(labels_path, images_path, output_dir):
    """Train/Validation/Test 데이터 분리"""
    if not os.path.exists(labels_path) or not os.path.exists(images_path):
        raise FileNotFoundError(f"Labels or images path does not exist: {labels_path}, {images_path}")

    labels = np.load(labels_path)
    image_files = np.array(sorted([f for f in os.listdir(images_path) if f.endswith(".npy")]))

    if len(labels) != len(image_files):
        raise ValueError("Number of labels and image files do not match.")

    train_images, temp_images, train_labels, temp_labels = train_test_split(
        image_files, labels, test_size=0.2, random_state=42)
    val_images, test_images, val_labels, test_labels = train_test_split(
        temp_images, temp_labels, test_size=0.5, random_state=42)

    np.save(os.path.join(output_dir, "train_images.npy"), train_images)
    np.save(os.path.join(output_dir, "val_images.npy"), val_images)
    np.save(os.path.join(output_dir, "test_images.npy"), test_images)
    np.save(os.path.join(output_dir, "train_labels.npy"), train_labels)
    np.save(os.path.join(output_dir, "val_labels.npy"), val_labels)
    np.save(os.path.join(output_dir, "test_labels.npy"), test_labels)