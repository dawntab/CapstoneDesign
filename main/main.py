from data_preparation import *
from data_preprocessing import *
from data_split import *
import os

def main():
    BASE_DIR = "data"
    INPUT_DIR = os.path.join(BASE_DIR, "input")
    OUTPUT_DIR = os.path.join(BASE_DIR, "output")
    PROCESSED_IMAGES_DIR = os.path.join(OUTPUT_DIR, "processed_images")
    NORMALIZED_IMAGES_DIR = os.path.join(OUTPUT_DIR, "normalized_images")
    LABELS_DIR = os.path.join(OUTPUT_DIR, "labels")

    setup_folder(BASE_DIR)
    setup_folder(INPUT_DIR)
    setup_folder(OUTPUT_DIR)
    setup_folder(PROCESSED_IMAGES_DIR)
    setup_folder(NORMALIZED_IMAGES_DIR)
    setup_folder(LABELS_DIR)

    print("Preparing data...")
    image_paths = get_all_image_paths(INPUT_DIR)
    if not image_paths:
        print(f"Error: Input directory '{INPUT_DIR}' is empty. Please add images.")
        return

    review_and_filter_images(image_paths)

    print("Processing images...")
    if not os.listdir(INPUT_DIR):
        print(f"Error: Input directory '{INPUT_DIR}' is empty. Cannot process images.")
        return
    crop_and_resize_images(INPUT_DIR, PROCESSED_IMAGES_DIR)

    if not os.listdir(PROCESSED_IMAGES_DIR):
        print(f"Error: Processed images directory '{PROCESSED_IMAGES_DIR}' is empty.")
        return
    normalize_images(PROCESSED_IMAGES_DIR, NORMALIZED_IMAGES_DIR)

    print("Generating labels...")
    if not os.listdir(NORMALIZED_IMAGES_DIR):
        print(f"Error: Normalized images directory '{NORMALIZED_IMAGES_DIR}' is empty.")
        return
    generate_labels(NORMALIZED_IMAGES_DIR, os.path.join(LABELS_DIR, "labels.npy"))

    if not os.path.exists(os.path.join(LABELS_DIR, "labels.npy")):
        print(f"Labels file not found: {os.path.join(LABELS_DIR, 'labels.npy')}. Exiting.")
        return

    print("Splitting data...")
    split_data(os.path.join(LABELS_DIR, "labels.npy"), NORMALIZED_IMAGES_DIR, LABELS_DIR)

    print("Data preparation completed successfully.")

if __name__ == "__main__":
    main()