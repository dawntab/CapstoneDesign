import os
import cv2
import numpy as np

def normalize_image(image):
    """
    이미지를 [-1, 1] 범위로 정규화
    """
    return (image / 255.0) * 2 - 1

def process_images(input_dir, output_dir):
    """
    입력 폴더의 이미지를 정규화하여 출력 폴더에 저장
    """
    # 입력 및 출력 폴더 생성
    if not os.path.exists(input_dir):
        print(f"입력 폴더가 없습니다. 새로 생성합니다: {input_dir}")
        os.makedirs(input_dir)
    if not os.path.exists(output_dir):
        print(f"출력 폴더가 없습니다. 새로 생성합니다: {output_dir}")
        os.makedirs(output_dir)

    # 이미지 파일 처리
    for file_name in os.listdir(input_dir):
        if file_name.lower().endswith((".jpg", ".png")):  # 이미지 파일만 처리
            input_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir, file_name.replace(".jpg", ".npy").replace(".png", ".npy"))

            # 이미지 로드
            image = cv2.imread(input_path)
            if image is None:
                print(f"이미지를 로드할 수 없습니다: {input_path}")
                continue

            # 이미지 정규화
            normalized_image = normalize_image(image)

            # 정규화된 이미지 저장
            np.save(output_path, normalized_image)
            print(f"정규화된 이미지 저장: {output_path}")

# 입력 및 출력 폴더 경로 설정
input_folder = "resized_images"  # 리사이징된 이미지가 저장된 폴더
output_folder = "normalized_images"  # 정규화된 이미지를 저장할 폴더

# 함수 실행
process_images(input_folder, output_folder)