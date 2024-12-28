import os
import cv2
import numpy as np

def setup_folder(folder_path):
    """폴더가 없으면 생성"""
    if not os.path.exists(folder_path):
        print(f"'{folder_path}' 폴더가 없습니다. 새로 생성합니다.")
        os.makedirs(folder_path)

def crop_and_resize_images(input_dir, output_dir, resize_dim=(200, 66), crop_ratio=0.3):
    """이미지 크롭 및 리사이즈"""
    setup_folder(output_dir)  # 출력 폴더가 없으면 생성
    for file_name in os.listdir(input_dir):
        if file_name.lower().endswith((".jpg", ".png")):  # 지원하는 파일 확장자 확인
            input_path = os.path.join(input_dir, file_name)
            image = cv2.imread(input_path)
            if image is None:
                print(f"이미지를 로드할 수 없습니다: {input_path}")
                continue
            height, _, _ = image.shape
            cropped_image = image[int(height * crop_ratio):, :]  # 상단 crop_ratio% 크롭
            resized_image = cv2.resize(cropped_image, resize_dim)  # 리사이즈
            output_path = os.path.join(output_dir, file_name)
            cv2.imwrite(output_path, resized_image)  # 결과 저장

def normalize_image(image):
    """이미지 정규화"""
    return (image / 255.0) * 2 - 1

def normalize_images(input_dir, output_dir, supported_extensions=(".jpg", ".png")):
    """정규화된 이미지 저장"""
    setup_folder(output_dir)  # 출력 폴더 생성
    if not os.listdir(input_dir):  # 입력 폴더가 비어 있는지 확인
        print(f"오류: 입력 폴더 '{input_dir}'가 비어 있습니다.")
        return

    for file_name in os.listdir(input_dir):
        if file_name.lower().endswith(supported_extensions):  # 지원 확장자 확인
            input_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir, file_name.replace(".jpg", ".npy").replace(".png", ".npy"))
            image = cv2.imread(input_path)
            if image is None:
                print(f"이미지를 로드할 수 없습니다: {input_path}")
                continue
            normalized_image = normalize_image(image)  # 이미지 정규화
            np.save(output_path, normalized_image)  # 결과 저장
            print(f"정규화된 이미지를 저장했습니다: {output_path}")