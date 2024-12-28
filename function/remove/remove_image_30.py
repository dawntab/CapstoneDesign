import os
import cv2

# 입력 및 출력 폴더 경로 설정
input_base_dir = os.path.abspath("input_images")
output_base_dir = os.path.abspath("output_images")

# 입력 폴더 확인
if not os.path.exists(input_base_dir):
    print(f"'{input_base_dir}' 폴더가 없습니다. 새로 생성합니다.")
    os.makedirs(input_base_dir)

# 출력 폴더 확인 및 생성
if not os.path.exists(output_base_dir):
    print(f"'{output_base_dir}' 폴더가 없습니다. 새로 생성합니다.")
    os.makedirs(output_base_dir)

# 입력 폴더 내 하위 폴더 리스트 가져오기
sub_dirs = [os.path.join(input_base_dir, d) for d in os.listdir(input_base_dir) if os.path.isdir(os.path.join(input_base_dir, d))]

# 모든 이미지 파일 경로를 리스트로 저장
image_paths = []
for sub_dir in sub_dirs:
    for angle_folder in os.listdir(sub_dir):
        angle_path = os.path.join(sub_dir, angle_folder)
        if os.path.isdir(angle_path):
            for file_name in os.listdir(angle_path):
                if file_name.lower().endswith((".jpg", ".png")):  # 이미지 파일만 필터링
                    image_paths.append(os.path.join(angle_path, file_name))

# 이미지 처리
for image_path in image_paths:
    # 이미지 불러오기
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        continue

    # 이미지의 위쪽 30% 잘라내기
    height, width, _ = image.shape
    crop_start = int(height * 0.3)
    cropped_image = image[crop_start:, :]

    # 이미지 리사이즈 (66 x 200)
    resized_image = cv2.resize(cropped_image, (200, 66))

    # 출력 경로 설정
    output_dir = os.path.join(output_base_dir, "processed_images")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, os.path.basename(image_path))

    # 수정된 이미지 저장
    cv2.imwrite(output_path, resized_image)
    print(f"Saved resized image to: {output_path}")

print("All images have been cropped, resized, and saved.")