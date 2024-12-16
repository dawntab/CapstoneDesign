import os
import cv2
import shutil
import numpy as np

# 최상위 폴더 경로 설정
base_dir = "images"

# 하위 폴더 리스트 가져오기
sub_dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

# 모든 이미지 파일 경로를 리스트로 저장
image_paths = []
for sub_dir in sub_dirs:
    for angle_folder in os.listdir(sub_dir):
        angle_path = os.path.join(sub_dir, angle_folder)
        if os.path.isdir(angle_path):
            for file_name in os.listdir(angle_path):
                if file_name.endswith((".jpg", ".png")):  # 이미지 파일만 필터링
                    image_paths.append(os.path.join(angle_path, file_name))

# 정렬
image_paths.sort()

# 이미지 인덱스 초기화
current_index = 0

# 화살표 생성 함수
def draw_arrow(image, angle):
    height, width, _ = image.shape
    center = (width // 2, height // 2)
    length = 100
    end_x = int(center[0] + length * np.cos(np.radians(angle)))
    end_y = int(center[1] - length * np.sin(np.radians(angle)))
    cv2.arrowedLine(image, center, (end_x, end_y), (0, 255, 0), 5, tipLength=0.3)

# 메인 루프
while True:
    if len(image_paths) == 0:
        print("No images found.")
        break

    # 현재 이미지 경로 가져오기
    image_path = image_paths[current_index]

    # 이미지 불러오기
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        continue

    # 파일명에서 각도 파싱
    try:
        file_name = os.path.basename(image_path)
        angle = int(file_name.split("_")[1])  # 파일명에서 각도를 추출한다고 가정
    except (IndexError, ValueError):
        print(f"Invalid file format: {image_path}")
        angle = 0

    # 각도 텍스트 추가
    cv2.putText(image, f"Angle: {angle}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 화살표 추가
    draw_arrow(image, angle)

    # 이미지 표시
    cv2.imshow("Image Viewer", image)

    # 키 입력 대기
    key = cv2.waitKey(0) & 0xFF

    if key == ord('d'):  # 다음 이미지
        current_index = (current_index + 1) % len(image_paths)
    elif key == ord('a'):  # 이전 이미지
        current_index = (current_index - 1) % len(image_paths)
    elif key == ord('r'):  # 이미지 삭제
        print(f"Removing {image_path}")
        os.remove(image_path)  # 이미지 삭제
        del image_paths[current_index]  # 리스트에서 제거
        if current_index >= len(image_paths):
            current_index = len(image_paths) - 1
        if len(image_paths) == 0:  # 모든 이미지 삭제 시 종료
            print("No images left.")
            break
    elif key == 27:  # ESC 키로 종료
        break

# OpenCV 창 닫기
cv2.destroyAllWindows()