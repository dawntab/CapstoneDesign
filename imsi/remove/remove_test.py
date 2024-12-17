import os
import cv2
import shutil
import numpy as np

# 절대 경로로 images 폴더 설정
base_dir = os.path.abspath("images")

# images 폴더가 없으면 생성
if not os.path.exists(base_dir):
    print(f"'{base_dir}' 폴더가 없습니다. 새로 생성합니다.")
    os.makedirs(base_dir)

# 하위 폴더 리스트 가져오기
sub_dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

# 모든 이미지 파일 경로를 리스트로 저장
image_paths = []
for sub_dir in sub_dirs:
    for angle_folder in os.listdir(sub_dir):
        angle_path = os.path.join(sub_dir, angle_folder)
        if os.path.isdir(angle_path):
            for file_name in os.listdir(angle_path):
                if file_name.lower().endswith((".jpg", ".png")):  # 이미지 파일만 필터링
                    image_paths.append(os.path.join(angle_path, file_name))

# 이미지 경로 정렬
image_paths.sort()

# 이미지 인덱스 초기화
current_index = 0

# 화살표 생성 함수
def draw_arrow(image, angle):
    height, width, _ = image.shape
    center = (width // 2, height // 2)
    length = 100

    # 각도 방향 반전
    corrected_angle = 180 - angle  # 30도와 150도의 방향을 교정
    end_x = int(center[0] + length * np.cos(np.radians(corrected_angle)))
    end_y = int(center[1] - length * np.sin(np.radians(corrected_angle)))
    cv2.arrowedLine(image, center, (end_x, end_y), (0, 255, 0), 5, tipLength=0.3)

# 메인 루프
while current_index < len(image_paths):
    if len(image_paths) == 0:
        print("No images found.")
        break

    # 현재 이미지 경로 가져오기
    image_path = image_paths[current_index]

    # 이미지 불러오기
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        current_index += 1
        continue

    # 파일명에서 각도 파싱
    try:
        file_name = os.path.basename(image_path)  # 파일명만 추출
        angle = int(file_name.split("_")[2])  # 두 번째 '_'로 구분된 값을 각도로 사용
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
        current_index += 1  # 다음 이미지로 이동
    elif key == ord('a'):  # 이전 이미지
        current_index = max(current_index - 1, 0)  # 이전 이미지로 이동
    elif key == ord('r'):  # 이미지 삭제
        print(f"Removing {image_path}")
        os.remove(image_path)  # 이미지 삭제
        del image_paths[current_index]  # 리스트에서 제거
        if current_index >= len(image_paths):  # 마지막 이미지를 삭제한 경우 인덱스 조정
            current_index -= 1
    elif key == 27:  # ESC 키로 종료
        print("Program manually terminated.")
        break

    # 모든 이미지를 한 번 확인한 경우 종료
    if current_index >= len(image_paths):
        print("All images have been reviewed. Exiting...")
        break

# OpenCV 창 닫기
cv2.destroyAllWindows()