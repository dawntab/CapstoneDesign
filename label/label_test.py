import os
import numpy as np

# 입력 폴더와 출력 파일 경로 설정
input_dir = "normalized_images"
output_file = "labels.npy"

# 입력 폴더 생성 (없을 경우 생성)
if not os.path.exists(input_dir):
    print(f"입력 폴더가 없습니다. 새로 생성합니다: {input_dir}")
    os.makedirs(input_dir)

# 라벨 리스트 초기화
labels = []

# 입력 폴더 내 파일 처리
if len(os.listdir(input_dir)) == 0:
    print(f"입력 폴더({input_dir})에 파일이 없습니다. 데이터를 추가하세요.")
else:
    for file_name in sorted(os.listdir(input_dir)):
        if file_name.lower().endswith(".npy"):  # .npy 파일만 처리
            try:
                # 파일명에서 조향각 추출 (예: "20241217_013217_60_30.npy" → 30)
                angle = int(file_name.split("_")[3].split(".")[0])
                labels.append(angle)
            except (IndexError, ValueError):
                print(f"잘못된 파일 형식: {file_name}")

    # 라벨 저장
    if labels:
        labels = np.array(labels)
        np.save(output_file, labels)
        print(f"라벨이 저장되었습니다: {output_file}")
        print(f"총 {len(labels)}개의 라벨이 저장되었습니다.")
    else:
        print("처리할 유효한 파일이 없습니다.")