import numpy as np
import os

# 라벨 및 이미지 파일 로드
labels = np.load("labels.npy")
image_files = sorted([f for f in os.listdir("normalized_images") if f.endswith(".npy")])

# 검증
assert len(labels) == len(image_files), "라벨과 이미지 파일 개수가 일치하지 않습니다!"
print(f"총 {len(image_files)}개의 이미지와 라벨이 준비되었습니다.")

# 라벨 값 범위 확인
print(f"라벨 값 범위: {labels.min()} ~ {labels.max()}")