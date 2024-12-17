import torch
import torch.nn as nn
import numpy as np
import os
import cv2

# PilotNet 모델 정의 (학습할 때 사용한 모델 구조와 동일해야 함)
class PilotNet(nn.Module):
    def __init__(self):
        super(PilotNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 1 * 18, 1164), nn.ReLU(),
            nn.Linear(1164, 100), nn.ReLU(),
            nn.Linear(100, 50), nn.ReLU(),
            nn.Linear(50, 10), nn.ReLU(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        return self.model(x)

# 테스트용 이미지 전처리 함수
def preprocess_image(image_path):
    """이미지를 불러와서 정규화하고 Tensor 형태로 변환"""
    image = cv2.imread(image_path)  # 이미지 읽기
    if image is None:
        print(f"Error: Cannot load image from {image_path}")
        return None

    height, _, _ = image.shape
    cropped_image = image[int(height * 0.3):, :]  # 상단 30% 크롭
    resized_image = cv2.resize(cropped_image, (200, 66))  # 리사이즈
    normalized_image = (resized_image / 255.0) * 2 - 1  # 정규화
    image_tensor = torch.tensor(normalized_image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    return image_tensor

# 테스트 함수
def test_model(model_path, test_image_path):
    """학습된 모델을 불러와서 이미지에서 각도 예측"""
    # 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 불러오기
    model = PilotNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 이미지 전처리
    image_tensor = preprocess_image(test_image_path)
    if image_tensor is None:
        return

    # 예측
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        predicted_angle = model(image_tensor)
        print(f"Predicted Steering Angle: {predicted_angle.item():.2f} degrees")

# 테스트 이미지와 모델 경로 설정
MODEL_PATH = "pilotnet.pth"  # 학습된 모델 파일 경로
TEST_IMAGE_PATH = "data/test/sample_image.jpg"  # 테스트할 이미지 파일 경로

# 모델 테스트 실행
test_model(MODEL_PATH, TEST_IMAGE_PATH)