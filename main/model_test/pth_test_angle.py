import torch
import torch.nn as nn
import numpy as np
import os
import cv2

# PilotNet 모델 정의
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

# 이미지 전처리 함수
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    height, _, _ = image.shape
    cropped_image = image[int(height * 0.3):, :]
    resized_image = cv2.resize(cropped_image, (200, 66))
    normalized_image = (resized_image / 255.0) * 2 - 1
    image_tensor = torch.tensor(normalized_image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    return image_tensor

# 수정된 모델 예측 함수
def predict_steering_angle(model, image_tensor, device):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        predicted_angle = model(image_tensor).item()
    
    # 각도를 30, 60, 90, 120 중 가장 가까운 값으로 매핑
    possible_angles = [30, 60, 90, 120]
    closest_angle = min(possible_angles, key=lambda x: abs(x - predicted_angle))
    return closest_angle

# GUI 함수
def gui_test_model(model_path, folder_path):
    # 모델 로드
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PilotNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 이미지 파일 리스트 수집
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.png')])
    if not image_files:
        print("No image files found in the specified folder.")
        return

    index = 0  # 현재 이미지 인덱스

    while True:
        # 현재 이미지 파일 경로
        image_path = os.path.join(folder_path, image_files[index])
        file_name = os.path.basename(image_path)

        # 파일명에서 각도 추출 (예: "20241217_012920_90_20.jpg" -> 90)
        true_angle = file_name.split('_')[2]

        # 이미지 전처리 및 모델 예측
        image_tensor = preprocess_image(image_path)
        if image_tensor is None:
            print(f"Cannot load image: {image_path}")
            continue

        predicted_angle = predict_steering_angle(model, image_tensor, device)

        # 이미지에 텍스트로 정보 추가
        image = cv2.imread(image_path)
        display_text = f"True: {true_angle} deg | Predicted: {predicted_angle:.2f} deg"
        cv2.putText(image, display_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 화면에 이미지 표시
        cv2.imshow("Model Test", image)

        # 키 입력 대기
        key = cv2.waitKey(0) & 0xFF
        if key == ord('d'):  # 다음 이미지
            index = (index + 1) % len(image_files)
        elif key == ord('a'):  # 이전 이미지
            index = (index - 1) % len(image_files)
        elif key == ord('q'):  # 종료
            break

    cv2.destroyAllWindows()

# 경로 설정
MODEL_PATH = "pilotnet.pth"  # 학습된 모델 파일 경로
FOLDER_PATH = "data/test"  # 이미지가 들어있는 폴더 경로

# GUI 실행
gui_test_model(MODEL_PATH, FOLDER_PATH)