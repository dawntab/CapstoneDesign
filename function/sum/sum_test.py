import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# **폴더 설정**
BASE_DIR = "data"
INPUT_DIR = os.path.join(BASE_DIR, "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

PROCESSED_IMAGES_DIR = os.path.join(OUTPUT_DIR, "processed_images")
NORMALIZED_IMAGES_DIR = os.path.join(OUTPUT_DIR, "normalized_images")
LABELS_DIR = os.path.join(OUTPUT_DIR, "labels")

# **폴더 생성 함수**
def setup_folder(folder_path):
    """폴더가 없으면 생성"""
    if not os.path.exists(folder_path):
        print(f"'{folder_path}' 폴더가 없습니다. 새로 생성합니다.")
        os.makedirs(folder_path)

# **이미지 처리 함수**
def get_all_image_paths(base_dir):
    """모든 이미지 파일 경로 가져오기"""
    sub_dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    image_paths = []
    for sub_dir in sub_dirs:
        for angle_folder in os.listdir(sub_dir):
            angle_path = os.path.join(sub_dir, angle_folder)
            if os.path.isdir(angle_path):
                for file_name in os.listdir(angle_path):
                    if file_name.lower().endswith((".jpg", ".png")):
                        image_paths.append(os.path.join(angle_path, file_name))
    return sorted(image_paths)

def crop_and_resize_images(input_dir, output_dir):
    """이미지 자르기 및 리사이즈"""
    image_paths = get_all_image_paths(input_dir)
    setup_folder(output_dir)
    for image_path in image_paths:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error loading image: {image_path}")
            continue
        height, _, _ = image.shape
        cropped_image = image[int(height * 0.3):, :]
        resized_image = cv2.resize(cropped_image, (200, 66))
        output_path = os.path.join(output_dir, os.path.basename(image_path))
        cv2.imwrite(output_path, resized_image)
        print(f"Saved resized image to: {output_path}")

def normalize_image(image):
    """이미지 정규화"""
    return (image / 255.0) * 2 - 1

def normalize_images(input_dir, output_dir):
    """정규화된 이미지 저장"""
    setup_folder(output_dir)
    for file_name in os.listdir(input_dir):
        if file_name.lower().endswith((".jpg", ".png")):
            input_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir, file_name.replace(".jpg", ".npy").replace(".png", ".npy"))
            image = cv2.imread(input_path)
            if image is None:
                print(f"이미지를 로드할 수 없습니다: {input_path}")
                continue
            normalized_image = normalize_image(image)
            np.save(output_path, normalized_image)
            print(f"정규화된 이미지 저장: {output_path}")

# **라벨 생성 및 데이터 분리**
def generate_labels(input_dir, output_file):
    """라벨 생성 및 저장"""
    labels = []
    for file_name in sorted(os.listdir(input_dir)):
        if file_name.lower().endswith(".npy"):
            try:
                angle = int(file_name.split("_")[3].split(".")[0])
                labels.append(angle)
            except (IndexError, ValueError):
                print(f"잘못된 파일 형식: {file_name}")
    if labels:
        labels = np.array(labels)
        setup_folder(os.path.dirname(output_file))
        np.save(output_file, labels)
        print(f"라벨이 저장되었습니다: {output_file}")
        print(f"총 {len(labels)}개의 라벨이 저장되었습니다.")

def split_data():
    """Train/Validation/Test 분리"""
    labels = np.load(os.path.join(LABELS_DIR, "labels.npy"))
    image_files = np.array(sorted([f for f in os.listdir(NORMALIZED_IMAGES_DIR) if f.endswith(".npy")]))
    train_images, temp_images, train_labels, temp_labels = train_test_split(
        image_files, labels, test_size=0.2, random_state=42
    )
    val_images, test_images, val_labels, test_labels = train_test_split(
        temp_images, temp_labels, test_size=0.5, random_state=42
    )
    # 데이터 저장
    setup_folder(LABELS_DIR)
    np.save(os.path.join(LABELS_DIR, "train_images.npy"), train_images)
    np.save(os.path.join(LABELS_DIR, "val_images.npy"), val_images)
    np.save(os.path.join(LABELS_DIR, "test_images.npy"), test_images)
    np.save(os.path.join(LABELS_DIR, "train_labels.npy"), train_labels)
    np.save(os.path.join(LABELS_DIR, "val_labels.npy"), val_labels)
    np.save(os.path.join(LABELS_DIR, "test_labels.npy"), test_labels)

# **PyTorch Dataset 및 DataLoader**
class DrivingDataset(Dataset):
    """PyTorch Dataset 클래스"""
    def __init__(self, image_files, labels, image_dir):
        self.image_files = image_files
        self.labels = labels
        self.image_dir = image_dir

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        image = np.load(image_path)
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label

def create_dataloaders():
    """DataLoader 생성"""
    train_images = np.load(os.path.join(LABELS_DIR, "train_images.npy"))
    train_labels = np.load(os.path.join(LABELS_DIR, "train_labels.npy"))
    val_images = np.load(os.path.join(LABELS_DIR, "val_images.npy"))
    val_labels = np.load(os.path.join(LABELS_DIR, "val_labels.npy"))

    train_dataset = DrivingDataset(train_images, train_labels, NORMALIZED_IMAGES_DIR)
    val_dataset = DrivingDataset(val_images, val_labels, NORMALIZED_IMAGES_DIR)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    return train_loader, val_loader

# **PilotNet 모델**
class PilotNet(nn.Module):
    """PilotNet 모델 정의"""
    def __init__(self):
        super(PilotNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 1 * 18, 1164),
            nn.ReLU(),
            nn.Linear(1164, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        return self.model(x)

# **학습 및 검증**
def train_and_validate(model, train_loader, val_loader, criterion, optimizer, num_epochs=50):
    """모델 학습 및 검증 (손실 기록 포함)"""
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs.squeeze(), labels)
                val_loss += loss.item()

        # 손실 기록
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    # 손실 그래프 시각화
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid()
    plt.show()

    return model

# **테스트**
def evaluate_model(model, test_loader, criterion):
    """모델 평가"""
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels)
            test_loss += loss.item()
    print(f"Test Loss: {test_loss/len(test_loader):.4f}")

# **전체 실행 흐름**
# 폴더 생성 및 데이터 처리
setup_folder(BASE_DIR)
setup_folder(INPUT_DIR)
setup_folder(OUTPUT_DIR)
setup_folder(PROCESSED_IMAGES_DIR)
setup_folder(NORMALIZED_IMAGES_DIR)
setup_folder(LABELS_DIR)

crop_and_resize_images(INPUT_DIR, PROCESSED_IMAGES_DIR)
normalize_images(PROCESSED_IMAGES_DIR, NORMALIZED_IMAGES_DIR)
generate_labels(NORMALIZED_IMAGES_DIR, os.path.join(LABELS_DIR, "labels.npy"))
split_data()

# DataLoader 생성
train_loader, val_loader = create_dataloaders()

# 모델 초기화
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PilotNet().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습 및 검증
model = train_and_validate(model, train_loader, val_loader, criterion, optimizer, num_epochs=50)

# 모델 저장
torch.save(model.state_dict(), "pilotnet.pth")
print("Model training complete and saved.")

# 테스트
test_images = np.load(os.path.join(LABELS_DIR, "test_images.npy"))
test_labels = np.load(os.path.join(LABELS_DIR, "test_labels.npy"))
test_dataset = DrivingDataset(test_images, test_labels, NORMALIZED_IMAGES_DIR)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
evaluate_model(model, test_loader, criterion)