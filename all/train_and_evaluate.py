import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

# 데이터셋 정의
class DrivingDataset(Dataset):
    def __init__(self, image_files, labels, image_dir):
        self.image_files = image_files
        self.labels = labels
        self.image_dir = image_dir

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        image = np.load(image_path)  # 정규화된 이미지 로드
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # (H, W, C) → (C, H, W)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label

# PilotNet 모델
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

# 학습 함수
def train_model(train_loader, val_loader, model, criterion, optimizer, device, num_epochs=20):
    train_losses, val_losses = [], []
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

        train_loss /= len(train_loader)
        val_loss = evaluate_model(val_loader, model, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    return train_losses, val_losses

# 검증 함수
def evaluate_model(val_loader, model, criterion, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels)
            val_loss += loss.item()
    return val_loss / len(val_loader)

# 메인 실행 부분
def main():
    # 데이터 경로 설정
    BASE_DIR = "data/output"
    NORMALIZED_IMAGES_DIR = os.path.join(BASE_DIR, "normalized_images")
    LABELS_DIR = os.path.join(BASE_DIR, "labels")

    # 데이터 로드
    train_images = np.load(os.path.join(LABELS_DIR, "train_images.npy"))
    train_labels = np.load(os.path.join(LABELS_DIR, "train_labels.npy"))
    val_images = np.load(os.path.join(LABELS_DIR, "val_images.npy"))
    val_labels = np.load(os.path.join(LABELS_DIR, "val_labels.npy"))

    train_dataset = DrivingDataset(train_images, train_labels, NORMALIZED_IMAGES_DIR)
    val_dataset = DrivingDataset(val_images, val_labels, NORMALIZED_IMAGES_DIR)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # 모델 및 학습 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PilotNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 모델 학습
    print("Training the model...")
    train_model(train_loader, val_loader, model, criterion, optimizer, device, num_epochs=30)

    # 모델 저장
    torch.save(model.state_dict(), "pilotnet.pth")
    print("Model saved as 'pilotnet.pth'.")

if __name__ == "__main__":
    main()