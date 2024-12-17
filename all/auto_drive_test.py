import torch
import torch.nn as nn
import cv2
import Jetson.GPIO as GPIO
import time
import subprocess
from pynput import keyboard
import threading
import os
from datetime import datetime

# Disable GPIO warnings
GPIO.setwarnings(False)

# Set up GPIO pins for servo and DC motor control
SERVO_PIN = 32  # PWM-capable pin for servo motor
DC_MOTOR_PWM_PIN = 33  # PWM-capable pin for DC motor speed
DC_MOTOR_DIR_PIN1 = 29  # Direction control pin 1
DC_MOTOR_DIR_PIN2 = 31  # Direction control pin 2

GPIO.setmode(GPIO.BOARD)
GPIO.setup(SERVO_PIN, GPIO.OUT)
GPIO.setup(DC_MOTOR_PWM_PIN, GPIO.OUT)
GPIO.setup(DC_MOTOR_DIR_PIN1, GPIO.OUT)
GPIO.setup(DC_MOTOR_DIR_PIN2, GPIO.OUT)

# Configure PWM on servo and DC motor pins
servo_pwm = GPIO.PWM(SERVO_PIN, 50)  # 50Hz for servo motor
dc_motor_pwm = GPIO.PWM(DC_MOTOR_PWM_PIN, 1000)  # 1kHz for DC motor
servo_pwm.start(0)
dc_motor_pwm.start(0)

# Predefined steering angles
steering_angles = [30, 60, 90, 120]
current_angle_index = 0  # Index for the current angle
motor_speed = 50  # Default motor speed set to 50%

# Function to set the servo angle
def set_servo_angle(angle):
    duty_cycle = 2 + (angle / 18)
    servo_pwm.ChangeDutyCycle(duty_cycle)
    time.sleep(0.05)

# Function to set DC motor speed and direction
def set_dc_motor(speed, direction="forward"):
    if direction == "forward":
        GPIO.output(DC_MOTOR_DIR_PIN1, GPIO.HIGH)
        GPIO.output(DC_MOTOR_DIR_PIN2, GPIO.LOW)
    elif direction == "backward":
        GPIO.output(DC_MOTOR_DIR_PIN1, GPIO.LOW)
        GPIO.output(DC_MOTOR_DIR_PIN2, GPIO.HIGH)
    dc_motor_pwm.ChangeDutyCycle(speed)
    time.sleep(0.05)

# PilotNet model definition
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

# Preprocess image
def preprocess_image(frame):
    height, _, _ = frame.shape
    cropped_image = frame[int(height * 0.3):, :]
    resized_image = cv2.resize(cropped_image, (200, 66))
    normalized_image = (resized_image / 255.0) * 2 - 1
    image_tensor = torch.tensor(normalized_image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    return image_tensor

# Load model
def load_model(model_path, device):
    model = PilotNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# Function to snap predicted angle to nearest predefined angle
def snap_to_nearest_angle(predicted_angle, predefined_angles):
    return min(predefined_angles, key=lambda x: abs(x - predicted_angle))

# Real-time control with camera input
def real_time_control(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera 0, trying camera 1")
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("Error: Cannot open camera 1")
            return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Cannot read frame")
                break

            image_tensor = preprocess_image(frame).to(device)
            with torch.no_grad():
                predicted_angle = model(image_tensor).item()

            # Snap predicted angle to nearest predefined angle
            servo_angle = snap_to_nearest_angle(predicted_angle, steering_angles)
            set_servo_angle(servo_angle)
            set_dc_motor(motor_speed)

            display_text = f"Predicted Angle: {servo_angle} deg | Motor Speed: {motor_speed}%"
            cv2.putText(frame, display_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Real-Time Control", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        servo_pwm.stop()
        dc_motor_pwm.stop()
        GPIO.cleanup()

# Model path and execution
MODEL_PATH = "pilotnet.pth"
real_time_control(MODEL_PATH)
