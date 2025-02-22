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

# Set the sudo password as a variable for easy updating
sudo_password = "your_password_here"

# Function to run shell commands with the sudo password
def run_command(command):
    full_command = f"echo {sudo_password} | sudo -S {command}"
    subprocess.run(full_command, shell=True, check=True)

# Check if busybox is installed; if not, install it
try:
    subprocess.run("busybox --help", shell=True, check=True)
    print("busybox is already installed.")
except subprocess.CalledProcessError:
    print("busybox not found. Installing...")
    run_command("apt update && apt install -y busybox")

# Define devmem commands
commands = [
    "busybox devmem 0x700031fc 32 0x45",
    "busybox devmem 0x6000d504 32 0x2",
    "busybox devmem 0x70003248 32 0x46",
    "busybox devmem 0x6000d100 32 0x00"
]

# Execute each devmem command
for command in commands:
    run_command(command)

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
program_running = True  # Flag to control threads
current_angle = 90  # Initialize servo angle
motor_running = False
motor_direction = "forward"

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
    cropped_image = frame[int(height * 0.3):, :]  # Crop bottom 70% of the image
    resized_image = cv2.resize(cropped_image, (200, 66))  # Resize to 200x66
    normalized_image = (resized_image / 255.0) * 2 - 1  # Normalize to [-1, 1]
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

# Camera capture thread
def camera_thread(model, device):
    global program_running, current_angle
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: Cannot open camera 0, trying camera 1")
        cap = cv2.VideoCapture(1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not cap.isOpened():
            print("Error: Cannot open camera 1")
            return
    
    try:
        while program_running:
            ret, frame = cap.read()
            if not ret:
                print("Error: Cannot read frame")
                break

            image_tensor = preprocess_image(frame).to(device)
            with torch.no_grad():
                predicted_angle = model(image_tensor).item()

            current_angle = snap_to_nearest_angle(predicted_angle, steering_angles)
            set_servo_angle(current_angle)

            display_text = f"Predicted Angle: {current_angle} deg | Motor Speed: {motor_speed}%"
            cv2.putText(frame, display_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Real-Time Control", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                program_running = False
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

# Motor control thread
def motor_thread():
    global program_running
    try:
        while program_running:
            set_dc_motor(motor_speed)
            time.sleep(0.1)
    finally:
        set_dc_motor(0)

# Keyboard control thread
def keyboard_thread():
    global program_running, current_angle, motor_running, motor_speed

    def on_press(key):
        try:
            if key.char == 'w':
                set_dc_motor(motor_speed, "forward")
            elif key.char == 's':
                set_dc_motor(motor_speed, "backward")
            elif key.char == 'a':
                set_servo_angle(max(30, current_angle - 10))
            elif key.char == 'd':
                set_servo_angle(min(120, current_angle + 10))
        except AttributeError:
            pass

    def on_release(key):
        global program_running
        if key == keyboard.Key.esc:
            program_running = False
            return False
        set_dc_motor(0)

    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

# Real-time control with threading
def real_time_control(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)

    # Start threads for camera, motor, and keyboard control
    camera = threading.Thread(target=camera_thread, args=(model, device))
    motor = threading.Thread(target=motor_thread)
    keyboard_listener = threading.Thread(target=keyboard_thread)

    camera.start()
    motor.start()
    keyboard_listener.start()

    camera.join()
    motor.join()
    keyboard_listener.join()

    print("Program terminated.")

# Model path and execution
MODEL_PATH = "pilotnet.pth"
real_time_control(MODEL_PATH)
