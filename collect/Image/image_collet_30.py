import Jetson.GPIO as GPIO
import time
import subprocess
from pynput import keyboard
import cv2
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

# Set up GPIO pins for servo and DC motor control
servo_pin = 32  # PWM-capable pin for servo motor
dc_motor_pwm_pin = 33  # PWM-capable pin for DC motor speed
dc_motor_dir_pin1 = 29  # Direction control pin 1
dc_motor_dir_pin2 = 31  # Direction control pin 2

GPIO.setmode(GPIO.BOARD)
GPIO.setup(servo_pin, GPIO.OUT)
GPIO.setup(dc_motor_pwm_pin, GPIO.OUT)
GPIO.setup(dc_motor_dir_pin1, GPIO.OUT)
GPIO.setup(dc_motor_dir_pin2, GPIO.OUT)

# Configure PWM on servo and DC motor pins
servo = GPIO.PWM(servo_pin, 50)  # 50Hz for servo motor
dc_motor_pwm = GPIO.PWM(dc_motor_pwm_pin, 1000)  # 1kHz for DC motor
servo.start(0)
dc_motor_pwm.start(0)

# Initialize variables
current_angle = 30  # Start at 30 degrees
motor_speed = 0  # Start with the motor stopped
program_running = True  # Flag to handle program termination

# Predefined steering angles
steering_angles = [30, 60, 90, 120, 150]
current_angle_index = 0  # Index for the current angle in the steering_angles list

# Create main directory for storing images
current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
base_output_dir = f"images/{current_timestamp}"
os.makedirs(base_output_dir, exist_ok=True)

# Create subdirectories for each steering angle
angle_dirs = {}
for angle in steering_angles:
    angle_dir = os.path.join(base_output_dir, str(angle))
    os.makedirs(angle_dir, exist_ok=True)
    angle_dirs[angle] = angle_dir

# Function to set the servo angle
def set_servo_angle(angle):
    duty_cycle = 2 + (angle / 18)
    servo.ChangeDutyCycle(duty_cycle)

# Function to set DC motor speed and direction
def set_dc_motor(speed, direction):
    if direction == "forward":
        GPIO.output(dc_motor_dir_pin1, GPIO.HIGH)
        GPIO.output(dc_motor_dir_pin2, GPIO.LOW)
    elif direction == "backward":
        GPIO.output(dc_motor_dir_pin1, GPIO.LOW)
        GPIO.output(dc_motor_dir_pin2, GPIO.HIGH)
    dc_motor_pwm.ChangeDutyCycle(speed)

# OpenCV setup for taking pictures
cap = cv2.VideoCapture(0)  # Open the default camera
if not cap.isOpened():
    print("Error: Camera not accessible. Check device index or connection.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set frame width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set frame height

# Keyboard control handlers
def on_press(key):
    global current_angle_index, motor_speed

    try:
        if key.char == 'w':  # Move forward
            motor_speed = 50
            set_dc_motor(motor_speed, "forward")
        elif key.char == 's':  # Move backward
            motor_speed = 50
            set_dc_motor(motor_speed, "backward")
        elif key.char == 'a':  # Rotate servo to the previous predefined angle
            if current_angle_index > 0:
                current_angle_index -= 1
                set_servo_angle(steering_angles[current_angle_index])
        elif key.char == 'd':  # Rotate servo to the next predefined angle
            if current_angle_index < len(steering_angles) - 1:
                current_angle_index += 1
                set_servo_angle(steering_angles[current_angle_index])
    except AttributeError:
        pass

def on_release(key):
    global motor_speed, program_running
    if hasattr(key, 'char') and key.char in ['w', 's']:
        motor_speed = 0
        set_dc_motor(0, "forward")  # Stop the motor
    elif key == keyboard.Key.esc:
        program_running = False  # Set flag to terminate the program
        return False

# Listener for keyboard events
listener = keyboard.Listener(on_press=on_press, on_release=on_release)

# Function to handle picture capturing and showing real-time feed
def capture_and_show():
    while program_running:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting.")
            break

        # Show the live video feed
        cv2.imshow("Live Feed", frame)

        # Save the frame to the current angle folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        current_angle = steering_angles[current_angle_index]
        image_filename = f"{timestamp}_{current_angle}_{motor_speed}.jpg"
        image_path = os.path.join(angle_dirs[current_angle], image_filename)
        cv2.imwrite(image_path, frame)

        print(f"Captured: {image_path}")

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Run both threads
try:
    capture_thread = threading.Thread(target=capture_and_show)  # Background image capture thread
    capture_thread.start()
    listener.start()

    while program_running:
        time.sleep(0.1)  # Main thread keeps running until ESC is pressed

finally:
    # Cleanup GPIO and resources
    program_running = False
    capture_thread.join()  # Ensure thread is terminated
    listener.stop()
    cap.release()  # Release camera
    cv2.destroyAllWindows()  # Close OpenCV windows
    servo.stop()
    dc_motor_pwm.stop()
    GPIO.cleanup()
    print("Program terminated.")