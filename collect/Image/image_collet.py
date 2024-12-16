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
current_angle = 90  # Start at 90 degrees
motor_speed = 0  # Start with the motor stopped
program_running = True  # Flag to handle program termination

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

# Create a timestamped directory for storing images
current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"images/{current_timestamp}"
os.makedirs(output_dir, exist_ok=True)

# Keyboard control handlers
def on_press(key):
    global current_angle, motor_speed

    try:
        if key.char == 'w':  # Move forward
            motor_speed = 50
            set_dc_motor(motor_speed, "forward")
        elif key.char == 's':  # Move backward
            motor_speed = 50
            set_dc_motor(motor_speed, "backward")
        elif key.char == 'a':  # Rotate servo left
            if current_angle > 10:
                current_angle -= 10
                set_servo_angle(current_angle)
        elif key.char == 'd':  # Rotate servo right
            if current_angle < 170:
                current_angle += 10
                set_servo_angle(current_angle)
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
    try:
        while program_running:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame. Exiting.")
                break

            # Show the live video feed
            cv2.imshow("Live Feed", frame)

            # Generate the filename with timestamp, angle, and speed
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_filename = f"{timestamp}_{current_angle}_{motor_speed}.jpg"
            image_path = os.path.join(output_dir, image_filename)

            # Save the frame as an image
            cv2.imwrite(image_path, frame)
            print(f"Captured: {image_path}")

            # Exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(1)  # Wait for 1 second before capturing the next frame
    finally:
        cap.release()
        cv2.destroyAllWindows()

# Run both threads
try:
    capture_thread = threading.Thread(target=capture_and_show)
    capture_thread.start()
    listener.start()

    while program_running:
        time.sleep(0.1)  # Main thread keeps running until ESC is pressed

finally:
    # Cleanup GPIO
    servo.stop()
    dc_motor_pwm.stop()
    GPIO.cleanup()
    print("Program terminated.")