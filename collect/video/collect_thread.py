import Jetson.GPIO as GPIO
import time
import subprocess
from pynput import keyboard
import cv2
import threading

# Disable GPIO warnings
GPIO.setwarnings(False)

# Set the sudo password as a variable for easy updating
sudo_password = "your_password_here"

# Function to run shell commands with the sudo password
def run_command(command):
    full_command = "echo {} | sudo -S {}".format(sudo_password, command)
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

# Initialize servo angle
current_angle = 90  # Start at 90 degrees
set_servo_angle = lambda angle: servo.ChangeDutyCycle(2 + (angle / 18))

# Function to set DC motor speed and direction
def set_dc_motor(speed, direction):
    if direction == "forward":
        GPIO.output(dc_motor_dir_pin1, GPIO.HIGH)
        GPIO.output(dc_motor_dir_pin2, GPIO.LOW)
    elif direction == "backward":
        GPIO.output(dc_motor_dir_pin1, GPIO.LOW)
        GPIO.output(dc_motor_dir_pin2, GPIO.HIGH)
    dc_motor_pwm.ChangeDutyCycle(speed)

# OpenCV video recording setup
cap = cv2.VideoCapture(0)  # Open the default camera
if not cap.isOpened():
    print("Error: Camera not accessible. Check device index or connection.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set frame width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set frame height

fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Define codec
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))  # Output file

# Keyboard control handlers
def on_press(key):
    global current_angle

    try:
        if key.char == 'w':  # Move forward
            set_dc_motor(100, "forward")
        elif key.char == 's':  # Move backward
            set_dc_motor(100, "backward")
        elif key.char == 'a':  # Rotate servo left
            if current_angle > 5:
                current_angle -= 5
                set_servo_angle(current_angle)
        elif key.char == 'd':  # Rotate servo right
            if current_angle < 175:
                current_angle += 5
                set_servo_angle(current_angle)
    except AttributeError:
        pass

def on_release(key):
    if key.char in ['w', 's']:
        set_dc_motor(0, "stop")  # Stop the motor

# Listener for keyboard events
listener = keyboard.Listener(on_press=on_press, on_release=on_release)

# Function to handle video recording
def video_recording():
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame. Exiting.")
                break

            # Write the frame to the output video file
            out.write(frame)

            # Optionally display the frame in a window
            cv2.imshow('Video Feed', frame)

            # Check if ESC key is pressed to exit
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        # Release the video capture and writer
        cap.release()
        out.release()
        cv2.destroyAllWindows()

# Run both threads
try:
    video_thread = threading.Thread(target=video_recording)
    listener.start()

    video_thread.start()
    video_thread.join()
finally:
    # Cleanup GPIO
    servo.stop()
    dc_motor_pwm.stop()
    GPIO.cleanup()

