import Jetson.GPIO as GPIO
import time
import subprocess
from pynput import keyboard

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

# Initialize servo angle and motor speed
current_angle = 90  # Start at 90 degrees
motor_speed = 50  # Start at 50% speed
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

# Flag variables for motor control
motor_running = False
motor_direction = "forward"

# Keyboard control handlers
def on_press(key):
    global current_angle, motor_running, motor_direction, motor_speed

    try:
        if key.char == 'w':  # Move forward
            motor_direction = "forward"
            motor_running = True
            set_dc_motor(motor_speed, "forward")
        elif key.char == 's':  # Move backward
            motor_direction = "backward"
            motor_running = True
            set_dc_motor(motor_speed, "backward")
        elif key.char == 'a':  # Rotate servo left
            if current_angle > 10:  # 변경: 최소 각도를 10도보다 크게 제한
                current_angle -= 10  # 변경: 10도씩 감소
                set_servo_angle(current_angle)
        elif key.char == 'd':  # Rotate servo right
            if current_angle < 170:  # 변경: 최대 각도를 170도로 제한
                current_angle += 10  # 변경: 10도씩 증가
                set_servo_angle(current_angle)
    except AttributeError:
        # Handle special keys
        if key == keyboard.Key.up:  # Increase speed
            motor_speed = min(100, motor_speed + 10)
            if motor_running:
                set_dc_motor(motor_speed, motor_direction)
        elif key == keyboard.Key.down:  # Decrease speed
            motor_speed = max(0, motor_speed - 10)
            if motor_running:
                set_dc_motor(motor_speed, motor_direction)

def on_release(key):
    global motor_running

    if key == keyboard.Key.esc:
        return False  # Exit program on ESC
    if hasattr(key, 'char') and key.char in ['w', 's']:
        motor_running = False
        set_dc_motor(0, motor_direction)  # Stop the motor

# Listener for keyboard events
listener = keyboard.Listener(on_press=on_press, on_release=on_release)

# Start the keyboard listener
listener.start()

print("Control the motors using W/S (forward/backward), A/D (servo left/right), and UP/DOWN (increase/decrease speed). Press ESC to quit.")

try:
    while True:
        time.sleep(0.1)
finally:
    # Clean up GPIO
    servo.stop()
    dc_motor_pwm.stop()
    GPIO.cleanup()
    listener.stop()