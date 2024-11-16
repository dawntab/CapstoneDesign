#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import Jetson.GPIO as GPIO
import time
import subprocess

# Set up pinmux using busybox devmem
def setup_pinmux():
    commands = [
        "busybox devmem 0x700031fc 32 0x45",
        "busybox devmem 0x6000d504 32 0x2",
        "busybox devmem 0x70003248 32 0x46",
        "busybox devmem 0x6000d100 32 0x00"
    ]
    for command in commands:
        subprocess.run(command, shell=True, check=True)

# Configure pinmux
setup_pinmux()

# Set up GPIO for servo motor
SERVO_PIN = 33
GPIO.setmode(GPIO.BOARD)
GPIO.setup(SERVO_PIN, GPIO.OUT)

# Set up PWM
pwm = GPIO.PWM(SERVO_PIN, 50)  # 50Hz for servo motor
pwm.start(0)  # Start with no signal

# Convert angle to duty cycle
def angle_to_duty_cycle(angle):
    return 2.5 + (angle / 180.0) * 10.0

try:
    while True:
        angle = int(input("Enter angle (0 ~ 180): "))
        if not (0 <= angle <= 180):
            print("Angle out of range. Exiting.")
            break

        duty_cycle = angle_to_duty_cycle(angle)
        print(f"Setting angle {angle} with duty cycle {duty_cycle:.2f}")
        pwm.ChangeDutyCycle(duty_cycle)
        time.sleep(0.5)
        pwm.ChangeDutyCycle(0)  # Stop PWM signal
finally:
    pwm.stop()
    GPIO.cleanup()