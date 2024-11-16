#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import Jetson.GPIO as GPIO
import time
import subprocess

# 핀mux 설정 (첫 번째 코드 방식 적용)
def setup_pinmux():
    commands = [
        "busybox devmem 0x700031fc 32 0x45",
        "busybox devmem 0x6000d504 32 0x2"
    ]
    for command in commands:
        subprocess.run(command, shell=True, check=True)

# 핀mux 설정 호출
setup_pinmux()

# 서보모터 연결 핀 설정
SERVO_PIN = 33 

# GPIO 설정
GPIO.setmode(GPIO.BOARD)
GPIO.setup(SERVO_PIN, GPIO.OUT)

# PWM 객체 생성 
pwm = GPIO.PWM(SERVO_PIN, 50)  # 50Hz PWM 신호 생성

# PWM 시작 (초기값 0)
pwm.start(0)
time.sleep(2)  # 초기화 대기

# 각도 -> Duty Cycle 변환 함수
def angle_to_duty_cycle(angle):
    # 서보모터의 동작 범위 (0.5ms ~ 2.5ms, 50Hz 기준 Duty Cycle = 2.5% ~ 12.5%)
    return 2.5 + (angle / 180.0) * 10.0

try:
    while True:
        angle = int(input("angle (0 ~ 180): "))
        
        if not(0 <= angle <= 180):
            print("Invalid angle! Please enter a value between 0 and 180.")
            continue

        duty_cycle = angle_to_duty_cycle(angle)
        pwm.ChangeDutyCycle(duty_cycle)
        time.sleep(0.5)  # 서보모터가 움직일 시간 대기
        pwm.ChangeDutyCycle(0)  # 신호를 끄기 (떨림 방지)
finally:
    pwm.stop()
    GPIO.cleanup()