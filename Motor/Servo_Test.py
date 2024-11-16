#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import Jetson.GPIO as GPIO
import time

# 서보모터 연결 핀 설정
SERVO_PIN = 33  

# GPIO 설정
GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO_PIN, GPIO.OUT)

# PWM 객체 생성 
pwm = GPIO.PWM(SERVO_PIN, 50)

# PWM 시작 
pwm.start(0)
time.sleep(2)  # 서보모터 초기화 시간 대기

# 각도 -> Duty Cycle 변환 함수
def angle_to_duty_cycle(angle):
    # 서보모터의 동작 범위 (0.5ms ~ 2.5ms, 50Hz 기준 Duty Cycle = 2.5% ~ 12.5%)
    return 2.5 + (angle / 180.0) * 10.0


while True:
    angle = int(input("angle(0 ~ 180): "))

    if not(0 <= angle <= 180):
        break

    pwm.ChangeDutyCycle(angle_to_duty_cycle(angle))
    

pwm.stop()
GPIO.cleanup()