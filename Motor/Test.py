import Jetson.GPIO as GPIO
import time

SERVO_PIN = 32  # BOARD 모드에서의 핀 번호
GPIO.setmode(GPIO.BOARD)
GPIO.setup(SERVO_PIN, GPIO.OUT)

pwm = GPIO.PWM(SERVO_PIN, 50)  # 50Hz PWM
pwm.start(7.5)  # 중립 위치

try:
    while True:
        pwm.ChangeDutyCycle(5)  # 최소 위치
        time.sleep(1)
        pwm.ChangeDutyCycle(10)  # 최대 위치
        time.sleep(1)
except KeyboardInterrupt:
    pwm.stop()
    GPIO.cleanup()