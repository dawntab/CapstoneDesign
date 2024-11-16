import Jetson.GPIO as GPIO
import time

#모터가 사용할 PWM PIN 설정
SERVO_PIN = 33

#GPIO 채널 설정
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(SERVO_PIN, GPIO.OUT)

#주파수 50hz, pwm high 주기 설정
pwm = GPIO.PWM(SERVO_PIN, 50)
pwm.start(3.0)                

for t_high in range(30, 125):            
    pwm.ChangeDutyCycle(t_high/10.0)     
    time.sleep(0.02)
    
pwm.ChangeDutyCycle(3.0)
time.sleep(1.0)
pwm.ChangeDutyCycle(0.0)

pwm.stop()
GPIO.cleanup()