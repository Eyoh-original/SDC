import RPi.GPIO as GPIO
import os
import time

class CameraSwitcher:

    def __init__(self):

        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BOARD)

        GPIO.setup(7, GPIO.OUT)
        GPIO.setup(11, GPIO.OUT)
        GPIO.setup(12, GPIO.OUT)

    def select(self, camera):

        if camera == "A":

            os.system("i2cset -y 10 0x70 0x00 0x04")

            GPIO.output(7, False)
            GPIO.output(11, False)
            GPIO.output(12, True)

        elif camera == "B":

            os.system("i2cset -y 10 0x70 0x00 0x05")

            GPIO.output(7, True)
            GPIO.output(11, False)
            GPIO.output(12, True)

        time.sleep(0.5)
