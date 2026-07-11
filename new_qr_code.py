from flask import Flask, Response
from picamera2 import Picamera2
import cv2
import numpy as np
import time
import os
import RPi.GPIO as GPIO

app = Flask(__name__)

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

        time.sleep(0.1)

switcher = CameraSwitcher()

# -----------------------------
# Load calibration
# -----------------------------

calibA = np.load("calibA.npz")
calibB = np.load("calibB.npz")
stereo = np.load("stereo_calib.npz")

cameraMatrixA = calibA["camera_matrix"]
distA = calibA["dist_coeffs"]

cameraMatrixB = calibB["camera_matrix"]
distB = calibB["dist_coeffs"]

P1 = stereo["P1"]
P2 = stereo["P2"]

R1 = stereo["R1"]
R2 = stereo["R2"]

# -----------------------------
# QR dimensions
# -----------------------------

qr_size = 0.05

object_points = np.array([
    [0, 0, 0],
    [qr_size, 0, 0],
    [qr_size, qr_size, 0],
    [0, qr_size, 0]
], dtype=np.float32)

# -----------------------------
# Camera setup
# -----------------------------

picam2 = Picamera2()

config = picam2.create_preview_configuration(
    main={"size": (1280, 720)}
)

picam2.configure(config)
picam2.start()

time.sleep(2)

# -----------------------------
# Detector setup
# -----------------------------

qr_detector = cv2.QRCodeDetector()

# -----------------------------
# Calculate camera matrix once
# -----------------------------

first_frame = picam2.capture_array()

first_frame = cv2.cvtColor(
    first_frame,
    cv2.COLOR_RGB2BGR
)

h, w = first_frame.shape[:2]

new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
    camera_matrix,
    dist_coeffs,
    (w, h),
    1,
    (w, h)
)

# -----------------------------
# Streaming function
# -----------------------------

def camera_capture_A():
  switcher.select("A")
  time.sleep(0.15)
  frame = picam2.capture_array()
  frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
  return frame 

def camera_capture_B():
  switcher.select("B")
  time.sleep(0.15)
  frame = picam2.capture_array()
  frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
  return frame 

def capture_stereo_pair():
  frameA = camera_capture_A()
  frameB = camera_capture_B()
  return frameA, frameB

newCameraMatrixA, roi = cv2.getOptimalNewCameraMatrix(
    cameraMatrixA, 
    distA, 
    (w, h), 
    1, 
    (w, h)
)
newCameraMatrixB, roi = cv2.getOptimalNewCameraMatrix(
    cameraMatrixB, 
    distB, 
    (w, h), 
    1, 
    (w, h)
)


def generate_frames():

    while True:

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_RGB2BGR
        )

        frame = cv2.undistort(
            frame,
            camera_matrix,
            dist_coeffs,
            None,
            new_camera_matrix
        )

        frameA, frameB = capture_stereo_pair()
        
        frameA = cv2.undistort(
            frameA, 
            cameraMatrixA, 
            distA, 
            None, 
            newCameraMatrixA
        )
        
        frameB = cv2.undistort(
            frameB,
            cameraMatrixB, 
            distB, 
            None, 
            newCameraMatrixB
        )
        retA, dataA, pointsA, _ = qr_detector.detectAndDecodeMulti(frameA)

        retB, dataB, pointsB, _ = qr_detector.detectAndDecodeMulti(frameB)
        for textA, pA in zip(dataA, pointsA):
          if textA == "":
            continue 
        for textB, pB in zip(dataB, pointsB):
          if textA != textB:
            continue 

            ptsA = cv2.undistortPoints(
              np.array([[centerA]], dtype = np.float32), 
              cameraMatrixA, 
              distA, 
              R=R1, 
              P=P1
            )

            ptsB = cv2.undistortPoints(
              np.array([[centerB]], dtype = np.float32), 
              cameraMatrixB, 
              distA, 
              R=R1, 
              P=P1
            )

            points3D = []
            for i in range(4):
              
              if cornerA[i] is none:
                continue 
              
              if cornerB[i] is none:
                continue

              point3D = triangulate
              
              triangulate corner i
              points3D.append(point3D)
              average_point = np.mean(points3D, axis=0)
              distance = np.linalg.norm(average_point)
              

# -----------------------------
# Video endpoint
# -----------------------------

@app.route('/video')
def video():

    return Response(
        generate_frames(),
        mimetype=
        'multipart/x-mixed-replace; boundary=frame'
    )

# -----------------------------
# Simple webpage
# -----------------------------

@app.route('/')
def index():

    return """
    <html>
        <body>
            <h1>QR Distance Scanner</h1>
            <img src="/video">
        </body>
    </html>
    """

# -----------------------------
# Start Flask
# -----------------------------

if __name__ == "__main__":

    app.run(
        host='0.0.0.0',
        port=5000,
        threaded=True
    )
cv2.destroyAllWindows()
