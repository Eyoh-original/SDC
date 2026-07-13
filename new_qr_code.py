from flask import Flask, Response
from picamera2 import Picamera2
import cv2
import numpy as np
import time
import os
import RPi.GPIO as GPIO

#Defining the camera switcher class. This is to switch between the two cameras. The select method takes in a string "A" or "B" and switches to the corresponding camera. The time.sleep(0.1) is to give the camera time to switch before capturing the next frame.
class CameraSwitcher:
    def __init__(self):
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BOARD)

        GPIO.setup(7, GPIO.OUT) # Camera A
        GPIO.setup(11, GPIO.OUT) # Camera B
        GPIO.setup(12, GPIO.OUT)

    def select(self, camera):
        if camera == "A":
            os.system("i2cset -y 10 0x70 0x00 0x04")

            GPIO.output(7, False)
            GPIO.output(11, False)
            GPIO.output(12, True)

            time.sleep(0.1)

        elif camera == "B":
            os.system("i2cset -y 10 0x70 0x00 0x05")

            GPIO.output(7, True)
            GPIO.output(11, False)
            GPIO.output(12, True)
            
            time.sleep(0.1)

switcher = CameraSwitcher()

#this bit is for loading the calibration date
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

#Just the QR dimensions
qr_size = 0.05

object_points = np.array([
    [0, 0, 0],
    [qr_size, 0, 0],
    [qr_size, qr_size, 0],
    [0, qr_size, 0]
], dtype=np.float32)


#Camera Setup 
picam2 = Picamera2()

config = picam2.create_preview_configuration(
    main = {"size": (640, 480)},
)
picam2.configure(config)
picam2.start()

time.sleep(2) # This is to allow camera to warm up 

#QR code detection setup
qr_detector = cv2.QRCodeDetector()

#Defining camera capture function 
def camera_capture_A():
    switcher.select("A")
    time.sleep(0.15)
    frame = picam2.capture_array()
    h, w = frame.shape[:2]
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame

def camera_capture_B():
    switcher.select("B")
    time.sleep(0.15)
    frame = picam2.capture_array()
    h, w = frame.shape[:2]
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame 

#This is to get the new camera matrix
newCameraMatrixA, roiA = cv2.getOptimalNewCameraMatrix(
    cameraMatrixA, 
    distA, 
    (w, h),
    1, 
    (w, h)
)

newCameraMatrixB, roiB = cv2.getOptimalNewCameraMatrix(
    cameraMatrixB, 
    distB, 
    (w, h),
    1, 
    (w, h)
)

def capture_stereo_pair():
    frameA = camera_capture_A()
    frameB = camera_capture_B()

    undistortedA = cv2.undistort(frameA, cameraMatrixA, distA, None, newCameraMatrixA)
    undistortedB = cv2.undistort(frameB, cameraMatrixB, distB, None, newCameraMatrixB)

    return undistortedA, undistortedB

def generate_frames():
    while True:
        frameA, frameB = capture_stereo_pair()
        
        # QR code detection for camera A
        retvalA, dataA, pointsA, _ = qr_detector.detectAndDecodeMulti(frameA)
        if pointsA is not None:
            for i in range(len(dataA)):
                if dataA[i]:
                    points = pointsA[i].reshape(-1, 2)
                    cv2.polylines(frameA, [points.astype(int)], True, (0, 255, 0), 2)
                    cv2.putText(frameA, dataA[i], tuple(points[0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if not retvalA or pointsA is None:
            continue  # Skip to the next iteration if no QR codes are detected in camera A

        # QR code detection for camera B
        retvalB, dataB, pointsB, _ = qr_detector.detectAndDecodeMulti(frameB)
        if pointsB is not None:
            for i in range(len(dataB)):
                if dataB[i]:
                    points = pointsB[i].reshape(-1, 2)
                    cv2.polylines(frameB, [points.astype(int)], True, (0, 255, 0), 2)
                    cv2.putText(frameB, dataB[i], tuple(points[0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if not retvalB or pointsB is None:
            continue  # Skip to the next iteration if no QR codes are detected in camera B
        
        #The dicitionary for the QR codes. This is to store the detected QR codes and their corresponding corner points for both cameras. It allows for easy comparison and triangulation of the QR code positions in 3D space.
        detectionsA = {}
        for text, corners in zip(dataA, pointsA):
            if text:
                detectionsA[text] = corners

        detectionsB = {}
        for text, corners in zip(dataB, pointsB):
            if text:
                detectionsB[text] = corners
        
        for qr_text in detectionsA:
            
            if qr_text not in detectionsB:
                print(f"QR code {qr_text} detected in Camera A but not in Camera B")
                continue

            cornersA = detectionsA[qr_text]
            cornersB = detectionsB[qr_text]
            points = []

            #This is camera loop. For every matching QR code detected, this loop will undistort the camera and then triangulate
            for i in range(4):
                cornerA = cornersA[i]
                cornerB = cornersB[i]

                np.array([[cornerA]], dtype=np.float32)
                undistortedA = cv2.undistortPoints(np.array([[cornerA]], 
                                             dtype=np.float32), 
                                             cameraMatrixA, 
                                             distA, 
                                             R=R1, 
                                             P=P1
                                             )
                
                np.array([[cornerB]], dtype=np.float32)
                undistortedB = cv2.undistortPoints(np.array([[cornerB]], 
                                             dtype=np.float32), 
                                             cameraMatrixB, 
                                             distB, 
                                             R=R2, 
                                             P=P2
                                             )
                point4D = cv2.triangulatePoints(P1, P2, undistortedA, undistortedB)
                point3D = point4D[:3] / point4D[3]
                points.append(point3D)
            
            if len(points) == 0:
                print(f"No valid triangulated points for QR code {qr_text}")
                continue
            
            #This bit just calculates the average point of the triangulated points and then calculates the distance from the camera to that point. It then draws the QR code corners on both camera frames and displays the distance on both frames.
            average_point = np.mean(points, axis=0)
            distance = np.linalg.norm(average_point)

            cv2.polylines(frameA, [cornersA.astype(int)], True, (0, 255, 0), 2)
            cv2.polylines(frameB, [cornersB.astype(int)], True, (0, 255, 0), 2)

            text = f"{qr_text}: {distance:.2f} m"
            cv2.putText(
                frameA,
                text, 
                (int(cornersA[0][0]), int(cornersA[0][1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
            cv2.putText(
                frameB,
                text,
                (int(cornersB[0][0]), int(cornersB[0][1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

        retA, bufferA = cv2.imencode(
        '.jpg',
        frameA
        )
        retB, bufferB = cv2.imencode(
        '.jpg',
        frameB
        )

        if not retA or not retB:
            continue

        frameA = bufferA.tobytes()
        frameB = bufferB.tobytes()
        combined = np.hstack((frameA, frameB))

        yield(
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'
            + combined +
            b'\r\n'
        )
        

#Video Endpoint 

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
    
