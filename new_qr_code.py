import cv2
import numpy as np
from picamera2 import Picamera2
import time

# Load calibration data
data = np.load("camera_calib.npz")

camera_matrix = data["camera_matrix"]
dist_coeffs = data["dist_coeffs"]

# Real QR code size in meters
qr_size = 0.05  # 5 cm

# 3D coordinates of QR corners in real world
object_points = np.array([
    [0, 0, 0],
    [qr_size, 0, 0],
    [qr_size, qr_size, 0],
    [0, qr_size, 0]
], dtype=np.float32)

# Initialize Picamera2
picam2 = Picamera2()

config = picam2.create_preview_configuration(
    main={"size": (640, 480)}
)

picam2.configure(config)
picam2.start()

# Give camera time to initialise
time.sleep(2)

qr_detector = cv2.QRCodeDetector()

print("Press Q to quit")

    frame = picam2.capture_array()
frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

h, w = frame.shape[:2]

new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
    camera_matrix,
    dist_coeffs,
    (w, h),
    1,
    (w, h)
)

while True:

    frame = cv2.undistort(
        frame,
        camera_matrix,
        dist_coeffs,
        None,
        new_camera_matrix
    )

    # Detect QR codes
    retval, decoded_info, points, _ = qr_detector.detectAndDecodeMulti(frame)

    if retval:

        for qr_data, point in zip(decoded_info, points):

            if qr_data:

                image_points = np.array(
                    point,
                    dtype=np.float32
                )

                success, rvec, tvec = cv2.solvePnP(
                    object_points,
                    image_points,
                    camera_matrix,
                    dist_coeffs
                )

                if success:

                    distance = np.linalg.norm(tvec)

                    pts = image_points.astype(int)

                    cv2.polylines(
                        frame,
                        [pts],
                        True,
                        (0, 255, 0),
                        2
                    )

                    center = pts.mean(axis=0).astype(int)

                    text = (
                        f"{qr_data} | "
                        f"Dist: {distance:.2f} m"
                    )

                    cv2.putText(
                        frame,
                        text,
                        (center[0], center[1]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

                    print(
                        f"{qr_data} -> "
                        f"Distance: {distance:.2f} m"
                    )

    cv2.imshow("QR Scanner", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
