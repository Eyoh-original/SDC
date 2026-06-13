import cv2
import numpy as np

chessboard_size = (7,7)

# prepare 3D object points
objp = np.zeros((np.prod(chessboard_size),3), np.float32)
objp[:,:2] = np.indices(chessboard_size).T.reshape(-1,2)

objpoints = []
imgpoints = []

cap = cv2.VideoCapture("libcamerasrc ! video/x-raw, width=640, height=480 ! videoconvert ! appsink", cv2.CAP_GSTREAMER)

print("Press SPACE to capture calibration frame")
print("Press Q to finish and calibrate")

while True:
    # grab frames to reduce latency
    cap.grab()

    ret, frame = cap.read()

    if not ret:
        print("Camera not detected")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ret_cb, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    display = frame.copy()

    if ret_cb:
        cv2.drawChessboardCorners(display, chessboard_size, corners, ret_cb)

    cv2.imshow("Camera Calibration", display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):  # SPACE pressed
        if ret_cb:
            objpoints.append(objp)
            imgpoints.append(corners)
            print("Captured image", len(objpoints))
        else:
            print("Chessboard not detected")

    elif key == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()


if len(objpoints) < 5:
    print("Not enough calibration images")
    exit()

print("Calibrating camera...")

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    gray.shape[::-1],
    None,
    None
)

print("\nCamera matrix:\n", mtx)
print("\nDistortion coefficients:\n", dist)

np.savez("camera_calibration.npz", camera_matrix=mtx, dist_coeffs=dist)

print("\nCalibration saved to camera_calibration.npz")
