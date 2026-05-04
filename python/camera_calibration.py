# ============================================================================
# Camera Calibration
# ============================================================================
# This script calibrates the camera using a printed chessboard pattern.
#
# Main tasks:
# - Opens the selected camera feed.
# - Detects chessboard corners in the live video frame.
# - Saves good calibration frames when the user presses SPACE.
# - Uses the captured chessboard points to compute camera calibration.
# - Prints the camera matrix, distortion coefficients, RMS error, and frame size.
# - These calibration values are used later for AprilTag pose estimation.
# ============================================================================

import numpy as np
import cv2

# --- SETUP ---
CHESSBOARD_WIDTH = 9
CHESSBOARD_HEIGHT = 6
SQUARE_SIZE_METERS = 0.022 # Change to your actual square size in meters

# Try 1 or 2 if your laptop has a built-in webcam at 0
CAMERA_INDEX = 1 

objp = np.zeros((CHESSBOARD_HEIGHT * CHESSBOARD_WIDTH, 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_WIDTH, 0:CHESSBOARD_HEIGHT].T.reshape(-1, 2) * SQUARE_SIZE_METERS

objpoints = [] 
imgpoints = [] 

cap = cv2.VideoCapture(CAMERA_INDEX)
print("Press SPACE to capture a frame. Press 'c' to calculate calibration.")

captured_frames = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to access camera. Try changing CAMERA_INDEX.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret_corners, corners = cv2.findChessboardCorners(gray, (CHESSBOARD_WIDTH, CHESSBOARD_HEIGHT), None)

    display_frame = frame.copy()
    if ret_corners:
        cv2.drawChessboardCorners(display_frame, (CHESSBOARD_WIDTH, CHESSBOARD_HEIGHT), corners, ret_corners)
    
    cv2.putText(display_frame, f"Captured: {captured_frames}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Phone Camera Calibration', display_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '): 
        if ret_corners:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners_refined)
            captured_frames += 1
            print(f"Captured {captured_frames} frames!")

    elif key == ord('c'):
        if captured_frames < 10:
             print("Warning: Need at least 10 frames!")
        else:
             break

print("Calibrating... please wait.")
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("\n--- COPY THESE INTO YOUR APRILTAG SCRIPT ---")
print("Camera Matrix (mtx):\n", repr(mtx))
print("Distortion Coeffs (dist):\n", repr(dist))
print("Reprojection RMS error:", ret)
print("Frame shape:", frame.shape)

cap.release()
cv2.destroyAllWindows()