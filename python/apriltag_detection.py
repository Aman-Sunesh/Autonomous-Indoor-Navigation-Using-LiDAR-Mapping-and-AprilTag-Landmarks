# ============================================================================
# AprilTag Detection and Pose Estimation
# ============================================================================
# This script detects AprilTags from a live camera feed and estimates their pose.
#
# Main tasks:
# - Opens the phone camera stream.
# - Detects AprilTags using OpenCV's AprilTag 36h11 dictionary.
# - Uses the calibrated camera matrix and distortion coefficients.
# - Estimates each tag's 3D position and rotation using solvePnP.
# - Draws the detected tag outline and coordinate axes on the video frame.
# - Prints the tag ID, distance, rotation vector, roll, pitch, and yaw.
# - Shows the live annotated camera feed until the user presses Q.
# ============================================================================

import cv2
import numpy as np

# Setup
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)

tag_size_meters = 0.167
half_size = tag_size_meters / 2.0
tag_3d_corners = np.array([
    [-half_size,  half_size, 0],
    [ half_size,  half_size, 0],
    [ half_size, -half_size, 0],
    [-half_size, -half_size, 0]
], dtype=np.float32)


# Camera Matrix
cam_matrix = np.array([[656.96495861,   0.        , 630.84080906],
                [  0.        , 658.39179162, 378.57855594],
                [  0.        ,   0.        ,   1.        ]])

# Distortion Coeffs
dist_coeffs = np.array([[0.10811153, -0.27102501, 0.00188884, 0.00495593, 0.20428807]])

def rvec_to_euler_degrees(rvec):
    """
    Convert OpenCV rotation vector into roll, pitch, yaw angles in degrees.

    Note:
    These angles describe the tag's orientation relative to the camera frame.
    """
    R, _ = cv2.Rodrigues(rvec)

    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0

    return np.degrees([roll, pitch, yaw])

# Phone camera index (must match the one used in calibration)
CAMERA_INDEX = 1 
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print(f"Could not open camera index {CAMERA_INDEX}")
    exit()

# Creating infinite loop
while True:
    ret, frame = cap.read()     # ret = checks if the cap was successful. frame = array of pixels
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)      # convert frame to black and white
    corners, ids, rejected = detector.detectMarkers(gray)
        # corners: holds 2D pixel coordinates of the 4 corners of every tag that it captures
        # ids: holds the ID number of every tag

    if ids is not None:                 # if a tag is detected,
        for i in range(len(ids)):       # for every detected tag,
            tag_id = ids[i][0]          # hold tag id
            tag_2d_corners = corners[i][0]      # hold 4 corner coordinates CW from top left

            success, rvec, tvec = cv2.solvePnP(
                tag_3d_corners, tag_2d_corners, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE
            )       # Take the 3D coordinates of the 4 corners, compare with 2D information,
                    # and using the calibration parameters, calculate success, rvec and tvec.
                        # success: boolean to show if the math worked
                        # rvec: tells how much the tag is rotated by
                        # tvec: tells how far the tag is

            if success:     # if the math worked,
                x, y, z = tvec[0][0], tvec[1][0], tvec[2][0]
                    # extra [0] at the end is to unwrap the list of lists. without it, it returns a list
                    # [17] instead of the int 17, which would crash the code.

                roll, pitch, yaw = rvec_to_euler_degrees(rvec)
                
                # Draw boxes and axes
                cv2.polylines(frame, [tag_2d_corners.astype(np.int32)], True, (0, 255, 0), 2)
                    # in frame (= the array of pixels), connect the corner coordinates
                cv2.drawFrameAxes(frame, cam_matrix, dist_coeffs, rvec, tvec, 0.03)
                    # draw the 3D coordinate axes in the middle of the tag, each line 3cm in length
                cv2.putText(frame, f"ID: {tag_id} | Z: {z:.2f}m", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)                
                cv2.putText(
                    frame,
                    f"R:{roll:+.0f} P:{pitch:+.0f} Y:{yaw:+.0f}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

                print(
                    f"Tag {tag_id} -> "
                    f"X: {x:.3f}m, Y: {y:.3f}m, Z: {z:.3f}m | "
                    f"rvec: [{rvec[0][0]:+.3f}, {rvec[1][0]:+.3f}, {rvec[2][0]:+.3f}] | "
                    f"roll: {roll:+.1f}°, pitch: {pitch:+.1f}°, yaw: {yaw:+.1f}°"
                )

    cv2.imshow("Phone AprilTag Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()