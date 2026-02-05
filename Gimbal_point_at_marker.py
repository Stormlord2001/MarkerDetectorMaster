import cv2
from control_gimbal import GimbalCommand
import nFoldEdgeCodeDisk.MarkerLocator as ml
import time
import os
import math
import numpy as np
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

use_gimbal = True

gimbal = GimbalCommand()
if use_gimbal:
    
    gimbal.center_gimbal()
    time.sleep(2)

    gimbal.pid_yaw.kp = 6   # P, I, D values for yaw
    gimbal.pid_yaw.ki = 6/8
    gimbal.pid_yaw.kd = 1.0
    gimbal.pid_pitch.kp = 5   # P, I, D values for pitch
    gimbal.pid_pitch.ki = 5/8
    gimbal.pid_pitch.kd = 1.0

# RTSP URL
url = "rtsp://192.168.144.25:8554/main.264"

# Force FFMPEG backend
#cap = cv2.VideoCapture(
#    url,
#    cv2.CAP_FFMPEG
#)

cap = cv2.VideoCapture(1)  # Use webcam for testing

# Reduce buffering
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # keep only the latest frame

# Disable auto exposure
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)

# Fast shutter (requires lots of light!)
cap.set(cv2.CAP_PROP_EXPOSURE, -8)         # range: -1 .. -13 (lower = faster)

# Keep gain low for less noise
cap.set(cv2.CAP_PROP_GAIN, 100)

if not cap.isOpened():
    print("Could not open video stream")
    exit()

# Camera intrinsics and distortion
intrinsics = np.array([[835.4362078622368, 0, 323.0605420101571],
                          [0, 835.9483791851382, 232.14120929722597],
                          [0, 0, 1]], dtype=float)
dist_coeffs = np.array([-0.0999921394506428, 2.185188066835036, -0.005726667745540125, 0.00027787706601120816, -7.636164458366145], dtype=float)

cd = ml.CameraDriver([5], default_kernel_size=13, scaling_parameter=1000, downscale_factor=2)  # Best in robolab.

lp = ml.LoadPosition(intrinsics, dist_coeffs, downscale_factor=2)


while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame lost")
        continue

    cd.current_frame = frame
    cd.process_frame()
    
    
    if cd.locations:
        test = lp.estimate_load_pose(cd.locations)
        #print("Estimated load position (x, y, z, roll, pitch, yaw): ", lp.load_position)
        lp.PE.display_pose(cd.current_frame, axis_length=0.05)
    
        if test is not None:

            tvec = test[1].ravel()
            # calc roll and pitch from tvec
            x, y, z = tvec[0], tvec[1], tvec[2] # in camera frame to payload frame
            pitch = math.atan2(y, z) * (180 / math.pi) # pitch in gimbal, roll in camera frame
            yaw = -math.atan2(x, z) * (180 / math.pi) # yaw in gimbal, -pitch in camera frame


            print(f'pitch: {pitch: 6.2f}, yaw: {yaw: 6.2f}')
        
            if use_gimbal:
                current_yaw, current_pitch, _ = gimbal.get_attitude()
            else:
                current_yaw, current_pitch = 10, 0
            yaw_error = yaw
            pitch_error = pitch


            wrap_yaw = math.atan2(math.sin(math.radians(yaw_error)), math.cos(math.radians(yaw_error))) * (180 / math.pi)
            wrap_pitch = math.atan2(math.sin(math.radians(pitch_error)), math.cos(math.radians(pitch_error))) * (180 / math.pi)

            #print(f'yaw pixel, {error_yaw:.2f},  desired {yaw:.2f}, current {current_yaw:.2f}, error {yaw_error:.2f}, wrap {wrap_yaw:.2f}')
            #print(f'pitch pixel, {error_pitch:.2f},  desired {pitch_desired:.2f}, current {current_pitch:.2f}, error {pitch_error:.2f}, wrap {wrap_pitch:.2f}')

            yaw_speed = gimbal.pid_yaw.update(wrap_yaw)
            pitch_speed = gimbal.pid_pitch.update(wrap_pitch)
            #print(f"Yaw speed: {yaw_speed}, Pitch speed: {pitch_speed}")
            if use_gimbal:
                #gimbal.move_speed(int(min(100, max(-100, yaw_speed))), 0)
                #gimbal.move_speed(0, int(min(100, max(-100, pitch_speed))))
                gimbal.move_speed(int(min(100, max(-100, yaw_speed))), int(min(100, max(-100, pitch_speed))))
                
        else:
            if use_gimbal:
                gimbal.move_speed(0, 0)
    else:
        if use_gimbal:
            gimbal.move_speed(0, 0)
        print("No marker detected")

    
    cd.draw_detected_markers()

    """if cd.locations:
        marker = cd.locations[0]  # Take the first detected marker
        #print(f"Marker ID: {marker.id}, Position: ({marker.x}, {marker.y}), Angle: {marker.theta}")
        image_center_x = frame.shape[1]/2
        error_yaw = marker.x * cd.downscale_factor - image_center_x
        image_center_y = frame.shape[0]/2
        error_pitch = marker.y * cd.downscale_factor - image_center_y

        # Use focal length to convert pixel error to angle error
        focal_length_x = intrinsics[0, 0]  # Assuming fx is the focal length in pixels
        focal_length_y = intrinsics[1, 1]  # Assuming fy is the focal length in pixels
        yaw_desired = math.degrees(math.atan2(error_yaw, focal_length_x))
        pitch_desired = math.degrees(math.atan2(error_pitch, focal_length_y))

        #print("Desired Yaw: ", yaw_desired, "Desired Pitch: ", pitch_desired)

        #print(f"Error Yaw: {error_yaw}, Error Pitch: {error_pitch}")
        # Command gimbal to point at marker
        if use_gimbal:
            current_yaw, current_pitch, _ = gimbal.get_attitude()
        else:
            current_yaw, current_pitch = 10, 0
        yaw_error = - yaw_desired #current_yaw - yaw_desired
        pitch_error = pitch_desired # - current_pitch + 180  # Adjust for gimbal pitch convention

        #print("Yaw error: ", yaw_error, "Pitch error: ", pitch_error)

        wrap_yaw = math.atan2(math.sin(math.radians(yaw_error)), math.cos(math.radians(yaw_error))) * (180 / math.pi)
        #wrap_pitch = (pitch_error + 180) % 360 - 180
        wrap_pitch = math.atan2(math.sin(math.radians(pitch_error)), math.cos(math.radians(pitch_error))) * (180 / math.pi)

        print(f'yaw pixel, {error_yaw:.2f},  desired {yaw_desired:.2f}, current {current_yaw:.2f}, error {yaw_error:.2f}, wrap {wrap_yaw:.2f}')
        #print(f'pitch pixel, {error_pitch:.2f},  desired {pitch_desired:.2f}, current {current_pitch:.2f}, error {pitch_error:.2f}, wrap {wrap_pitch:.2f}')

        if use_gimbal:
            yaw_speed = gimbal.pid_yaw.update(wrap_yaw)
            pitch_speed = gimbal.pid_pitch.update(wrap_pitch)

            #gimbal.move_speed(int(min(100, max(-100, yaw_speed))), 0)
            #gimbal.move_speed(0, int(min(100, max(-100, pitch_speed))))
            gimbal.move_speed(int(min(100, max(-100, yaw_speed))), int(min(100, max(-100, pitch_speed))))
        #print(f"Yaw speed: {yaw_speed}, Pitch speed: {pitch_speed}")
    else:
        if use_gimbal:
            gimbal.move_speed(0, 0)
        print("No marker detected")"""

    #cv2.imshow("A8 Mini Low Latency", frame)
    if cv2.waitKey(1) == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()