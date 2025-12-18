import cv2
from control_gimbal import GimbalCommand
import nFoldEdgeCodeDisk.MarkerLocator as ml
import time
import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"


gimbal = GimbalCommand()
gimbal.center_gimbal()
time.sleep(2)

# RTSP URL
url = "rtsp://192.168.144.25:8554/main.264"

# Force FFMPEG backend
cap = cv2.VideoCapture(
    url,
    cv2.CAP_FFMPEG
)

# Reduce buffering
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # keep only the latest frame

if not cap.isOpened():
    print("Could not open video stream")
    exit()

cd = ml.CameraDriver([5], default_kernel_size=13, scaling_parameter=1000, downscale_factor=2)  # Best in robolab.

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame lost")
        continue

    cd.current_frame = frame
    cd.process_frame()
    cd.draw_detected_markers()

    if cd.locations:
        marker = cd.locations[0]  # Take the first detected marker
        #print(f"Marker ID: {marker.id}, Position: ({marker.x}, {marker.y}), Angle: {marker.theta}")
        image_center_x = frame.shape[1] / 2
        error_yaw = marker.x * cd.downscale_factor - image_center_x
        image_center_y = frame.shape[0] / 2
        error_pitch = marker.y * cd.downscale_factor - image_center_y
        #print(f"Error Yaw: {error_yaw}, Error Pitch: {error_pitch}")
        # Command gimbal to point at marker
        current_yaw, current_pitch, _ = gimbal.get_attitude()
        yaw_error = current_yaw - error_yaw/25
        pitch_error = error_pitch - current_pitch/10

        wrap_yaw = (yaw_error + 180) % 360 - 180
        yaw_error = wrap_yaw
        wrap_pitch = (pitch_error + 180) % 360 - 180
        pitch_error = wrap_pitch

        yaw_speed = gimbal.pid_yaw.update(yaw_error)
        pitch_speed = gimbal.pid_pitch.update(pitch_error)

        gimbal.move_speed(int(min(100, max(-100, yaw_speed))), 0) #int(min(100, max(-100, pitch_speed))))
    else:
        gimbal.move_speed(0, 0)
        print("No marker detected")

    #cv2.imshow("A8 Mini Low Latency", frame)
    if cv2.waitKey(1) == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()