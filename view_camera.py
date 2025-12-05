import cv2

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

while True:
    ret, frame = cap.read()
    print("Frame shape:", frame.shape if ret else "No frame")
    if not ret:
        print("Frame lost")
        continue

    cv2.imshow("A8 Mini Low Latency", frame)
    if cv2.waitKey(1) == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
