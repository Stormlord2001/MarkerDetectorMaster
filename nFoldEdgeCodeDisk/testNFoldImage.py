import numpy as np
import cv2

from MarkerTracker import MarkerTracker

 
tracker = MarkerTracker(order=5, kernel_size=31, scale_factor=1.0)

img = cv2.imread("generated_marker.png")

frame_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

poses = tracker.locate_marker(frame_gray) 

for pose in poses:
    print("Marker found at x: %.2f, y: %.2f, theta: %.2f, quality: %.2f" % (pose.x, pose.y, pose.theta, pose.quality))
    cv2.circle(img, (int(pose.x), int(pose.y)), 5, (0, 255, 0), -1)
cv2.imshow("Detected Markers", img)
cv2.waitKey(0)



