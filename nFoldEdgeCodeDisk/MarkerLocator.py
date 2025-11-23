from time import time

import cv2
import numpy as np 
import math

from MarkerPose import MarkerPose
from MarkerTracker import MarkerTracker
from PoseEstimator import PoseEstimator

# parameters
show_image = True
print_debug_messages = False
print_iteration_time = True
check_keystroke = True
list_of_markers_to_find = [5]
grayscale_with_mahalanobis = False
grayscale_with_hsv = False
color = "magenta"  # only used if grayscale_with_hsv is True

# camera intrinsics (fx, fy, cx, cy) and distortion
camera_matrix = np.array([[6195.77376865, 0, 4078.14040656],
                          [0, 6192.15216282, 2997.84935944],
                          [0, 0, 1]], dtype=float)
#dist_coeffs = np.zeros(5)  # or the real distortion values
dist_coeffs = np.array([-0.0550747, -0.05581563, -0.00113507, 0.0002599, 0.10360782], dtype=float)



class CameraDriver:
    """
    Purpose: capture images from a camera and delegate procesing of the
    images to a different class.
    """

    def __init__(self, marker_orders=[6], default_kernel_size=30, scaling_parameter=2500, downscale_factor=1):
        # Initialize camera driver.
        # Open output window.
        if show_image is True:
            cv2.namedWindow('filterdemo', cv2.WINDOW_AUTOSIZE)

        # Select the camera where the images should be grabbed from.
        #set_camera_focus()
        #self.camera = cv2.VideoCapture("Videos/.mp4")
        #self.camera = cv2.VideoCapture("Videos/markersRotatingPulsing.mp4")
        #self.camera = cv2.VideoCapture("nFoldEdgeCodeDisk/output.avi")
        self.camera = cv2.VideoCapture("output.avi")
        #self.set_camera_resolution()

        # Storage for image processing.
        self.current_frame = None
        self.processed_frame = None
        self.running = True
        self.downscale_factor = downscale_factor

        # Storage for trackers.
        self.trackers = []
        self.old_locations = []

        # Initialize trackers.
        for marker_order in marker_orders:
            temp = MarkerTracker(marker_order, default_kernel_size, scaling_parameter, downscale_factor)
            self.trackers.append(temp)
            self.old_locations.append(MarkerPose(None, None, None, None, None))

    def set_camera_resolution(self):
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    def get_image(self):
        self.current_frame = self.camera.read()[1]

    def process_frame(self):

        # Convert to grayscale.
        frame_gray = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)

        # binary thresholding with Otsu's method
        #_, frame_gray = cv2.threshold(frame_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        
        #cv2.imshow("mask", frame_gray)

        # Denoising https://www.youtube.com/watch?v=xtRY_iT41U4 
        #frame_gray = cv2.medianBlur(frame_gray, 5)


        reduced_image = cv2.resize(frame_gray, (0, 0), fx=1.0/self.downscale_factor, fy=1.0 / self.downscale_factor)

        self.locations = []
        for k in range(len(self.trackers)):
            poses = self.trackers[k].locate_marker(reduced_image)
            for pose in poses:
                self.locations.append(pose)

        self.old_locations = self.locations

    def draw_detected_markers(self):
        if show_image is True:
            display_frame = self.current_frame.copy()
            for pose in self.locations:
                x = int(pose.x * self.downscale_factor)
                y = int(pose.y * self.downscale_factor)
                cv2.circle(display_frame, (x, y), 10, (0, 255, 0), 2)
                # might need to be 50, but think it is just the length of the line
                #cv2.line(display_frame, (x, y), (int(x + 20 * math.cos(pose.theta)), int(y + 20 * math.sin(pose.theta))), (255, 0, 0), 2)
                #cv2.putText(display_frame, f"{pose.quality:0.2f}", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(display_frame, f"{pose.id}", (x + 10, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            
            display_frame = cv2.resize(display_frame, (0, 0), fx=1.0/1, fy=1.0/1)
            cv2.imshow('filterdemo', display_frame)



def main():
    # 3840x2160 video: default_kernel_size=73, scaling_parameter=1000, downscale_factor=1

    cd = CameraDriver(list_of_markers_to_find, default_kernel_size=25, scaling_parameter=1000, downscale_factor=1 )  # Best in robolab.
    # cd = ImageDriver(list_of_markers_to_find, defaultKernelSize = 21) 
    t0 = time()

    while cd.running:
        (t1, t0) = (t0, time())
        if print_iteration_time is True:
            print("time for one iteration: %f" % (t0 - t1))
        cd.get_image()
        cd.process_frame()
        cd.draw_detected_markers()
        
        if check_keystroke is True:
            key = cv2.waitKey(10000)
            if key == 27:  # Esc
                    cd.running = False
            # save frame when s is pressed
            if key == ord('s'):
                cv2.imwrite(f"output/frame_{int(time())}.png", cd.current_frame)
        """cd.show_processed_frame() 
        cd.handle_keyboard_events()
        y = cd.return_positions()
        for k in range(len(y)):
            try:
                # pose_corrected = perspective_corrector.convertPose(y[k])
                pose_corrected = y[k]
                print("%8.3f %8.3f %8.3f %8.3f %s" % (pose_corrected.x,
                                                        pose_corrected.y,
                                                        pose_corrected.theta,
                                                        pose_corrected.quality,
                                                        pose_corrected.order))
            except Exception as e:
                print("%s" % e)"""
            

    print("Stopping")

main()