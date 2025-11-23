from time import time

import cv2
import numpy as np
import math

from MarkerPose import MarkerPose
from MarkerTracker import MarkerTracker
from PoseEstimator import PoseEstimator

# parameters
show_image = False
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


class Mahalanobis:
    def __init__(self, mean_BGR, cov_matrix):
        self.mean_BGR = mean_BGR
        self.cov_inv = np.linalg.inv(cov_matrix)

    def mahalanobis(self, image):
        img = cv2.GaussianBlur(image, (5, 5), 0)
        pixels = np.reshape(img, (-1, 3))
        diff = pixels - self.mean_BGR
        moddotprod = diff * (diff @ self.cov_inv)
        mahalanobis_dist = np.sum(moddotprod, axis=1)
        distance = np.sqrt(np.reshape(mahalanobis_dist, (img.shape[0], img.shape[1])))

        return distance


class LoadPosition:
    def __init__(self, downscale_factor=1.0):
        self.marker_angle_offset = [53, 0, 22, 36] #[60, 45, 70, 0]#[65, 0, 21, 43]
        self.load_angle = 0
        self.angular_limit = 5

        self.marker_positions = []
        self.marker_1 = []
        self.marker_2 = []
        self.marker_3 = []
        self.marker_4 = []

        self.marker_seen = [False, False, False, False]

        self.marker_timer = [0, 0, 0, 0]

        self.period = 360 / list_of_markers_to_find[0]

        self.downscale_factor = downscale_factor
        self.PE = PoseEstimator(camera_matrix, dist_coeffs, alpha=0.5, max_reproj_error=10.0, downscale_factor=downscale_factor)
        self.marker_placements = {0: (-0.11, -0.11, 0.0),
                                  1: (0.11, -0.115, 0.0),
                                  2: (-0.115, 0.11, 0.0),
                                  3: (0.12, 0.12, 0.0)}
        self.marker_detections = {}

    def organize_markers_angle(self, locations):
        self.marker_positions = []
        self.marker_1 = []
        self.marker_2 = []
        self.marker_3 = []
        self.marker_4 = []
        

        for pose in locations:
            angle = pose.theta * 180 / math.pi
            if pose.quality < 0.01:
                continue

            for i in range(len(self.marker_angle_offset)):
                if abs(self.angular_distance(angle, (self.marker_angle_offset[i] + self.load_angle))) < self.angular_limit:
                    getattr(self, f"marker_{i+1}").append(pose)
                    #print(f"Angle: {angle:0.1f} deg fits marker {i+1} at {getattr(self, f'marker_{i+1}_angle')} deg, with diff {self.angular_distance(angle, getattr(self, f'marker_{i+1}_angle'), period=90):0.1f} deg")
                    break

        for i in range(len(self.marker_angle_offset)):
            marker_list = getattr(self, f"marker_{i+1}")
            if len(marker_list) == 0:
                self.marker_positions.append(MarkerPose(0, 0, 0, 0, 0))
            elif len(marker_list) == 1:
                self.marker_positions.append(marker_list[0])
            else:
                quality = []
                for pose in marker_list:
                    quality.append(pose.quality)
                
                # sort locations based on quality
                marker_list = [x for _, x in sorted(zip(quality, marker_list), key=lambda pair: pair[0], reverse=True)]
                self.marker_positions.append(marker_list[0])
        
        #print(f"Marker 1 candidates: {len(self.marker_1)}")
        #print(f"Marker 2 candidates: {len(self.marker_2)}")
        #print(f"Marker 3 candidates: {len(self.marker_3)}")
        #print(f"Marker 4 candidates: {len(self.marker_4)}")

        # update marker seen status
        for i in range(len(self.marker_angle_offset)):
            if self.marker_positions[i].quality == 0:
                self.marker_seen[i] = False
            else:
                self.marker_seen[i] = True

        # update marker angles
        divisor = 0
        angle_sum = 0
        for i in range(len(self.marker_angle_offset)):
            # only update if we have a valid marker
            if self.marker_seen[i] is True:
                angle = self.limit_angle(self.marker_positions[i].theta * 180 / math.pi)
                angle_diff = self.angular_distance(angle, self.marker_angle_offset[i] + self.load_angle)
                angle_sum += angle_diff
                divisor += 1
        self.load_angle = (angle_sum / divisor) + self.load_angle if divisor > 0 else self.load_angle
        if print_debug_messages is True:
            print(f"Load angle: {self.load_angle:0.1f} deg")
        
        

        for i in range(len(self.marker_angle_offset)):
            if self.marker_seen[i] is True:
                self.marker_timer[i] = 0
            else:
                self.marker_timer[i] += 1

        #self.marker_detections = {i: self.marker_positions[i] for i in range(len(self.marker_positions)) if self.marker_seen[i] is True}
        self.marker_detections = {i: (self.marker_positions[i].x*self.downscale_factor, self.marker_positions[i].y*self.downscale_factor) for i in range(len(self.marker_positions)) if self.marker_seen[i] is True}

        if print_debug_messages is True:
            print(f"Marker detections for pose estimation: {len(self.marker_detections)} markers")
            print(self.marker_detections)

        if len(self.marker_detections) >= 3:
            rvec, tvec, R, cam_pos, inliers = self.PE.estimate_pose(
                self.marker_placements,
                self.marker_detections
            )
            if print_debug_messages is True:
                print("Estimated pose:")
                print("rvec:", rvec.ravel())
                print("tvec:", tvec.ravel())
                print("camera pos (object frame):", cam_pos.ravel())

        #for i in range(len(self.marker_angle_offset)):
        #    print(f"Marker {i+1} timer: {self.marker_timer[i]}")

    def organize_markers_color(self, locations, frame):
        self.marker_positions = []
        self.marker_1 = []
        self.marker_2 = []
        self.marker_3 = []
        self.marker_4 = []

        for pose in locations:
            # get color at marker position
            x = int(pose.x * self.downscale_factor)
            y = int(pose.y * self.downscale_factor)

            #convert to hsv
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hue = hsv[y, x, 0]
            saturation = hsv[y, x, 1]
            value = hsv[y, x, 2]

            # red: hue ~0°, orange: hue ~30°, blue: hue ~120°, magenta: hue ~150°
            if saturation < 50 or value < 50:
                continue

            if hue < 10 or hue > 165:
                pose.color = "red"
                self.marker_1.append(pose)
            elif 10 < hue < 45:
                pose.color = "orange"
                self.marker_2.append(pose)
            elif 100 < hue < 135:
                pose.color = "blue"
                self.marker_3.append(pose)
            elif 135 < hue < 165:
                pose.color = "magenta"
                self.marker_4.append(pose)

        for i in range(4):
            marker_list = getattr(self, f"marker_{i+1}")
            if len(marker_list) == 0:
                self.marker_positions.append(MarkerPose(0, 0, 0, 0, 0))
            elif len(marker_list) == 1:
                self.marker_positions.append(marker_list[0])
            else:
                quality = []
                for pose in marker_list:
                    quality.append(pose.quality)
                
                # sort locations based on quality
                marker_list = [x for _, x in sorted(zip(quality, marker_list), key=lambda pair: pair[0], reverse=True)]
                self.marker_positions.append(marker_list[0])

    def angular_distance(self, angle, reference):
        """
        Compute the smallest signed angular difference between two angles (in degrees),
        wrapping every `period` degrees (default 90°).
        Returns a value in [-period/2, +period/2).
        """
        diff = (angle - reference + self.period / 2) % self.period - self.period / 2
        return diff
    
    def limit_angle(self, angle):
        """
        Limit angle to [0, period)
        """
        return angle % self.period

class CameraDriver:
    """
    Purpose: capture images from a camera and delegate procesing of the
    images to a different class.
    """

    def __init__(self, marker_orders=[6], default_kernel_size=21, scaling_parameter=2500, downscale_factor=1):
        # Initialize camera driver.
        # Open output window.
        if show_image is True:
            cv2.namedWindow('filterdemo', cv2.WINDOW_AUTOSIZE)

        # Select the camera where the images should be grabbed from.
        #set_camera_focus()
        #self.camera = cv2.VideoCapture("grass.mp4")
        #self.camera = cv2.VideoCapture("white_arms.mp4")
        self.camera = cv2.VideoCapture("Videos/3840x2160.mp4")
        #self.camera = cv2.VideoCapture("1920x1080.mp4")
        #self.camera = cv2.VideoCapture("color_center.mp4")
        #self.set_camera_resolution()

        # Storage for image processing.
        self.current_frame = None
        self.processed_frame = None
        self.running = True
        self.downscale_factor = downscale_factor

        # load position
        self.load_position = LoadPosition(downscale_factor=downscale_factor)

        # Storage for trackers.
        self.trackers = []
        self.old_locations = []

        if grayscale_with_mahalanobis is True:
            mean_and_cov = np.loadtxt("mean_and_cov.csv", delimiter=",")
            mean_BGR = mean_and_cov[0]
            cov_matrix = mean_and_cov[1:4, :]
            self.mahalanobis_converter = Mahalanobis(mean_BGR, cov_matrix)
            self.threshold_value =  17

        # Initialize trackers.
        for marker_order in marker_orders:
            temp = MarkerTracker(marker_order, default_kernel_size, scaling_parameter)
            self.trackers.append(temp)
            self.old_locations.append(MarkerPose(None, None, None, None, None))

    def set_camera_resolution(self):
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    def get_image(self):
        self.current_frame = self.camera.read()[1]

    def process_frame(self):

        # Convert to grayscale.
        if grayscale_with_mahalanobis is True:
            frame_gray = self.mahalanobis_converter.mahalanobis(self.current_frame)
            if print_debug_messages is True:
                print(f"Max Mahalanobis distance in frame: {np.max(frame_gray):0.2f}")
            frame_gray = np.where(frame_gray < self.threshold_value, 255, 0).astype(np.uint8)
        elif grayscale_with_hsv is True:
            hsv = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2HSV)

            if color == "orange":
                # Define orange color range (tune if needed)
                lower_orange = np.array([5, 75, 75])   # Hue ~5°, avoid too low saturation/value
                upper_orange = np.array([15, 255, 255])
            elif color ==  "magenta":
                lower_orange = np.array([150, 50, 50])  # Hue ~150°, avoid too low saturation/value
                upper_orange = np.array([170, 255, 255])
            elif color == "blue":
                lower_orange = np.array([100, 75, 55])  # Hue ~120°, avoid too low saturation/value
                upper_orange = np.array([140, 255, 255])
            else:
                raise ValueError("Unsupported color")

            # Threshold to get only orange pixels
            frame_gray = cv2.inRange(hsv, lower_orange, upper_orange)
        else:
            frame_gray = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)
        
        
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
                cv2.line(display_frame, (x, y), (int(x + 20 * math.cos(pose.theta)), int(y + 20 * math.sin(pose.theta))), (255, 0, 0), 2)
                cv2.putText(display_frame, f"{pose.quality:0.2f}", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(display_frame, f"{self.load_position.limit_angle(pose.theta*180/math.pi):0.1f} deg", (x + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


            for pose in self.load_position.marker_positions:
                if pose.x == 0 and pose.y == 0:
                    continue
                cv2.circle(display_frame, (int(pose.x * self.downscale_factor), int(pose.y * self.downscale_factor)), 15, (0, 0, 255), 3)
                cv2.putText(display_frame, f"{pose.quality:0.2f}", (int(pose.x * self.downscale_factor) + 15, int(pose.y * self.downscale_factor) - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(display_frame, f"{self.load_position.limit_angle(pose.theta*180/math.pi):0.1f} deg", (int(pose.x * self.downscale_factor) + 15, int(pose.y * self.downscale_factor) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                #cv2.putText(display_frame, f"{pose.color}", (int(pose.x * self.downscale_factor) + 15, int(pose.y * self.downscale_factor) + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            display_frame = cv2.resize(display_frame, (0, 0), fx=1.0/2, fy=1.0/2)
            cv2.imshow('filterdemo', display_frame)



def main():
    # 3840x2160 video: default_kernel_size=73, scaling_parameter=1000, downscale_factor=1

    cd = CameraDriver(list_of_markers_to_find, default_kernel_size=13, scaling_parameter=1000, downscale_factor=5)  # Best in robolab.
    # cd = ImageDriver(list_of_markers_to_find, defaultKernelSize = 21) 
    t0 = time()

    while cd.running:
        (t1, t0) = (t0, time())
        if print_iteration_time is True:
            print("time for one iteration: %f" % (t0 - t1))
        cd.get_image()
        cd.process_frame()
        cd.load_position.organize_markers_angle(cd.locations)
        #cd.load_position.organize_markers_color(cd.locations, cd.current_frame)
        cd.load_position.PE.display_pose(cd.current_frame, axis_length=0.05)
        cd.draw_detected_markers()
        
        #for pose in cd.load_position.marker_positions:
        #    print(f"Marker at x: {pose.x:0.1f}, y: {pose.y:0.1f}")
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