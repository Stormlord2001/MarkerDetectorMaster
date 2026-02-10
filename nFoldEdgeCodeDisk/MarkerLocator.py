from time import time
import cv2

try:
    from MarkerTracker import MarkerTracker
except ImportError:
    from nFoldEdgeCodeDisk.MarkerTracker import MarkerTracker

# parameters
show_image = True
print_iteration_time = True
check_keystroke = True
list_of_markers_to_find = [5]

class CameraDriver:
    """
    Purpose: capture images from a camera and delegate procesing of the
    images to a different class.
    """

    def __init__(self, marker_orders=[6], marker_ids = [], default_kernel_size=30, scaling_parameter=2500, downscale_factor=1, VideoFile=1):
        # Initialize camera driver.
        # Open output window.
        if show_image is True:
            cv2.namedWindow('filterdemo', cv2.WINDOW_AUTOSIZE)

        # Select the camera where the images should be grabbed from.
        self.camera = cv2.VideoCapture(VideoFile)
        #self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'BGR3'))
        self.set_camera_resolution()
        # Reduce buffering
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # keep only the latest frame
        # Disable auto exposure
        self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        # Fast shutter (requires lots of light!)
        self.camera.set(cv2.CAP_PROP_EXPOSURE, -8)         # range: -1 .. -13 (lower = faster)
        # Keep gain low for less noise
        self.camera.set(cv2.CAP_PROP_GAIN, 200)
        if not self.camera.isOpened():
            print("Could not open video stream")
            exit()


        # Storage for image processing.
        self.current_frame = None
        self.processed_frame = None
        self.running = True
        self.downscale_factor = downscale_factor

        # Storage for trackers.
        self.trackers = []

        # Initialize trackers.
        for marker_order in marker_orders:
            temp = MarkerTracker(marker_order, default_kernel_size, scaling_parameter, marker_ids, downscale_factor)
            self.trackers.append(temp)

    def set_camera_resolution(self):
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    def get_image(self):
        self.current_frame = self.camera.read()[1]

    def process_frame(self):
        # Convert to grayscale.
        frame_gray = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)

        # Downscale image for faster processing.
        reduced_image = cv2.resize(frame_gray, (0, 0), fx=1.0/self.downscale_factor, fy=1.0 / self.downscale_factor)
        print(f"reduced_image shape: {reduced_image.shape}")
        self.locations = []
        for k in range(len(self.trackers)):
            poses = self.trackers[k].locate_marker(reduced_image)
            for pose in poses:
                self.locations.append(pose)

    def draw_detected_markers(self):
        if show_image is True:
            display_frame = self.current_frame.copy()
            for pose in self.locations:
                x = int(pose.x * self.downscale_factor)
                y = int(pose.y * self.downscale_factor)
                cv2.circle(display_frame, (x, y), 10, (0, 255, 0), 2)
                cv2.putText(display_frame, f"{pose.id}", (x + 10, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            
            display_frame = cv2.resize(display_frame, (0, 0), fx=1.0/1, fy=1.0/1)
            cv2.imshow('filterdemo', display_frame)



def main():
    # 3840x2160 video: default_kernel_size=73, scaling_parameter=1000, downscale_factor=1

    cd = CameraDriver(list_of_markers_to_find, default_kernel_size=25, scaling_parameter=1000, downscale_factor=1 )  # Best in robolab.
    # cd = ImageDriver(list_of_markers_to_find, defaultKernelSize = 21) 
    t0 = time()

    total_frames = 0
    total_time = 0

    while cd.running:
        (t1, t0) = (t0, time())
        if print_iteration_time is True:
            print("time for one iteration: %f" % (t0 - t1))
            total_time += (t0 - t1)
            total_frames += 1
        cd.get_image()
        if cd.current_frame is None:
            print("No more frames to read from camera/video.")
            break
        cd.process_frame()
        cd.draw_detected_markers()
        
        if check_keystroke is True:
            key = cv2.waitKey(1000000)
            if key == 27:  # Esc
                    cd.running = False
            # save frame when s is pressed
            if key == ord('s'):
                cv2.imwrite(f"output/frame_{int(time())}.png", cd.current_frame)
            
    print("Average time per frame: %f" % (total_time / total_frames))
    print("average fps: %f" % (total_frames / total_time))
    print("Stopping")

if __name__ == "__main__":
    main()