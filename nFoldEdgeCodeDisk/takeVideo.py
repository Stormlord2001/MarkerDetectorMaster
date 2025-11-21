import cv2


# generate a script to take a video from the camera and save it to disk
class CameraDriver:
    def __init__(self, camera_index=0, output_filename='output.avi', fps=20.0, frame_size=(640, 480)):
        self.camera = cv2.VideoCapture(camera_index)
        self.output_filename = output_filename
        self.fps = fps
        self.frame_size = frame_size
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter(self.output_filename, self.fourcc, self.fps, self.frame_size)
        self.running = True
        self.frame_number = 0

        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, frame_size[0])
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_size[1])
        self.camera.set(cv2.CAP_PROP_FPS, fps)

        # Disable auto exposure
        self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        
        # Fast shutter (requires lots of light!)
        self.camera.set(cv2.CAP_PROP_EXPOSURE, -13)         # range: -1 .. -13 (lower = faster)

        # Keep gain low for less noise
        self.camera.set(cv2.CAP_PROP_GAIN, 0)

        # Disable autofocus (C270 is fixed focus)
        self.camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)

        # Optional: set focus (but C270 may ignore)
        self.camera.set(cv2.CAP_PROP_FOCUS, 0)

        # Disable auto gain
        #self.camera.set(cv2.CAP_PROP_GAIN, 0)

        # Disable auto white balance
        #self.camera.set(cv2.CAP_PROP_AUTO_WB, 0)
        #self.camera.set(cv2.CAP_PROP_WB_TEMPERATURE, 4500)

        # Manual brightness/contrast/saturation
        #self.camera.set(cv2.CAP_PROP_BRIGHTNESS, 100)
        #self.camera.set(cv2.CAP_PROP_CONTRAST, 50)
        #self.camera.set(cv2.CAP_PROP_SATURATION, 50)

    def get_image(self):
        print(f"Capturing frame {self.frame_number}...")
        ret, frame = self.camera.read()
        print(f"Shape of captured frame: {frame.shape}")
        if ret:
            self.current_frame = frame
            self.frame_number += 1
        else:
            self.running = False

    def save_frame(self):
        if hasattr(self, 'current_frame'):
            self.out.write(self.current_frame)

    def show_frame(self):
        if hasattr(self, 'current_frame'):
            cv2.imshow('frame', self.current_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False

    def release_resources(self):
        self.camera.release()
        self.out.release()
        cv2.destroyAllWindows()

def main():
    cd = CameraDriver(camera_index=2, output_filename='output.avi', fps=60.0, frame_size=(640, 480))

    while cd.running:
        cd.get_image()
        cd.save_frame()
        cd.show_frame()

    cd.release_resources()
    print("Video saved to", cd.output_filename)

main()