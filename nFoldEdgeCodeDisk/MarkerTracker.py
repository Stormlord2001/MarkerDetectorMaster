
import cv2
import numpy as np
import math
from MarkerPose import MarkerPose
from decode import decode_marker


class MarkerTracker:
    def __init__(self, order, kernel_size, scale_factor, downscale_factor=1):
        self.kernel_size = kernel_size
        (kernel_real, kernel_imag) = self.generate_symmetry_detector_kernel(order, kernel_size)

        self.order = order
        self.mat_real = kernel_real / scale_factor
        self.mat_imag = kernel_imag / scale_factor

        # showing the kernels
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(self.mat_real)
        #cv2.imshow("mat_real_norm", 255*self.mat_real/max_val)

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(self.mat_imag)
        #cv2.imshow("mat_imag_norm", 255*self.mat_imag/max_val)

        # Values used in quality-measure
        self.kernelComplex = np.array(kernel_real + 1j*kernel_imag, dtype=complex)
        absolute = np.absolute(self.kernelComplex)
        self.threshold = 0.4*absolute.max()
        self.quality = None
        self.y1 = int(math.floor(float(self.kernel_size)/2))
        self.y2 = int(math.ceil(float(self.kernel_size)/2))
        self.x1 = int(math.floor(float(self.kernel_size)/2))
        self.x2 = int(math.ceil(float(self.kernel_size)/2))

        # Information about the located marker.
        self.pose = None

        # Using codering to id markers
        r_code_inner = int(21/downscale_factor)
        self.r_code_outer = int(32/downscale_factor)
        bits = 8
        transitions = 2
        self.decoder = decode_marker(r_code_inner, self.r_code_outer, bits, transitions)

    @staticmethod
    def generate_symmetry_detector_kernel(order, kernel_size):
        # type: (int, int) -> numpy.ndarray
        value_range = np.linspace(-1, 1, kernel_size)
        temp1 = np.meshgrid(value_range, value_range)
        kernel = temp1[0] + 1j * temp1[1]

        magnitude = abs(kernel)
        kernel = np.power(kernel, order)
        kernel = kernel * np.exp(-8 * magnitude ** 2)

        return np.real(kernel), np.imag(kernel)
    
    def locate_marker(self, frame):
        assert len(frame.shape) == 2, "Input image is not a single channel image."
        frame_real = frame.copy()
        frame_imag = frame.copy()

        # Convolve image with kernels.
        frame_real = cv2.filter2D(frame_real, cv2.CV_32F, self.mat_real)
        frame_imag = cv2.filter2D(frame_imag, cv2.CV_32F, self.mat_imag)
        frame_real_squared = cv2.multiply(frame_real, frame_real, dtype=cv2.CV_32F)
        frame_imag_squared = cv2.multiply(frame_imag, frame_imag, dtype=cv2.CV_32F)
        frame_sum_squared = cv2.add(frame_real_squared, frame_imag_squared, dtype=cv2.CV_32F)

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(frame_sum_squared)
        
        ###cv2.imshow("frame", frame)
        frame_sum_squared_max_circled = cv2.circle(frame_sum_squared.copy(), max_loc, self.kernel_size//2, (255, 255, 255), 2)
        ###cv2.imshow("frame_sum_squared_max_circled", 255*frame_sum_squared_max_circled)

        threshold_value = max_val * 0.5

        thres_img = np.where(frame_sum_squared > threshold_value, frame_sum_squared, 0)
        min_val, max_val_thresh, min_loc, max_loc_thresh = cv2.minMaxLoc(thres_img)
       


        frame_sum_text = cv2.putText(frame_sum_squared.copy()*10000, f"max val: {max_val*10000}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        #cv2.imshow("combined", np.vstack((np.hstack((100*frame_real, 100*frame_imag)),np.hstack((10000*frame_real_squared, 10000*frame_imag_squared)), np.hstack((frame_sum_text, thres_img*10000)))))

        ##cv2.imshow("frame_sum_squared_norm", 1*255*frame_sum_squared)
        ##cv2.imshow("thres_img_norm", 255*thres_img/max_val_thresh)

        # extract conturs
        contours, hierarchy = cv2.findContours(np.uint8(thres_img/max_val_thresh*255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_img = cv2.cvtColor(np.uint8(frame), cv2.COLOR_GRAY2BGR)

        poses = []

        #for contour in contours:
        #    cv2.drawContours(contour_img, [contour], -1, (0, 255, 0), 2)
        #cv2.imshow("contours", contour_img)
        #cv2.waitKey(0)

        if len(contours) > 10:
            print(f"Too many contours detected ({len(contours)}), skipping frame.")
            return []

        print(f"length contours: {len(contours)}")
        for contour in contours:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            if radius > 2:
                continue
            if x - self.r_code_outer < 0 or x + self.r_code_outer + 1 >= frame.shape[1]:
                continue
            if y - self.r_code_outer < 0 or y + self.r_code_outer + 1 >= frame.shape[0]:
                continue
            #marker_size = 200
            #marker = frame[int(y)-marker_size:int(y)+marker_size, int(x)-marker_size:int(x)+marker_size]
            
            #min_val_c, max_val_c, min_loc_c, max_loc_c = cv2.minMaxLoc(marker)
            #max_loc_c = (max_loc_c[0] + x, max_loc_c[1] + y)

            
            frame_sum_cutout = self.extract_window_around_marker_location(frame_sum_squared, (int(x), int(y)))
            #print("frame_sum_cutout shape:", frame_sum_cutout.shape)
            #print(frame_sum_cutout)
            min_val_c, max_val_c, min_loc_c, max_loc_c = cv2.minMaxLoc(frame_sum_cutout)
            #print(f"max_loc_c before refining: {max_loc_c}, x: {x}, y: {y}")
            (dx, dy) = self.refine_marker_location_new(frame_sum_squared, max_loc_c[0], max_loc_c[1])
            refined_location = (max_loc_c[0] + dx, max_loc_c[1] + dy)
            #print(f"refined location: {refined_location}, max_loc_c: {max_loc_c}, dx: {dx}, dy: {dy}")

            # Decode the marker ID
            #marker_id = self.decoder.extract_and_decode(frame, (int(refined_location[0]), int(refined_location[1])))
            marker_id = self.decoder.extract_and_decode(frame, (int(x), int(y)))
            
            if marker_id is None:
                continue

            orientation = 0 #self.determine_marker_orientation(frame, frame_real, frame_imag, max_loc_c)
            quality = 0 #self.determine_marker_quality(frame, orientation, max_loc_c)
            pose = MarkerPose(x, y, orientation, quality, self.order)
            pose.id = marker_id

            poses.append(pose)

        
        return poses

    def determine_marker_orientation(self, frame, frame_real, frame_imag, marker_loc):
        (xm, ym) = marker_loc
        real_value = frame_real[ym, xm]
        imag_value = frame_imag[ym, xm]
        orientation = (math.atan2(-real_value, imag_value) - math.pi / 2) / self.order

        #print(f"real_value: {real_value}  imag_value: {imag_value}  orientation: {orientation*180/math.pi:0.1f} deg")

        max_value = 0
        max_orientation = orientation
        search_distance = self.kernel_size / 3
        for k in range(self.order):
            orient = orientation + 2 * k * math.pi / self.order
            xm2 = int(xm + search_distance * math.cos(orient))
            ym2 = int(ym + search_distance * math.sin(orient))
            try:
                intensity = frame[ym2, xm2]
                if intensity > max_value:
                    max_value = intensity
                    max_orientation = orient
            except Exception as e:
                #print("determineMarkerOrientation: error: %d %d %d %d" % (ym2, xm2, frame.shape[0], frame.shape[1]))
                #print(f"xm: {xm}  xm2: {xm2}  framex: {frame.shape[1]}")
                #print(f"ym: {ym}  ym2: {ym2}  framey: {frame.shape[0]}")
                #print(e)
                pass

        orientation = self.limit_angle_to_range(max_orientation)
        return orientation

    @staticmethod
    def limit_angle_to_range(angle):
        while angle < math.pi:
            angle += 2 * math.pi
        while angle > math.pi:
            angle -= 2 * math.pi
        return angle
    
    def determine_marker_quality(self, frame, orientation, marker_loc):
        (bright_regions, dark_regions) = self.generate_template_for_quality_estimator(orientation)
        # cv2.imshow("bright_regions", 255*bright_regions)
        # cv2.imshow("dark_regions", 255*dark_regions)

        try:
            frame_img = self.extract_window_around_marker_location(frame, marker_loc)
            (bright_mean, bright_std) = cv2.meanStdDev(frame_img, mask=bright_regions)
            (dark_mean, dark_std) = cv2.meanStdDev(frame_img, mask=dark_regions)

            mean_difference = bright_mean - dark_mean
            normalised_mean_difference = mean_difference / (0.5*bright_std + 0.5*dark_std)
            # Ugly hack for translating the normalised_mean_differences to the range [0, 1]
            temp_value_for_quality = 1 - 1/(1 + math.exp(0.75*(-7+normalised_mean_difference)))
            quality = temp_value_for_quality
        except Exception as e:
            #print("error")
            #print(e)
            quality = 0.0
            
        return quality
        
    def extract_window_around_marker_location(self, frame, marker_loc):
        (xm, ym) = marker_loc
        frame_tmp = np.array(frame[ym - self.y1:ym + self.y2, xm - self.x1:xm + self.x2])
        #frame_img = frame_tmp.astype(np.uint8)
        return frame_tmp

    def generate_template_for_quality_estimator(self, orientation):
        phase = np.exp((self.limit_angle_to_range(-orientation)) * 1j)
        #angle_threshold = 3.14 / (2 * self.order)
        #t3 = np.angle(self.KernelRemoveArmComplex * phase) < angle_threshold
        #t4 = np.angle(self.KernelRemoveArmComplex * phase) > -angle_threshold

        #signed_mask = 1 - 2 * (t3 & t4)
        adjusted_kernel = self.kernelComplex * np.power(phase, self.order)
        #if self.track_marker_with_missing_black_leg:
        #    adjusted_kernel *= signed_mask
        bright_regions = (adjusted_kernel.real < -self.threshold).astype(np.uint8)
        dark_regions = (adjusted_kernel.real > self.threshold).astype(np.uint8)

        return bright_regions, dark_regions

    def refine_marker_location(self, frame_sum_squared):
        try: 
            delta = 1
            # Fit a parabola to the frame_sum_squared marker response
            # and then locate the top of the parabola.
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(frame_sum_squared)
            x = max_loc[1]
            y = max_loc[0]
            frame_sum_squared_cutout = frame_sum_squared[x-delta:x+delta+1, y-delta:y+delta+1]
            # Taking the square root of the frame_sum_squared improves the accuracy of the 
            # refied marker position.
            frame_sum_squared_cutout = np.sqrt(frame_sum_squared_cutout)

            nx, ny = (1 + 2*delta, 1 + 2*delta)
            x = np.linspace(-delta, delta, nx)
            y = np.linspace(-delta, delta, ny)
            xv, yv = np.meshgrid(x, y)

            xv = xv.ravel()
            yv = yv.ravel()

            coefficients = np.concatenate([[xv**2], [xv], [yv**2], [yv], [yv**0]], axis = 0).transpose()
            values = frame_sum_squared_cutout.ravel().reshape(-1, 1)
            solution, residuals, rank, s = np.linalg.lstsq(coefficients, values, rcond=None)
            dx = -solution[1] / (2*solution[0])
            dy = -solution[3] / (2*solution[2])
            return dx[0], dy[0]
        except np.linalg.LinAlgError as e:
            # This error is triggered when the marker is detected close to an edge.
            # In that case the refine method bails out and returns two zeros.
            print("error in refine_marker_location")
            print(e)
            return 0, 0

    def refine_marker_location_new(self, frame_sum_squared, x, y):
        try: 
            delta = 1
            # Fit a parabola to the frame_sum_squared marker response
            # and then locate the top of the parabola.
            frame_sum_squared_cutout = frame_sum_squared[x-delta:x+delta+1, y-delta:y+delta+1]
            #print("shape frame_sum_squared_cutout:", frame_sum_squared_cutout.shape)

            # Taking the square root of the frame_sum_squared improves the accuracy of the 
            # refied marker position.
            frame_sum_squared_cutout = np.sqrt(frame_sum_squared_cutout)

            nx, ny = (1 + 2*delta, 1 + 2*delta)
            x = np.linspace(-delta, delta, nx)
            y = np.linspace(-delta, delta, ny)
            xv, yv = np.meshgrid(x, y)

            xv = xv.ravel()
            yv = yv.ravel()

            coefficients = np.concatenate([[xv**2], [xv], [yv**2], [yv], [yv**0]], axis = 0).transpose()
            values = frame_sum_squared_cutout.ravel().reshape(-1, 1)
            solution, residuals, rank, s = np.linalg.lstsq(coefficients, values, rcond=None)
            dx = -solution[1] / (2*solution[0])
            dy = -solution[3] / (2*solution[2])
            return dx[0], dy[0]
        except np.linalg.LinAlgError as e:
            # This error is triggered when the marker is detected close to an edge.
            # In that case the refine method bails out and returns two zeros.
            print("error in refine_marker_location")
            print(e)
            return 0, 0



