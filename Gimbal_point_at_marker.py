import cv2
from control_gimbal import GimbalCommand
import nFoldEdgeCodeDisk.MarkerLocator as ml
from nFoldEdgeCodeDisk.PoseEstimator import PoseEstimator
import time
import os
import math
import numpy as np
import cProfile


def main():
    use_gimbal = False

    PID_yaw = (6, 6/8, 1.0)   # P, I, D values for yaw
    PID_pitch = (5, 5/8, 1.0) # P, I, D values for pitch
    gimbal = GimbalCommand(PID_yaw=PID_yaw, PID_pitch=PID_pitch)
    if use_gimbal:
        gimbal.center_gimbal()
        time.sleep(2)


    # Camera intrinsics and distortion
    intrinsics = np.array([[835.4362078622368, 0, 323.0605420101571],
                            [0, 835.9483791851382, 232.14120929722597],
                            [0, 0, 1]], dtype=float)
    dist_coeffs = np.array([-0.0999921394506428, 2.185188066835036, -0.005726667745540125, 0.00027787706601120816, -7.636164458366145], dtype=float)


    marker_ids = [17, 27, 39, 119]
    marker_placements = {marker_ids[0]: (-0.495, -0.495, 0.0),
                        marker_ids[1]: (0.495, -0.495, 0.0),
                        marker_ids[2]: (0.495, 0.495, 0.0),
                        marker_ids[3]: (-0.495, 0.495, 0.0)}

    downsacle_factor = 1
    lp = PoseEstimator(intrinsics, dist_coeffs, marker_ids, marker_placements, alpha=0.5, max_reproj_error=10.0, downscale_factor=downsacle_factor)

    cd = ml.CameraDriver([5], marker_ids=marker_ids, default_kernel_size=int(13/downsacle_factor), scaling_parameter=1000, downscale_factor=downsacle_factor, VideoFile=0)#, VideoFile="output.avi") 

    t0 = time.time()
    total_frames = 0
    total_time = 0

    while True:
        (t1, t0) = (t0, time.time())
        print("time for one iteration: %f" % (t0 - t1))
        total_time += (t0 - t1)
        total_frames += 1
        

        cd.get_image()
        cd.process_frame()
        
        if cd.locations:
            test = lp.estimate_load_pose(cd.locations)
            #print("Estimated load position (x, y, z, roll, pitch, yaw): ", lp.load_position)
            lp.display_pose(cd.current_frame, axis_length=0.05)
        
            if test is not None:

                tvec = test[1].ravel()
                # calc roll and pitch from tvec
                x, y, z = tvec[0], tvec[1], tvec[2] # in camera frame to payload frame
                pitch = math.atan2(y, z) * (180 / math.pi) # pitch in gimbal, roll in camera frame
                yaw = -math.atan2(x, z) * (180 / math.pi) # yaw in gimbal, -pitch in camera frame


                #print(f'pitch: {pitch: 6.2f}, yaw: {yaw: 6.2f}')
            
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
                    gimbal.move_speed(int(min(100, max(-100, yaw_speed))), int(min(100, max(-100, pitch_speed))))
                    
            else:
                if use_gimbal:
                    gimbal.move_speed(0, 0)
        else:
            if use_gimbal:
                gimbal.move_speed(0, 0)
            print("No marker detected")

        
        cd.draw_detected_markers()

        #cv2.imshow("A8 Mini Low Latency", frame)
        if cv2.waitKey(1) == 27:  # ESC to exit
            cd.camera.release()
            break

    print("Average time per frame: %f" % (total_time / total_frames))
    print("average fps: %f" % (total_frames / total_time))
    print("Stopping")

    #cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    cProfile.run('main()')