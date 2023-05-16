import csv
import os
import time

import cv2
import mediapipe as mp
import numpy as np

from utils.fps import FPS
from utils.thread_utils import FileVideoStream

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

if __name__ == "__main__":
    is_export = False
    # FALL or ADL ?
    root_dir = r'../selfmade_dataset/'
    video_list = os.listdir(root_dir)
    for video_name in video_list:
        video_path = root_dir + video_name
        print("Current progress --- " + video_path)

        fvs = FileVideoStream(root_dir + video_name).start()
        time.sleep(1.0)
        # start the FPS timer
        fps = FPS().start()

        frame_index = 0
        frame_width = 640
        frame_height = 480

        contour_w_set = np.zeros(fvs.length)
        contour_h_set = np.zeros(fvs.length)
        ratio_set = np.zeros(fvs.length)
        angle_set = np.zeros(fvs.length)

        centroid_h_set = np.zeros(fvs.length)
        centroid_h_change_set = np.zeros(fvs.length)
        norm_h_set = np.zeros(fvs.length)
        norm_w_set = np.zeros(fvs.length)

        down_velocity_set = np.zeros(fvs.length)

        with mp_pose.Pose(
                model_complexity=2,
                min_detection_confidence=0.75,
                min_tracking_confidence=0.50) as pose:
            while fvs.more():
                # grab the frame from the threaded video file stream, resize
                # it, and convert it to grayscale (while still retaining 3
                # channels)
                image = fvs.read()
                image = cv2.resize(image, (640, 480))

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                results = pose.process(image)

                # Draw the pose annotation on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

                # mp_drawing.plot_landmarks(
                #     results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                if results.pose_landmarks:
                    lm = results.pose_landmarks
                    lm_pose = mp_pose.PoseLandmark

                    # print(lm)

                    # Contour Analysis
                    x_set = np.zeros(32)
                    y_set = np.zeros(32)
                    z_set = np.zeros(32)

                    for i in range(32):
                        x_set[i] = lm.landmark[i].x
                        y_set[i] = lm.landmark[i].y
                        z_set[i] = lm.landmark[i].z

                    contour_width = int((np.max(x_set[x_set != 0]) - np.min(x_set[x_set != 0])) * 640)
                    contour_height = int((np.max(y_set[y_set != 0]) - np.min(y_set[y_set != 0])) * 480)

                    contour_w_set[frame_index] = round(contour_width, 3)
                    contour_h_set[frame_index] = round(contour_height, 3)
                    ratio_set[frame_index] = round(contour_width / contour_height, 3)
                    centroid_h_set[frame_index] = round((y_set[29] + y_set[30] - y_set[23] - y_set[24]) * 0.5 * 480, 3)

                    norm_w_set[frame_index] = round(
                        contour_w_set[frame_index] / np.max(contour_w_set[contour_w_set != 0]), 3)
                    norm_h_set[frame_index] = round(
                        contour_h_set[frame_index] / np.max(contour_h_set[contour_h_set != 0]), 3)

                    # centroid_h_change_set[frame_index] = centroid_h_set[frame_index] - np.max(
                    #     centroid_h_set[centroid_h_set != 0])

                    if frame_index >= 6:
                        if contour_h_set[frame_index] == 0 or contour_h_set[frame_index - 6] == 0:
                            down_velocity_set[frame_index] = 0
                        else:
                            down_velocity_set[frame_index] = round(
                                (contour_h_set[frame_index - 6] - contour_h_set[frame_index]) * 0.5 / 0.1, 3)

                    shoulder_x, shoulder_y = np.abs(lm.landmark[11].x - lm.landmark[12].x), np.abs(
                        lm.landmark[11].y - lm.landmark[12].y)
                    angle_set[frame_index] = round(np.abs((np.arctan(shoulder_y / shoulder_x) / np.pi * 180)), 3)

                frame_index = frame_index + 1

                if cv2.waitKey(1) == 27:
                    break

                fps.update()
            # FALL or ADL ?
            target_path = r'../csv_0304/my_test.csv'
            print(target_path)
            with open(target_path, mode='a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                for num in range(fvs.length):
                    if ratio_set[num] == 0:
                        continue
                    else:
                        writer.writerow([video_name, num + 1, centroid_h_set[num],
                                         contour_h_set[num], norm_h_set[num],
                                         contour_w_set[num], norm_w_set[num],
                                         ratio_set[num],
                                         angle_set[num],
                                         down_velocity_set[num]])

        # stop the timer and display FPS information
        fps.stop()

        cv2.destroyAllWindows()
        fvs.stop()
