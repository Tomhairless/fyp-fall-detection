import csv
import os
import time

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from joblib import load

from utils.fps import FPS
from utils.thread_utils import FileVideoStream

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

if __name__ == "__main__":
    is_export = False

    rf = load('model/rf_0325.joblib')
    print(rf)

    # FALL or ADL ?
    root_dir = r'dataset/adl/'
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

        fall_frame = []

        contour_w_set = np.zeros(fvs.length)
        contour_h_set = np.zeros(fvs.length)
        ratio_set = np.zeros(fvs.length)
        angle_set = np.zeros(fvs.length)

        centroid_h_set = np.zeros(fvs.length)
        centroid_h_change_set = np.zeros(fvs.length)
        norm_h_set = np.zeros(fvs.length)

        down_velocity_set = np.zeros(fvs.length)
        foot_x_set = np.zeros(fvs.length)
        foot_y_set = np.zeros(fvs.length)
        gait_velocity_set = np.zeros(fvs.length)

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
                    x_set = np.zeros(33)
                    y_set = np.zeros(33)
                    z_set = np.zeros(33)

                    for i in range(33):
                        x_set[i] = lm.landmark[i].x
                        y_set[i] = lm.landmark[i].y
                        z_set[i] = lm.landmark[i].z

                    contour_width = int((np.max(x_set[x_set != 0]) - np.min(x_set[x_set != 0])) * 640)
                    contour_height = int((np.max(y_set[y_set != 0]) - np.min(y_set[y_set != 0])) * 480)

                    cv2.rectangle(image, (int((np.min(x_set[x_set != 0]) * 640)), int(np.min(y_set[y_set != 0]) * 480)),
                                  (int((np.max(x_set[x_set != 0]) * 640)), int(np.max(y_set[y_set != 0]) * 480)),
                                  (127, 255, 0),
                                  2)

                    # print(contour_width, contour_height)
                    # print(contour_width / contour_height)
                    # print(frame_index)

                    contour_w_set[frame_index] = round(contour_width, 3)
                    contour_h_set[frame_index] = round(contour_height, 3)
                    ratio_set[frame_index] = round(contour_width / contour_height, 3)

                    centroid_h_set[frame_index] = round((y_set[29] + y_set[30] - y_set[23] - y_set[24]) * 0.5 * 480, 3)

                    foot_x_set[frame_index] = round(x_set[31] + x_set[32], 3) * 0.5 * 640
                    foot_y_set[frame_index] = round(y_set[31] + y_set[32], 3) * 0.5 * 480

                    if frame_index >= 6:
                        if contour_h_set[frame_index] == 0 or contour_h_set[frame_index - 6] == 0:
                            down_velocity_set[frame_index] = 0
                            # gait_velocity_set[frame_index] = 0
                        else:
                            down_velocity_set[frame_index] = round(
                                (contour_h_set[frame_index - 6] - contour_h_set[frame_index]) * 0.5 / 0.2, 3)
                            # point_1 = np.array([foot_x_set[frame_index], foot_y_set[frame_index]])
                            # point_2 = np.array([foot_x_set[frame_index - 6], foot_y_set[frame_index - 6]])
                            # gait_velocity_set[frame_index] = round(np.linalg.norm(point_2 - point_1) / 0.2, 3)

                    shoulder_x, shoulder_y = np.abs(lm.landmark[11].x - lm.landmark[12].x), np.abs(
                        lm.landmark[11].y - lm.landmark[12].y)
                    angle_set[frame_index] = round(np.abs((np.arctan(shoulder_y / shoulder_x) / np.pi * 180)), 1)

                    if ratio_set[frame_index] < 0.70 or down_velocity_set[frame_index] < 25:
                        cv2.putText(image, 'STAND', (500, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
                    else:
                        cv2.putText(image, '?', (500, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
                        X = np.array([[centroid_h_set[frame_index], contour_w_set[frame_index], ratio_set[frame_index],
                                       angle_set[frame_index], down_velocity_set[frame_index]]])
                        X_df = pd.DataFrame(X, columns=['centroid_h', 'contour_w', 'ratio', 'angle', 'down_velocity'])
                        y = rf.predict(X_df)
                        if y:
                            cv2.putText(image, 'FALL', (100, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                            fall_frame.append(frame_index)

                cv2.putText(image, str(frame_index), (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
                cv2.imshow(video_name, image)

                frame_index = frame_index + 1

                if cv2.waitKey(1) == 27:
                    break

                fps.update()

            # FALL or ADL ?
            target_path = r'csv_0304/results_0326.csv'
            print(target_path)
            with open(target_path, mode='a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                for value in fall_frame:
                    writer.writerow([video_name, value])

        # stop the timer and display FPS information
        fps.stop()

        cv2.destroyAllWindows()
        fvs.stop()
