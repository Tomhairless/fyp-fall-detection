import time

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from joblib import load
from matplotlib import pyplot as plt

from utils.fps import FPS
from utils.thread_utils import FileVideoStream

rf = load('model/rf_0325.joblib')
print(rf)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

mp_pose = mp.solutions.pose

bg_subtractor = cv2.createBackgroundSubtractorKNN(detectShadows=True)
bg_subtractor.setHistory(50)
bg_subtractor.setDist2Threshold(1500)

erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 5))

file_path = 'dataset/adl/adl-06.avi'  # adl-10, 21, 23
fvs = FileVideoStream(file_path).start()
time.sleep(1.0)
# start the FPS timer
fps = FPS().start()

frame_index = 0

frame_width = 640
frame_height = 480

first_stage_set = np.zeros(fvs.length)

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

flag = False

with mp_pose.Pose(
        model_complexity=1,
        min_detection_confidence=0.75,
        min_tracking_confidence=0.50) as pose:
    while fvs.more():
        # grab the frame from the threaded video file stream, resize
        # it, and convert it to grayscale (while still retaining 3
        # channels)
        if frame_index == 0:
            start_time = time.time()

        image = fvs.read()

        # while not flag:
        #     bgs_frame = cv2.resize(image, (240, 180))
        #     fg_mask = bg_subtractor.apply(bgs_frame, 0.005)
        #
        #     ret, thresh = cv2.threshold(fg_mask, 244, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        #
        #     morph_dilate = cv2.dilate(thresh, dilate_kernel, iterations=3)
        #
        #     morph_erode = cv2.erode(morph_dilate, erode_kernel, iterations=1)
        #
        #     morph = cv2.GaussianBlur(morph_erode, (5, 5), 0)
        #
        #     result_after_process = morph
        #
        #     contours, hierarchy = cv2.findContours(result_after_process, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #
        #     if contours:
        #         roi_contour = max(contours, key=cv2.contourArea)
        #         x, y, w, h = cv2.boundingRect(roi_contour)
        #
        #         if cv2.contourArea(roi_contour) > 100:
        #             flag = True
        #             print("!")
        #
        # while flag:

        pe_frame = cv2.resize(image, (640, 480))

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        pe_frame.flags.writeable = False
        pe_frame = cv2.cvtColor(pe_frame, cv2.COLOR_BGR2RGB)

        results = pose.process(pe_frame)

        # Draw the pose annotation on the image.
        pe_frame.flags.writeable = True
        pe_frame = cv2.cvtColor(pe_frame, cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        # mp_drawing.plot_landmarks(
        #     results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        if results.pose_landmarks:
            lm = results.pose_landmarks

            # Contour Analysis`
            x_set = np.zeros(33)
            y_set = np.zeros(33)
            z_set = np.zeros(33)

            for i in range(33):
                x_set[i] = lm.landmark[i].x
                y_set[i] = lm.landmark[i].y
                z_set[i] = lm.landmark[i].z

            contour_width = int((np.max(x_set[x_set != 0]) - np.min(x_set[x_set != 0])) * 640)
            contour_height = int((np.max(y_set[y_set != 0]) - np.min(y_set[y_set != 0])) * 480)

            if frame_index < 340:
                cv2.rectangle(pe_frame,
                              (int((np.min(x_set[x_set != 0]) * 640)), int(np.min(y_set[y_set != 0]) * 480)),
                              (int((np.max(x_set[x_set != 0]) * 640)), int(np.max(y_set[y_set != 0]) * 480)),
                              (0, 255, 0),
                              2)
                # cv2.putText(image, 'FALL', (500, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            else:
                cv2.rectangle(pe_frame,
                              (int((np.min(x_set[x_set != 0]) * 640)), int(np.min(y_set[y_set != 0]) * 480)),
                              (int((np.max(x_set[x_set != 0]) * 640)), int(np.max(y_set[y_set != 0]) * 480)),
                              (0, 0, 255),
                              2)
                cv2.putText(pe_frame, 'FALL', (500, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

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

            if ratio_set[frame_index] < 0.75:
                # cv2.putText(image, 'STAND', (500, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
                first_stage_set[frame_index] = 3
            else:
                # cv2.putText(image, '?', (500, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
                first_stage_set[frame_index] = 10
                X = np.array([[centroid_h_set[frame_index], contour_w_set[frame_index], ratio_set[frame_index],
                               angle_set[frame_index], down_velocity_set[frame_index]]])
                X_df = pd.DataFrame(X, columns=['centroid_h', 'contour_w', 'ratio', 'angle', 'down_velocity'])
                y = rf.predict(X_df)
                if y:
                    # cv2.putText(image, 'FALL', (100, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    print(frame_index)

        cv2.putText(pe_frame, str(frame_index), (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
        cv2.imshow(file_path, pe_frame)

        frame_index = frame_index + 1

        if cv2.waitKey(1) == 27:
            break

        fps.update()

# width_set = scipy.signal.savgol_filter(width_set, 3, 2)
# height_set = scipy.signal.savgol_filter(height_set, 3, 2)
# ratio_set = scipy.signal.savgol_filter(ratio_set, 3, 2)

# height_change_set = np.zeros(fvs.length)
# std_sum = 0
# print('---------')
# for num in range(3, fvs.length):
#     if ratio_set[num] == None:
#         height_change_set[num] == None
#     else:
#         if height_set[num] == None or height_set[num - 3] == None:
#             dec_rate = 0
#         else:
#             dec_rate = (height_set[num - 3] - height_set[num]) / 0.1
#             print("------")
#             print(dec_rate)
#             print("------")
#             height_change_set[num] = round(dec_rate, 3)

# acceler_set = np.full(fvs.length, None)
# for num in range(1, fvs.length):
#     if (centroid_h_change_set[num]) == None or centroid_h_change_set[num - 1] == None:
#         acceler_set[num] = None
#     else:
#         acceler_set[num] = (centroid_h_change_set[num - 1] - centroid_h_change_set[num]) * 10

# plt.subplot(4, 1, 4)


# std_set = scipy.signal.savgol_filter(std_set, 3, 2)

# plt.subplot(4, 1, 1)
# plt.scatter(np.arange(fvs.length), ratio_set, s=2)
# plt.title('RATIO')
#
# plt.subplot(4, 1, 2)
# plt.scatter(np.arange(fvs.length), centroid_h_set, s=2)
# plt.title('HEIGHT')
#
# plt.subplot(4, 1, 3)
# plt.scatter(np.arange(fvs.length), centroid_h_change_set, s=2)
# plt.title('HEIGHT CHANGE')
#
# print(ratio_set)

# print(first_stage_set)
# plt.scatter(np.arange(fvs.length), first_stage_set, c='r', s=2)

plt.show()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
print("[INFO] approx. num of frames:" + str(fps._numFrames))

end_time = time.time()
elapsed_time = end_time - start_time
print(elapsed_time)

cv2.destroyAllWindows()