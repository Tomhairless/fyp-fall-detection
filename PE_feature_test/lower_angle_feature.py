import time

import cv2
import mediapipe as mp
import numpy as np
import scipy as scipy
from matplotlib import pyplot as plt

from utils.fps import FPS
from utils.thread_utils import FileVideoStream

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

file_path = r'../dataset/fall/fall-11.avi'
fvs = FileVideoStream(file_path).start()
time.sleep(1.0)
# start the FPS timer
fps = FPS().start()

# cap = cv2.VideoCapture(root_dir + video_name)

frame_index = 0

frame_width = 640
frame_height = 480

angle_set = np.zeros(fvs.length)

with mp_pose.Pose(
        model_complexity=1,
        min_detection_confidence=0.85,
        min_tracking_confidence=0.15) as pose:
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

            # Shoulders
            average_sd_y = (lm.landmark[11].y + lm.landmark[12].y) / 2
            average_sd_z = (lm.landmark[11].z + lm.landmark[12].z) / 2

            # Ankle
            average_ak_y = (lm.landmark[27].y + lm.landmark[28].y) / 2
            average_ak_z = (lm.landmark[27].z + lm.landmark[28].z) / 2

            angle_y = average_ak_y - average_sd_y
            angle_z = average_ak_z - average_sd_z

            print(angle_y, angle_z)

            angle = np.abs(np.arctan(angle_z / angle_y) / np.pi * 180)
            print(angle)

            print(frame_index)
            angle_set[frame_index] = round(angle, 3)
            print(angle_set)

        else:
            print("?")

        cv2.putText(image, str(frame_index), (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
        cv2.imshow(file_path, image)

        frame_index = frame_index + 1

        if cv2.waitKey(1) == 27:
            break

        fps.update()


angle_set = scipy.signal.savgol_filter(angle_set, 5, 3)
print(len(angle_set))
fig, ax = plt.subplots()  # Create a figure containing a single axes.
ax.scatter(np.arange(fvs.length), angle_set, s=3)  # Plot some data on the axes.

plt.show()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
print("[INFO] approx. num of frames:" + str(fps._numFrames))

cv2.destroyAllWindows()
fvs.stop()
