import time

import cv2
import mediapipe as mp
import numpy as np
import scipy
from matplotlib import pyplot as plt

from utils.fps import FPS
from utils.thread_utils import FileVideoStream

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

file_path = r'../dataset/fall/fall-19.avi'
fvs = FileVideoStream(file_path).start()
time.sleep(1.0)
# start the FPS timer
fps = FPS().start()

# cap = cv2.VideoCapture(root_dir + video_name)

frame_index = 0

frame_width = 640
frame_height = 480

ratio_set = np.zeros(fvs.length)

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

            print(lm)

            # Contour Analysis
            x_set = np.zeros(32)
            y_set = np.zeros(32)
            for i in range(32):
                if (i > 10) and (i < 23):
                    x_set[i] = None
                    y_set[i] = None
                else:
                    x_set[i] = lm.landmark[i].x
                    y_set[i] = lm.landmark[i].y

            contour_width = int((max(x_set) - min(x_set)) * 640)
            contour_height = int((max(y_set) - min(y_set)) * 480)

            print(contour_width, contour_height)
            print(contour_width / contour_height)
            print(frame_index)
            ratio_set[frame_index] = round(contour_width / contour_height, 3)
            print(ratio_set)
            # Gait Analysis
            # left_foot = np.array([int(lm.landmark[31].x * 640), int((lm.landmark[31].y * 360))])
            # right_foot = np.array([int(lm.landmark[32].x * 640), int((lm.landmark[32].y * 360))])
            # connection = left_foot - right_foot
            # print(connection)
            # distance = np.linalg.norm(left_foot - right_foot)
            # print("distance: " + str(distance))

        else:
            ratio_set[frame_index] = None

        cv2.putText(image, str(frame_index), (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
        cv2.imshow(file_path, image)

        frame_index = frame_index + 1

        if cv2.waitKey(1) == 27:
            break

        fps.update()

ratio_set = scipy.signal.savgol_filter(ratio_set, 5, 3)
fig, ax = plt.subplots()  # Create a figure containing a single axes.
ax.scatter(np.arange(fvs.length), ratio_set, s=2)  # Plot some data on the axes.

plt.show()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
print("[INFO] approx. num of frames:" + str(fps._numFrames))

cv2.destroyAllWindows()
fvs.stop()
