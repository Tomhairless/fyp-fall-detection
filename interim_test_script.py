import os
import time
from pathlib import Path

import cv2
import numpy as np
from matplotlib import pyplot as plt

from utils.fps import FPS
from utils.thread_utils import FileVideoStream

figure, ax = plt.subplots(figsize=(10, 8))
ax.set_xlabel('frame')
ax.set_ylabel('height')
x_value = []
y_value = []

error_log = []
score_log = []

if __name__ == "__main__":
    is_export = False
    root_dir = "dataset/fall/"
    video_list = os.listdir(root_dir)
    for video_name in video_list:
        video_path = root_dir + video_name
        export_path = 'export/'
        print("[INFO] starting video file thread...")

        fvs = FileVideoStream(root_dir + video_name).start()
        time.sleep(1.0)
        # start the FPS timer
        fps = FPS().start()

        # cap = cv2.VideoCapture(root_dir + video_name)

        # Create new directory for the export of video frame after processing
        if is_export:
            print("Export: ON")
            p = Path('D:\\FYP\\opencv_demo\\export\\' + video_name)
            try:
                p.mkdir()
            except FileExistsError as e:
                print(e)
        else:
            print("Export: OFF")

        frame_width = 240
        frame_height = 160

        # erode_kernel = np.ones((3, 3), np.uint8)
        # dilate_kernel = np.ones((3, 3), np.uint8)
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 5))

        frame_num = 0
        prev_x, prev_y, prev_w, prev_h = None, None, None, None
        prev_hw_ratio = None
        history = []
        is_potential_fall = True
        score = 0
        potential_score = 0

        bg_subtractor = cv2.createBackgroundSubtractorKNN(detectShadows=True)
        bg_subtractor.setHistory(50)
        bg_subtractor.setDist2Threshold(1500)

        # loop over frames from the video file stream
        while fvs.more():
            # grab the frame from the threaded video file stream, resize
            # it, and convert it to grayscale (while still retaining 3
            # channels)
            frame = fvs.read()
            frame = cv2.resize(frame, (frame_width, frame_height))
            cv2.imshow(video_name, frame)
            [r, g, b] = frame[100, 100]

            is_augment = False
            if int(r) + int(g) + int(b) < 96:
                print(r, g, b)
                frame = cv2.medianBlur(frame, 3)
                array_alpha = np.array([1.50])
                array_beta = np.array([50.0])
                # add a beta value to every pixel
                cv2.add(frame, array_beta, frame)
                # multiply every pixel value by alpha
                cv2.multiply(frame, array_alpha, frame)
                new_frame = frame
                cv2.imshow('augmentation', new_frame)
                # Apply KNN background subtractor
                fg_mask = bg_subtractor.apply(new_frame, 0.005)
                is_augment = True
            else:
                fg_mask = bg_subtractor.apply(frame, 0.005)

            cv2.imshow('foreground', fg_mask)

            ret, thresh = cv2.threshold(fg_mask, 244, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            cv2.imshow('threshold', thresh)

            if is_augment:
                morph_dilate = cv2.dilate(thresh, dilate_kernel, iterations=2)
                # cv2.imshow('morph_dilate', morph_dilate)

                morph_erode = cv2.erode(morph_dilate, erode_kernel, iterations=2)
                # cv2.imshow('morph_erode', morph_erode)
            else:
                morph_dilate = cv2.dilate(thresh, dilate_kernel, iterations=3)
                # cv2.imshow('morph_dilate', morph_dilate)

                morph_erode = cv2.erode(morph_dilate, erode_kernel, iterations=1)
                # cv2.imshow('morph_erode', morph_erode)

            morph = cv2.GaussianBlur(morph_erode, (5, 5), 0)

            result_after_process = morph
            cv2.imshow('result', result_after_process)

            is_wrong_contour = False
            is_track_history = False
            if frame_num >= 10:
                # Find the contours of the targeted person
                contours, hierarchy = cv2.findContours(result_after_process, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if contours:
                    roi_contour = max(contours, key=cv2.contourArea)

                    x, y, w, h = cv2.boundingRect(roi_contour)

                    rect = cv2.minAreaRect(roi_contour)
                    # box = cv2.boxPoints(rect)
                    # box = np.int0(box)
                    # cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)
                    print(rect)
                    # print(box)

                    hw_ratio = h / w  # Obtain the ratio of the width and height
                    print(frame_num, "h：" + str(h), "w：" + str(w), hw_ratio)

                    # Qualify the contour area
                    if cv2.contourArea(roi_contour) < frame_width / 8 * frame_height / 8 or cv2.contourArea(
                            roi_contour) > frame_width / 3 * frame_height:
                        print("Error AREA")
                        is_wrong_contour = True

                    # Qualify the changing rate
                    if prev_x and prev_y and prev_w and prev_h and prev_hw_ratio is not None:
                        if abs(prev_x - x) > 230 or abs(prev_y - y) > 155:
                            is_wrong_contour = True
                            print("Error XY")
                        if w / prev_w > 1.50 or prev_w / w > 1.50:
                            is_wrong_contour = True
                            print("Error W")
                        if h / prev_h > 3.00 or prev_h / h > 3.00:
                            is_wrong_contour = True
                            print("Error H")
                        if prev_hw_ratio / hw_ratio > 2.25 or hw_ratio / prev_hw_ratio > 2.25:
                            is_wrong_contour = True
                            print("Error WH")

                    if is_wrong_contour:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (153, 0, 153),
                                      2)  # cv2.rectangle(image, start_point, end_point, color, thickness)
                        print("WRONG CONTOUR")
                        cv2.putText(frame, 'WRONG CONTOUR', (x + w + 15, y + h + 15), cv2.FONT_HERSHEY_TRIPLEX, 0.4,
                                    (0, 255, 255),
                                    1)

                        # Replace the wrong contour with previous correct one
                        if prev_h and prev_w is not None:
                            history.append(prev_h)
                            is_capture = True


                    else:

                        x_value.append(frame_num)
                        y_value.append(h)
                        print(history)
                        history.append(h)

                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255),
                                      2)  # cv2.rectangle(image, start_point, end_point, color, thickness)

                        current_key = len(history) - 1
                        if len(history) >= 20:
                            max_h = np.mean(history[current_key - 19:current_key - 10])
                            min_h = np.mean(history[current_key - 9:current_key])
                            print(max_h, min_h)

                            decrease_rate = (max_h - min_h) / max_h
                            print(decrease_rate)

                            if hw_ratio <= 0.80 and decrease_rate >= 0.10:
                                score += 3
                                print("SCORE_1")
                            if decrease_rate >= 0.25:
                                score += 3
                                print("SCORE_2")
                            if decrease_rate >= 0.40:
                                score += 3
                                print("SCORE_3")

                            if score >= 40:
                                print("FALL")
                                cv2.putText(frame, 'FALL', (200, 150), cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 0, 255),
                                            2)

                        # if prev_h is not None:
                        #     if h > prev_h * 1.25:
                        #         print("FALL")
                        #         cv2.putText(frame, 'FALL', (x + 15, y + 15), cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 0, 255), 1)

                    if is_wrong_contour is False:
                        prev_x = x
                        prev_y = y
                        prev_w = w
                        prev_h = h
                        prev_hw_ratio = hw_ratio

            cv2.imshow('with contour', frame)

            # Export the video frame
            if is_export:
                cv2.imwrite('D:\\FYP\\opencv_demo\\export\\' + video_name + "\\frame_" + str(frame_num) + ".png",
                            fg_mask)

            if cv2.waitKey(1) == 27:
                break

            # success, frame = cap.read()
            frame_num += 1
            fps.update()

        # stop the timer and display FPS information
        fps.stop()

        score_log.append(video_name + ": " + str(score))
        if score <= 40:
            error_log.append(video_name)

        print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
        print("[INFO] approx. num of frames:" + str(fps._numFrames))

        ax.plot(x_value, y_value)
        plt.show()

        cv2.destroyAllWindows()
        fvs.stop()

    print(score_log)
    print(error_log)
