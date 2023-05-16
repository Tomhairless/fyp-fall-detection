import os

import cv2

error_list = []


def main():
    for index in range(31, 41):
        data_path = r'D:\FYP\dataset\UR' + r'\adl-' + str(index) + r'-cam0-rgb'
        fps = 30
        size = (640, 480)

        amount = len([lists for lists in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, lists))])
        print(amount)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(r'D:\FYP\fall_detection\new_dataset' + r'\adl-' + str(index) + r'.avi', fourcc, fps,
                                size)

        if amount >= 100:
            for m in range(1, 10):
                image_path = data_path + r'\adl-' + str(index) + r'-cam0-rgb-00' + str(m) + '.png'
                print(image_path)
                img = cv2.imread(image_path)
                video.write(img)

            for m in range(10, 100):
                image_path = data_path + r'\adl-' + str(index) + r'-cam0-rgb-0' + str(m) + '.png'
                print(image_path)
                img = cv2.imread(image_path)
                video.write(img)

            for m in range(100, amount + 1):
                image_path = data_path + r'\adl-' + str(index) + r'-cam0-rgb-' + str(m) + '.png'
                print(image_path)
                img = cv2.imread(image_path)
                video.write(img)

        if amount < 100:
            for m in range(1, 10):
                image_path = data_path + r'\adl-' + str(index) + r'-cam0-rgb-00' + str(m) + '.png'
                print(image_path)
                img = cv2.imread(image_path)
                video.write(img)

            for m in range(10, amount):
                image_path = data_path + r'\adl-' + str(index) + r'-cam0-rgb-0' + str(m) + '.png'
                print(image_path)
                img = cv2.imread(image_path)
                video.write(img)

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
