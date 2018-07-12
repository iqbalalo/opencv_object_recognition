import cv2
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib import pyplot as plt
import time
import imutils


def load_image_and_save_grayscale(read_file, file_to_save):
    img = cv2.imread(read_file, 0)
    img2 = cv2.imwrite(file_to_save, img)
    return img2


def opencv_show_image(title, file):
    img = cv2.imread(file)
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def opencv_image_shown_by_matplot(file):
    bgr_img = cv2.imread(file)
    b, g, r = cv2.split(bgr_img)
    rgb_img = cv2.merge([r, g, b])
    plt.subplot(121)
    plt.imshow(bgr_img)
    plt.xticks([]), plt.yticks([])
    plt.subplot(122, facecolor='b')
    plt.imshow(rgb_img)
    plt.xticks([]), plt.yticks([])
    plt.show()


def video_capture():
    cap = cv2.VideoCapture(0)

    if cap:
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()

            # Our operations on the frame come here
            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = imutils.resize(frame, width=640)

            # Display the resulting frame
            cv2.imshow('Camera 1', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Video capture error..")


def end_video_capture():
    cap = cv2.VideoCapture(0)
    cap.release()

video_capture()