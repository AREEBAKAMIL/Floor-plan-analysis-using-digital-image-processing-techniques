import cv2
import random
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import numpy as numpy
from skimage import io
from skimage import morphology


def get_bearing_walls(origianl_image):
    gray = cv2.cvtColor(origianl_image, cv2.COLOR_RGB2GRAY)

    # binary inverse thresholding
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((3, 3), np.float32)
    bearing_walls = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # _, bearing_walls_inv = cv2.threshold(bearing_walls, 127, 255, cv2.THRESH_BINARY_INV)

    return bearing_walls


def get_external_bearing_walls(img):

    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    ret, th = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    kernel1 = np.ones((5, 5), np.uint8)  # square image kernel used for erosion
    erode1 = cv2.erode(th, kernel1, iterations=2)
    # cv2.imshow("erode1", erode1)

    kernel1 = np.ones((5, 5), np.uint8)  # square image kernel used for erosion
    dilate1 = cv2.dilate(erode1, kernel1, iterations=2)
    # cv2.imshow("dilate1", dilate1)

    return dilate1


def get_completed_external_bearing_walls(input_img):

    image = get_external_bearing_walls(input_img)

    # fill horizontal gaps
    selem_horizontal = morphology.rectangle(1, 50)
    img_filtered = morphology.closing(image, selem_horizontal)

    # fill vertical gaps
    selem_vertical = morphology.rectangle(80, 1)
    img_filtered = morphology.closing(img_filtered, selem_vertical)

    # plt.imshow(img_filtered, cmap="gray")
    # plt.gca().axis("off")
    # plt.show()

    return img_filtered

def get_component(original_img):

    bearing_walls = get_bearing_walls(original_img)
    _, bearing_walls_inv = cv2.threshold(bearing_walls, 200, 255, cv2.THRESH_BINARY_INV)


    gray = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)

    _, original_img_thresh2 = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    kernel1 = np.ones((1, 1), dtype=np.uint8)
    dilated_bearing_walls = cv2.dilate(bearing_walls_inv, kernel1)
    graphical_img2 = original_img_thresh2 - bearing_walls_inv

    ret, thresh1 = cv2.threshold(graphical_img2, 0, 255, cv2.THRESH_BINARY_INV)

    kernel2 = np.zeros((5, 5), dtype=np.uint8)
    dilated_graphics_layer = cv2.dilate(thresh1, kernel2)

    return dilated_graphics_layer