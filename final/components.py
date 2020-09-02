import cv2
import random
from matplotlib import pyplot as plt
import numpy as np
from bearing_walls import *

def get_component(original_img, bearing_walls_inv):

    gray = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)


    _, original_img_thresh2 = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    kernel1 = np.ones((1, 1), dtype=np.uint8)
    dilated_bearing_walls = cv2.dilate(bearing_walls_inv, kernel1)
    graphical_img2 = original_img_thresh2 - bearing_walls_inv

    ret, thresh1 = cv2.threshold(graphical_img2, 0, 255, cv2.THRESH_BINARY_INV)

    kernel2 = np.zeros((5, 5), dtype=np.uint8)
    dilated_graphics_layer = cv2.dilate(thresh1, kernel2)

    return dilated_graphics_layer



