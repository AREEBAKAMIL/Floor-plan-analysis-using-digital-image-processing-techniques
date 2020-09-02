import cv2
import random
import numpy as np
from matplotlib import pyplot as plt
import sys


def get_contours(original_image):
    img_copy = original_image.copy()

    a, b, c = original_image.shape
    blank_img = np.zeros([a, b, c], np.uint8)
    blank_img1 = np.zeros([a, b, c], np.uint8)

    blank_img.fill(255)
    blank_img1.fill(255)

    original_img_gray = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)

    thresh = cv2.adaptiveThreshold(original_img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    ret, thresh = cv2.threshold(original_img_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    _, thresh = cv2.threshold(original_img_gray, 200, 255, cv2.THRESH_BINARY_INV)

    kernel1 = np.ones((5, 5), np.uint8)  # square image kernel used for erosion
    dilate_img1 = cv2.dilate(thresh, kernel1, iterations=1)
    erode_img1 = cv2.erode(dilate_img1, kernel1, iterations=1)
    kernel2 = np.ones((5, 5), np.uint8)
    dilate_img2 = cv2.dilate(erode_img1, kernel2, iterations=2)
    erode_img2 = cv2.erode(dilate_img2, kernel2, iterations=2)

    contours, hierachy = cv2.findContours(erode_img2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # sort out the contours
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for c in sorted_contours:
        cv2.drawContours(img_copy, [c], -1,
                         (random.randrange(0, 255), random.randrange(0, 256), random.randrange(0, 255))
                         , 2)

    for c in sorted_contours:
        cv2.drawContours(blank_img1, [c], -1,
                         (random.randrange(0, 255), random.randrange(0, 256), random.randrange(0, 255))
                         , 2)

    cv2.drawContours(blank_img, [sorted_contours[0]], -1, (255, 0, 0), 2)

    return sorted_contours



# original_image_path = "../Data/floorplans_VOA/DFirstFloor.jpg"
# original_img = cv2.imread(original_image_path)
#
# contoured_img, all_contours, external_contour = get_contours(original_img)
#
# plt.figure()
# plt.imshow(contoured_img, cmap='gray')
# plt.title('contoured_img')
# plt.xticks([]), plt.yticks([])
# plt.show()
#
# plt.figure()
# plt.imshow(all_contours, cmap='gray')
# plt.title('blanall_contoursk_img')
# plt.xticks([]), plt.yticks([])
# plt.show()
#
# plt.figure()
# plt.imshow(external_contour, cmap='gray')
# plt.title('external_contour')
# plt.xticks([]), plt.yticks([])
# plt.show()

# cv2.imshow("external_contour",external_contour)
# cv2.waitKey(0)
# cv2.destroyAllWindows()