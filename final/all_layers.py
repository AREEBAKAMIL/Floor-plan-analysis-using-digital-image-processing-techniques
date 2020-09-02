import cv2
import numpy as np
from matplotlib import pyplot as plt
from bearing_walls import *
import random
from colored_space import *
from external_wall_contours import *
from colored_space import *
from textual_layer import *

#read the image
img_path = "/Users/areebakamil/PycharmProjects/floorPlanAnalysis/Data/floorplans_VOA_cropped/DGroundFloor_cropped.png"
img_path = "/Users/areebakamil/PycharmProjects/floorPlanAnalysis/Data/floorplans_VOA_cropped/Ground Floor Stafford-cropped.png"
# img_path = "/Users/areebakamil/PycharmProjects/floorPlanAnalysis2/Data/other/floorplan12.jpg"

original_image = cv2.imread(img_path)

#get different layer
bearing_walls = get_bearing_walls(original_image)
_, bearing_walls_inv = cv2.threshold(bearing_walls, 200, 255, cv2.THRESH_BINARY_INV)

external_walls_only_skeleton = get_external_bearing_walls(original_image)
_, external_walls_only_skeleton_inv = cv2.threshold(external_walls_only_skeleton, 200, 255, cv2.THRESH_BINARY_INV)

completed_external_walls = get_completed_external_bearing_walls(original_image)
_, completed_external_walls_inv = cv2.threshold(completed_external_walls, 200, 255, cv2.THRESH_BINARY_INV)

icons_layers = get_component(original_image)

# display all the layers
plt.figure()
plt.imshow(original_image, cmap='gray')
plt.title('original_image')
plt.xticks([]), plt.yticks([])
plt.show()

plt.figure()
plt.imshow(bearing_walls_inv, cmap='gray')
plt.title('bearing_walls_inv')
plt.xticks([]), plt.yticks([])
plt.show()

plt.figure()
plt.imshow(external_walls_only_skeleton_inv, cmap='gray')
plt.title('external_walls_only_skeleton_inv')
plt.xticks([]), plt.yticks([])
plt.show()

plt.figure()
plt.imshow(icons_layers, cmap='gray')
plt.title('icons_layers')
plt.xticks([]), plt.yticks([])
plt.show()

plt.figure()
plt.imshow(completed_external_walls_inv, cmap='gray')
plt.title('completed_external_walls_inv')
plt.xticks([]), plt.yticks([])
plt.show()

#get image contours
sorted_contours_external_walls = get_contours(original_image)

# sort out the contours
sorted_contours = sorted(sorted_contours_external_walls, key=cv2.contourArea, reverse=True)

img_copy = original_image.copy()
a, b, c = original_image.shape
blank_img = np.zeros([a, b, c], np.uint8)
blank_img.fill(255)

all_contours_blank_img = blank_img.copy()
exterior_contours = blank_img.copy()
interior_contours = blank_img.copy()
room_bounary_contours = blank_img.copy()


# draw all contours on the blank image
for c in sorted_contours:
    cv2.drawContours(all_contours_blank_img, [c], -1,
                     (random.randrange(0, 255), random.randrange(0, 256), random.randrange(0, 255)), 2)

#draw all contours on the actual image
for c in sorted_contours:
    cv2.drawContours(img_copy, [c], -1,
                     (random.randrange(0, 255), random.randrange(0, 256), random.randrange(0, 255)), 2)

# draw the contours of the external walls and use this to calculate the RCA
cv2.drawContours(exterior_contours, [sorted_contours[0]], -1, (255, 0, 0), 2)
area_px_sq = cv2.contourArea(sorted_contours[0])
print(area_px_sq)
print(area_px_sq * 0.0007000434)

#get contours of external walls
contours, hierachy = cv2.findContours(completed_external_walls, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# draw contours of completed external walls and use this to calulcate the EFA
for c in contours:
    cv2.drawContours(interior_contours, [c], -1,
                     (random.randrange(0, 255), random.randrange(0, 256), random.randrange(0, 255)), 2)


rooms1, colored_rooms = find_rooms(bearing_walls.copy())
colored_rooms1 = colored_rooms.copy()
gray2 = cv2.cvtColor(colored_rooms1, cv2.COLOR_BGR2GRAY)
_, thresh2 = cv2.threshold(gray2, 0, 255, cv2.THRESH_BINARY_INV)

# draw contours over colored rooms
contours2, hierachy2 = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for c in contours2:
    cv2.drawContours(room_bounary_contours, [c], -1,
                     (random.randrange(0, 200), random.randrange(0, 200), random.randrange(0, 200))
                     , 2)


plt.figure()
plt.imshow(interior_contours, cmap='gray')
plt.title('interior_contours')
plt.xticks([]), plt.yticks([])
plt.show()

plt.figure()
plt.imshow(all_contours_blank_img, cmap='gray')
plt.title('all_contours_blank_img')
plt.xticks([]), plt.yticks([])
plt.show()

plt.figure()
plt.imshow(img_copy, cmap='gray')
plt.title('img_copy')
plt.xticks([]), plt.yticks([])
plt.show()

plt.figure()
plt.imshow(exterior_contours, cmap='gray')
plt.title('exterior_contours')
plt.xticks([]), plt.yticks([])
plt.show()

plt.figure()
plt.imshow(room_bounary_contours, cmap='gray')
plt.title('room_bounary_contours')
plt.xticks([]), plt.yticks([])
plt.show()

dst = cv2.addWeighted(room_bounary_contours,0.5,exterior_contours,0.5,0)
dst2 = cv2.addWeighted(dst,0.5,interior_contours,0.5,0)

plt.figure()
plt.imshow(dst, cmap='gray')
plt.title('dst')
plt.xticks([]), plt.yticks([])
plt.show()

plt.figure()
plt.imshow(dst2, cmap='gray')
plt.title('dst2')
plt.xticks([]), plt.yticks([])
plt.show()

# # get textual layer
# text_list, orig_image, blank_img, blank_img2 = get_text_layer(original_image)
#
# plt.figure()
# plt.imshow(orig_image, cmap='gray')
# plt.title('orig_image')
# plt.xticks([]), plt.yticks([])
# plt.show()
#
# plt.figure()
# plt.imshow(blank_img, cmap='gray')
# plt.title('blank_img')
# plt.xticks([]), plt.yticks([])
# plt.show()
#
#
# plt.figure()
# plt.imshow(blank_img2, cmap='gray')
# plt.title('blank_img2')
# plt.xticks([]), plt.yticks([])
# plt.show()
#
