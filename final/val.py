import numpy as np
import cv2
from imutils.object_detection import non_max_suppression
import pytesseract
from matplotlib import pyplot as plt
import sys
from spellchecker import SpellChecker
from collections import Counter
import pandas as pd
from text_detector import *

df = pd.read_csv("../final/dataset.csv")

paths = df['path']
rooms = df['rooms']

list_actual_rooms = []
list_detected_rooms = []
total_no_detected_rooms = 0
total_no_actual_rooms = 0

print("DETECTED:")
for path in paths:
    image = cv2.imread(path)
    text = get_text(image)
    list_detected_rooms.append(text)
    print(text)
    total_no_detected_rooms += len(text)

print("ACTUAL:")
for room in rooms:
    temp = room.split(",")
    print(temp)
    list_actual_rooms.append(temp)
    total_no_actual_rooms += len(temp)


print("****")
print(total_no_detected_rooms)
print(total_no_actual_rooms)
flattened_list_detected_text = [y for x in list_detected_rooms for y in x]
print(flattened_list_detected_text)
flattened_list_actual_text = [y for x in list_actual_rooms for y in x]
print(flattened_list_actual_text)

# count the number of rooms detected
print("\nCount of the number of rooms detected:")
final_count = Counter(flattened_list_detected_text)
for key, value in final_count.items():
    print(key + ": ", value)

print("\nCount of the number of actual:")
final_count = Counter(flattened_list_actual_text)
for key, value in final_count.items():
    print(key + ": ", value)








