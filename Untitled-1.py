import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import imutils
from matplotlib.colors import hsv_to_rgb
import helper
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors

# open image (cv2 opens it in BGR)
img = cv2.imread('./assets/coq2.jpg')
img = imutils.resize(img, width=1200)

# convert to hsv
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# hsv mask values
min_hue, min_saturation, min_value = 0, 0, 0
max_hue, max_saturation, max_value = 0, 255, 255

#display the initial masks
mask = cv2.inRange(hsv, (min_hue, min_saturation, min_value), (max_hue, max_saturation, max_value))
mask = cv2.erode(mask, None, iterations=2)
mask = cv2.dilate(mask, None, iterations=2)
mask = cv2.bitwise_not(mask)

print('Mask shape: ' + str(mask.shape))

gray = mask

# treshhold the brighness
tresh = 100
ret, tresh_img = cv2.threshold(gray, tresh, 255, cv2.THRESH_BINARY)

# cv2.imshow('binary image', tresh_img)
# cv2.waitKey(0)

# grab contours
contours, _ = cv2.findContours(tresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# create an empty image for the contours
contours_img = np.zeros(img.shape)

# draw the contours on the empty image
cv2.drawContours(image=contours_img, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=3, lineType=cv2.LINE_AA)

cv2.imshow('contours', contours_img)
cv2.imshow('Mask', mask)
cv2.waitKey(0)