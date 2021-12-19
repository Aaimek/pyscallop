import cv2 as cv2
import numpy as np
import cvutils
import imutils

image_path = './assets/gray.jpg'
img = cv2.imread(image_path)
img = imutils.resize(img, width=1200)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cvutils.draw_contours(gray)

cv2.waitKey(0)