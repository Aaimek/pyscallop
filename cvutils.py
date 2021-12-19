import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import imutils
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors

# HSV image as input
def plot_hsv_3d(hsv):
    h, s, v = cv2.split(hsv)
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection="3d")

    pixel_colors = hsv.reshape((np.shape(hsv)[0]*np.shape(hsv)[1], 3))
    norm = colors.Normalize(vmin=-1.,vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()

    axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
    axis.set_xlabel("Hue")
    axis.set_ylabel("Saturation")
    axis.set_zlabel("Value")
    plt.show()

# Draw contours on a black baground 
# Grayscale image as input
def draw_contours(img):
    img = imutils.resize(img, width=1200)

    # Convert to grayscale
    gray = img
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # cv2.imshow('grayscale image', gray)
    # cv2.waitKey(0)

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
    # cv2.namedWindow("contours")
    cv2.imshow('contours', contours_img)
    cv2.waitKey(0)




# image_path = './assets/gray.jpg'
# img = cv2.imread(image_path)
# img = imutils.resize(img, width=1200)

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# print(gray.dtype)
# draw_contours(img)

# cv2.waitKey(0)