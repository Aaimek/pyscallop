import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import imutils
from numpy.core.fromnumeric import trace
import cvutils
import pandas as pd

# params
image_path = './assets/coq_mix_v2.jpg'
title_window = 'Window'

# initial values
min_hsv = (0, 0, 0)
max_hsv = (230, 250, 250)

# find the blue ruler and measure its lengh in pixel in order to measure other objects
def calibrate_ruler(hsv_image):
    min_hsv = (0, 0, 0)
    max_hsv = (100, 255, 255)

    # filter the hsv image
    mask = cv2.inRange(hsv_image, min_hsv, max_hsv)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    mask = cv2.bitwise_not(mask)
    gray = mask

    # treshhold the brighness
    tresh = 100
    ret, tresh_img = cv2.threshold(gray, tresh, 255, cv2.THRESH_BINARY)

    # grab contours
    contours, _ = cv2.findContours(tresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # create an empty image for the contours
    contours_img = np.zeros(object_image.shape)

    # draw the contours on the empty image
    cv2.drawContours(image=contours_img, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=3, lineType=cv2.LINE_AA)

    biggest_contour = contours[-1]
    for contour in contours:
        if cv2.contourArea(contour) > cv2.contourArea(biggest_contour):
            biggest_contour = contour
    
    _,_,w,_ = cv2.boundingRect(biggest_contour)

    # w in how much 10cm is in terms of pixels
    return w


# filter the image, grab contours and display them on a separate black image
def treat_image(hsv_image, min_hsv, max_hsv, lenunit):
    # filter the hsv image
    mask = cv2.inRange(hsv_image, min_hsv, max_hsv)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    mask = cv2.bitwise_not(mask)
    gray = mask

    # treshhold the brighness
    tresh = 100
    ret, tresh_img = cv2.threshold(gray, tresh, 255, cv2.THRESH_BINARY)

    # grab contours
    contours, _ = cv2.findContours(tresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # We will make the contours go through a series of selection in order to only output the relevant ones, that most likerely correspond to an object

    # This dataframe will hold all the data for the contours that went through the selection steps
    # contour: contour object
    # circle: ((x, y), center)
    # lengh: lengh of the thing
    # state: NOISE, GOOD or BAD for now
    contours_df = pd.DataFrame(columns=['contour', 'circle', 'lengh', 'state'])

    # populate the dataframe with the contours at least
    contours_df = contours_df.append([{'contour': contours[i], 'circle': None, 'lengh': None, 'state': None} for i in range(len(contours))], ignore_index=True)

    # array for the contours that don't make it to the selection step
    correct_contours = []

    # Eliminate the objects of an insignificant size
    for index, row in contours_df.iterrows():
        contour = row['contour']

        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x),int(y))
        radius = int(radius)

        # drop it from the df if its an insignificant size
        if radius < lenunit/3:
            row['state'] = 'NOISE'
        else:
            lengh = 10*radius*2/lenunit # in cm
            row['circle'] = (center, radius)
            row['lengh'] = lengh # for now the lengh is just the radius

            if lengh >= 10.2:
                row['state'] = 'GOOD'
            else:
                row['state'] = 'BAD'
    
    # save the dataframe 
    contours_df.to_pickle('contours_df.pkl')

    # display the last mask
    cv2.imshow('Mask', mask)

    draw_output(contours_df)

def draw_output(contours_df):
    # create an empty image for the contours
    contours_img = np.zeros(object_image.shape)
    rendr_img = object_image.copy()

    for index, row in contours_df[contours_df['state'] != 'NOISE'].iterrows():
        contour = row['contour']
        center = row['circle'][0]
        radius = row['circle'][1]
        lengh = row['lengh']
        state = row['state']

        #draw the circle
        cv2.circle(contours_img, center, radius, (0,0,255), 2)

        lengh_text = str(round(lengh, 1)) + ' cm'
        cv2.putText(contours_img, lengh_text, center, cv2.FONT_HERSHEY_SIMPLEX, 1, (209, 80, 0, 255), 3)
        cv2.putText(rendr_img, lengh_text, center, cv2.FONT_HERSHEY_SIMPLEX, 1, (209, 80, 0, 255), 3)

        # write a if the coquille is good or not
        x, y = center
        y += 35

        if state == 'GOOD':
            cv2.putText(contours_img, 'GOOD', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0, 255), 4)
            cv2.putText(rendr_img, 'GOOD', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0, 255), 4)
            cv2.drawContours(image=contours_img, contours=contour, contourIdx=-1, color=(0, 255, 0), thickness=3, lineType=cv2.LINE_AA)
        elif state == 'BAD':
            cv2.putText(contours_img, 'BAD', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255, 255), 4)
            cv2.putText(rendr_img, 'BAD', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255, 255), 4)
            cv2.drawContours(image=contours_img, contours=contour, contourIdx=-1, color=(0, 0, 255), thickness=3, lineType=cv2.LINE_AA)
        elif state == 'NOISE':
            pass
            # cv2.drawContours(image=contours_img, contours=contour, contourIdx=-1, color=(255, 255, 255), thickness=3, lineType=cv2.LINE_AA)

    print('Nombre de coquilles valides: ' + str(len(contours_df[contours_df['state'] == 'GOOD'])))
    print('Nombre de coquilles invalides: ' + str(len(contours_df[contours_df['state'] == 'BAD'])))

    cv2.imshow('contours', contours_img)
    cv2.imshow('object image', rendr_img)

def on_trackbar(val):
    min_hue = cv2.getTrackbarPos('Min hue:', title_window)
    max_hue = cv2.getTrackbarPos('Max hue:', title_window)
    min_saturation = cv2.getTrackbarPos('Min saturation:', title_window)
    max_saturation = cv2.getTrackbarPos('Max saturation:', title_window)
    min_value = cv2.getTrackbarPos('Min value:', title_window)
    max_value = cv2.getTrackbarPos('Max value:', title_window)

    treat_image(hsv, (min_hue, min_saturation, min_value), (max_hue, max_saturation, max_value), unitlenpxl)

if __name__ == '__main__':
    # open a window and pop trackbars
    cv2.namedWindow(title_window, cv2.WINDOW_FULLSCREEN)
    cv2.createTrackbar('Min hue:', title_window , 0, 255, on_trackbar)
    cv2.createTrackbar('Max hue:', title_window , 0, 255, on_trackbar)
    cv2.createTrackbar('Min saturation:', title_window , 0, 255, on_trackbar)
    cv2.createTrackbar('Max saturation:', title_window , 0, 255, on_trackbar)
    cv2.createTrackbar('Min value:', title_window , 0, 255, on_trackbar)
    cv2.createTrackbar('Max value:', title_window , 0, 255, on_trackbar)
    
    # read the image
    object_image = cv2.imread(image_path)
    object_image = imutils.resize(object_image, width=1200)
    # cv2.imshow('object image', object_image)

    # convert to hsv
    hsv = cv2.cvtColor(object_image, cv2.COLOR_BGR2HSV)
    rgb = cv2.cvtColor(object_image, cv2.COLOR_BGR2RGB)

    # set up the ruler
    unitlenpxl = calibrate_ruler(hsv)

    # pop contours, lengh and stuff for every coquille
    treat_image(hsv, min_hsv, max_hsv, unitlenpxl)

    cv2.waitKey(0)
