import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import imutils
from matplotlib.colors import hsv_to_rgb
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
import time
import uuid

start_time = time.time()

def plot_hsv(image_path):
    img = cv2.imread(image_path)
    img = imutils.resize(img, width=1200)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, s, v = cv2.split(hsv)
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection="3d")

    pixel_colors = rgb.reshape((np.shape(rgb)[0]*np.shape(rgb)[1], 3))
    norm = colors.Normalize(vmin=-1.,vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()

    axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
    axis.set_xlabel("Teinte")
    axis.set_ylabel("Saturation")
    axis.set_zlabel("Intensité")

    #create an uuid for the image name
    uid = str(uuid.uuid4())[:4]
    save_path ='./output/3d_hsv_colorspace' + uid + '.jpg'
    plt.savefig(save_path, dpi=1000)

image_paths = ['./assets/coq_mix.jpg', './assets/coq_mix_from_real_2.jpg', './assets/coq_mix_background.jpg']

for path in image_paths:
    plot_hsv(path)    

print("--- %s seconds ---" % (time.time() - start_time))

