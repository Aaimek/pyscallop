from typing import Tuple
import matplotlib
import cv2
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

# Calcul de la distance entre deux points d'une image
def distance(pt1: Tuple, pt2: Tuple):
    x, y = pt1
    a, b = pt2
    dist = (x-a)**2 + (y-b)**2
    
    return dist**0.5

# Fonction utile pour tracer la distribution des mesures a la fin d'une vidéo
def draw_distrib(ser):
    data = ser.astype('float64').to_numpy()
    mu, std = norm.fit(data)

    fig, ax = plt.subplots()
    width_inch = 7
    height_inch = 0.75 * width_inch
    fig.set_size_inches(width_inch, height_inch)
    text = "Moyenne: " + str(round(mu, 2)) + " cm\n"
    text += "Ecart-type: " + str(round(std, 2)) + " cm\n"
    text += "Nombre de mesures: " + str(len(data))
    at = AnchoredText(text, prop=dict(size=10), frameon=True, loc='upper left')
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(at)

    ax.hist(data, bins='auto')
    ax.set_title("Histogramme des mesures d'une même coquille")
    ax.set_xlabel("Taille (cm)")
    ax.set_ylabel("Nombre de mesures")
    ax.axvline(mu, color='k', linestyle='dashed', linewidth=1)
    ax.legend(['Moyenne', 'Nombre de mesures'])

    plt.savefig('./output/histograme_mesures_non_corrigées.jpg', dpi=1000)
    plt.show()

# Changement la taille d'une image en conservant son format
def resize(object_image, width):
    img_w, img_h = int(object_image.shape[1]), int(object_image.shape[0])
    aspect_ratio = img_w/img_h
    
    height = int(width/aspect_ratio)
    desired_size = (width, height)
    
    return cv2.resize(object_image, desired_size)