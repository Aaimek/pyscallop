from contextlib import redirect_stderr
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import imutils
import helper
import pandas as pd

# Chemin d'accès aux vidéos/photos
image_path = './assets/coq_mix_from_real_2_concombre.jpg'
video_path = './assets/videos/3_coq.MP4'

# Paramètres du filtre HSV principal
min_hsv = (0, 50, 0)
max_hsv = (15, 255, 250)

# Trouver l'étalon bleu et mesurer sa hauteur pour déterminer l'échelle
def calibrate_ruler(hsv_image):
    ruler_min_hsv = (50, 70, 220)
    ruler_max_hsv = (125, 255, 255)

    # Filtrer l'image HSV
    mask = cv2.inRange(hsv_image, ruler_min_hsv, ruler_max_hsv)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    mask = cv2.bitwise_not(mask)
    gray = mask

    # Seuil de luminosité
    tresh = 100
    ret, tresh_img = cv2.threshold(gray, tresh, 255, cv2.THRESH_BINARY)

    # Chercher les contours de l'image
    contours, _ = cv2.findContours(tresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Création d'une image vide pour y afficher les contours
    contours_img = np.zeros(hsv_image.shape)

    # Recherche du plus gros contour
    biggest_contour = contours[-1]

    # Détermination du rectangle le plus petit englobant ce contour (étalon)
    x,y,w,h = cv2.boundingRect(biggest_contour)

    # Affichange du rectangle
    cv2.rectangle(contours_img,(x,y),(x+w,y+h),(255,0,0),3)
    cv2.drawContours(image=contours_img, contours=biggest_contour, contourIdx=-1, color=(0, 255, 0), thickness=5, lineType=cv2.LINE_AA)

    # Affichange de l'échelle en texte sur l'image
    text = "10cm = " + str(h) + " pixels"
    center = (int(x)-300, int(y)-10)
    cv2.putText(contours_img, text, center, cv2.FONT_HERSHEY_SIMPLEX, 1, (209, 80, 0, 255), 3)

    # Affichange de l'image
    cv2.imshow('ruler contours', contours_img)
    cv2.imwrite('./output/ruler_contours.jpg', contours_img)

    # échelle en pixel/mètre
    # 10cm = x pxl
    # 1m = 10*x pxl
    return 10*h * 0.8 #10*h


# Filtrer l'image, déterminer les contours, les cercles, les dimentions
# Puis tout stocker dans un dataframe ./output/contours_df.pkl
def treat_image(hsv_image, object_image, min_hsv, max_hsv, scale):
    # Filtrer l'image HSV
    mask = cv2.inRange(hsv_image, min_hsv, max_hsv)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=4)
    mask = cv2.bitwise_not(mask)
    gray = mask

    # seuil de luminosité
    tresh = 100
    ret, tresh_img = cv2.threshold(gray, tresh, 255, cv2.THRESH_BINARY)

    # déterminer les contours sur l'image
    contours, _ = cv2.findContours(tresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Création d'une image vide pour y afficher les contours
    contours_img = np.zeros(object_image.shape)
    # Affichange des contours sur l'image vide (fond noir)
    cv2.drawContours(image=contours_img, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=5, lineType=cv2.LINE_AA)

    # On fait passer une série de séléction aux contours, pour ne garder que ceux qui correspondent probablement a une coquille
    # Consiste a regarder la taille
    # Eliminer les contours qui se trouvent les uns sur les autres

    # Ce dataFrame contiendras toutes les donées sur les contours
    # contour: contour object
    # center: (x, y)
    # radius: rayons du cercle (en pixels)
    # lengh: envergure de l'objet
    # state: NOISE, GOOD ou BAD
    contours_df = pd.DataFrame(columns=['contour', 'center', 'radius', 'lengh', 'state'])

    # Remplir le dataFrame avec les contours
    contours_df = contours_df.append([{'contour': contours[i], 'center': None, 'radius': None, 'lengh': None, 'state': None} for i in range(len(contours))], ignore_index=True)

    # Elimination des objects étant trops petits (NOISE)
    # Classification des autres en fonction de leur taille (>10.2 = GOOD et <=10.2 = BAD)
    for index, row in contours_df.iterrows():
        contour = row.contour

        # cercle minimal englobant ce contour
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x),int(y))
        radius = int(radius)

        # Eliminer si la taille est bien trop petite
        if radius < scale/30:
            row.state = 'NOISE'
        else:
            # x px = 1m
            # y px = y/x m
            lengh = radius*2 / scale # in m (meters)
            row.lengh = lengh
            row.center = center
            row.radius = radius

            # Critère primordial
            if lengh >= 0.102:
                row.state = 'GOOD'
            else:
                row.state = 'BAD'
    
    # Eliminer le contour qui prend toute l'image
    biggestrow = contours_df.sort_values(by='lengh', ascending=False).iloc[0]
    max_lengh = biggestrow.lengh
    contours_df.loc[contours_df.lengh==max_lengh, 'state'] = 'NOISE'

    # Eliminer les contours qui se trouvent les uns dans les autres
    not_noise = contours_df[contours_df.state!='NOISE']

    for index1, row1 in not_noise.iterrows():
        for index2, row2 in not_noise.iterrows():

            # Eviter le cas ou on doit vérifier si un cercle est contenu dans lui-même
            if index1==index2:
                continue
            
            dist = helper.distance(row1.center, row2.center)
            max_radius, min_radius = max(row1.radius, row2.radius), min(row1.radius, row2.radius)

            if dist < max_radius:
                # On élimine le petit cercle qui est dans le gros, en le passant en NOISE
                contours_df.loc[contours_df.radius==min_radius, 'state'] = 'NOISE'
                

    # Sauvegarder le dataFrame
    contours_df.to_pickle('./output/data/contours_df.pkl')

    # Afficher le dernier masque
    # cv2.imshow('Mask', mask)

    # fonction définie juste après
    draw_output(contours_df, object_image)

# Afficher les images nécéssaires:
# - Image de base + contours des objets et tailles
# - Les différentes étapes du filtre
# - L'image telle qu'elle est vue par le programme
def draw_output(contours_df, object_image):
    # Image vide pour afficher les contours
    contours_img = np.zeros(object_image.shape)

    # Image originalle pour afficher les contours par-dessus
    rendr_img = object_image.copy()

    # itération dans le dataFrame, contour par contour
    for index, row in contours_df[contours_df['state'] != 'NOISE'].iterrows():
        contour = row.contour
        center = row.center
        radius = row.radius
        lengh = row.lengh
        state = row.state

        # Tracer le cercle
        cv2.circle(contours_img, center, radius, (0,0,255), 2)
        cv2.circle(rendr_img, center, radius, (0,0,255), 2)

        lengh_cm = lengh*100 #cm
        lengh_text = str(round(lengh_cm, 1)) + ' cm'
        cv2.putText(contours_img, lengh_text, center, cv2.FONT_HERSHEY_SIMPLEX, 1, (209, 80, 0, 255), 3)
        cv2.putText(rendr_img, lengh_text, center, cv2.FONT_HERSHEY_SIMPLEX, 1, (209, 80, 0, 255), 3)

        # Afficher si la coquille est bonne ou non
        # Centre du texte a afficher
        x, y = center
        y += 35

        if state == 'GOOD':
            cv2.putText(contours_img, 'GOOD', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0, 255), 4)
            cv2.putText(rendr_img, 'GOOD', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0, 255), 4)
            cv2.drawContours(image=contours_img, contours=contour, contourIdx=-1, color=(0, 255, 0), thickness=5, lineType=cv2.LINE_AA)
        elif state == 'BAD':
            cv2.putText(contours_img, 'BAD', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255, 255), 4)
            cv2.putText(rendr_img, 'BAD', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255, 255), 4)
            cv2.drawContours(image=contours_img, contours=contour, contourIdx=-1, color=(0, 0, 255), thickness=5, lineType=cv2.LINE_AA)
        elif state == 'NOISE':
            # pass
            cv2.drawContours(image=contours_img, contours=contour, contourIdx=-1, color=(255, 255, 255), thickness=5, lineType=cv2.LINE_AA)

    # Ecrire en temps réel les coquilles détectées
    # print('Nombre de coquilles valides: ' + str(len(contours_df[contours_df['state'] == 'GOOD'])))
    # print('Nombre de coquilles invalides: ' + str(len(contours_df[contours_df['state'] == 'BAD'])))

    # Afficher et sauvegarder les différentes inages
    cv2.imshow('contours', contours_img)
    cv2.imwrite('./output/contours.jpg', contours_img)
    cv2.imshow('rendr_image', rendr_img)
    cv2.imwrite('./output/rendr_image.jpg', rendr_img)

# Traiter et afficher une seule image
def image(image_path):   
    # Lecture de l'image originale
    object_image = cv2.imread(image_path)
    object_image = imutils.resize(object_image, width=1200)

    # Affichage de l'image originale
    #cv2.imshow('object image', object_image)

    # Copies dans différents espaces de couleurs
    hsv = cv2.cvtColor(object_image, cv2.COLOR_BGR2HSV)
    rgb = cv2.cvtColor(object_image, cv2.COLOR_BGR2RGB)

    # Etaloner la mesure (pxl/m)
    scale = calibrate_ruler(hsv)

    # Afficher tout les contours etc...
    treat_image(hsv, object_image, min_hsv, max_hsv, scale)

    cv2.waitKey(0)

# Traiter et afficher une vidéo, image par image
def video(video_path):    
    cap = cv2.VideoCapture(video_path)
    while (cap.isOpened()):
        _, object_image = cap.read()
        object_image = imutils.resize(object_image, width=1200)


        # Copies dans différents espaces de couleurs
        hsv = cv2.cvtColor(object_image, cv2.COLOR_BGR2HSV)
        rgb = cv2.cvtColor(object_image, cv2.COLOR_BGR2RGB)

        # Etaloner la mesure (pxl/m)
        scale = calibrate_ruler(hsv)

        # Afficher tout les contours etc...
        treat_image(hsv, object_image, min_hsv, max_hsv, scale)

        # Appuyer sur n'importe quelle touche pour passer a l'image suivante
        # Appuyer sur x pour quitter
        cv2.waitKey(0)

        if cv2.waitKey(0) == ord('x'): # exit
            break

    cap.release()
    cv2.destroyAllWindows()

# Produit les données pour pouvoir obtenir des résultats statistiques sur les mesures
# Sauvegarde la somme de tout les dataFrames d'une vidéo
# Permet d'avoir tout les objets détectés dans une vidéo
# Fournir une vidéo contenant un seul objet
def histogram(video_path):
    # grand dataFrame final
    all_contours_df = pd.DataFrame(columns=['contour', 'center', 'radius', 'lengh', 'state'])
    
    # lecture de la vidéos
    cap = cv2.VideoCapture(video_path)
    while (cap.isOpened()):
        ret, object_image = cap.read()
        if not ret:
            print("Can't receive frame")
            break

        object_image = imutils.resize(object_image, width=1200)

        # Copies dans différents espaces de couleurs
        hsv = cv2.cvtColor(object_image, cv2.COLOR_BGR2HSV)
        rgb = cv2.cvtColor(object_image, cv2.COLOR_BGR2RGB)

        # Etaloner la mesure (pxl/m)
        scale = calibrate_ruler(hsv)

        # Afficher tout les contours etc...
        treat_image(hsv, object_image, min_hsv, max_hsv, scale)

        # lecture du dataFrame et ajout au grand dataFrame
        contours_df = pd.read_pickle('./output/data/contours_df.pkl')
        all_contours_df = pd.concat([all_contours_df, contours_df])

    cap.release()
    cv2.destroyAllWindows()

    # Sauvegarder le grand dataFrame final
    all_contours_df.to_pickle('./output/data/all_contours.pkl')
    print("Saved dataframe as ./output/data/all_contours.pkl")

if __name__ == '__main__':
    histogram(video_path)