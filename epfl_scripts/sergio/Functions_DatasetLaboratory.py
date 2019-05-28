# -*- coding: utf-8 -*-
"""
Created on May 2019

@author: SergioF

Funciones revisadas y que funcionan correctamente para Dataset Laboratory
"""

"""
Para Abel --- Listado de funciones  --> marcadas las que vas a necesitar

    rectangle_text(img,ident,x1,x2,y1,y2)
    get_iou(bb1, bb2)
    Calc_Histrogram(path)
    Calc_Histogram_Simple(image, mascara)
-->    Calc_Histogram_Simple_Norm(image, mascara, boolean)
    Calc_DistanceMatrix(vect1, vect2)
-->    Calc_Distance(hist1, hist2)
    Calc_HOG(path)
    plot_matrix(DistMatrix, ImagesList, color, name, savingpath, vmin, vmax, show=True, save=True)
    Matrix_SuperPixel(ListaImages, Matrix)
    Process_random_images(camnumber, BB_path, BBmasks_path, newfolder, nframes)
-->    Calc_MaskRatio(img, boolean)
-->    Calc_AspectRatio(img, boolean)
    Find_ncam_BBs(BBs, Masks, BB_path, BBmasks_path)
-->    IstheSamePerson(BB_A, Mask_A, BB_B, Mask_B, boolean, functions)
-->    PersonsInaFrame_Vall(frame_t, frame_t_menos1, boolean, functions)
    PersonsInDifCams_Vall(CAM_A, CAM_B, npersons, boolean, functions):
    

    
"""
# Importo las librerias necesarias

import os

import cv2
import numpy as np
from scipy.spatial import distance  # distancia entre 2 histogramas


#%%
def rectangle_text(img,ident,x1,x2,y1,y2):
    """
    Recuadra el BB introducido y lo printa en img con su identificador (ident)
    :param img: imagen en RGB
    :param ident: identificador numerico del BB
    :param x1: The top left x-coordinate of the bounding box.
    :param x2: The bottom right x-coordinate of the bounding box.
    :param y1: The top left y-coordinate of the bounding box.
    :param y2: The bottom right y-coordinate of the bounding box.
    :return: print "ok"
    """
    cv2.rectangle(img,(int(x1),int(y1)),(int(x2),int(y2)),(255,0,0),2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "id = "+str(ident),(int(x1),int(y1)-5), font, 0.5,(255,255,255),1,cv2.LINE_AA)

    return "ok"

#%%
def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    Calcula el % de overlap entre 2 bounding boxes respecto a su area total. Si no hay overlap, return 0.
    :param bb1: [xmin, ymin, xmax, ymax] siendo:
        xmin: The top left x-coordinate of the bounding box.
        ymin: The top left y-coordinate of the bounding box.
        xmax: The bottom right x-coordinate of the bounding box.
        ymax: The bottom right y-coordinate of the bounding box.
    :param bb2: [xmin, ymin, xmax, ymax] siendo:
        xmin: The top left x-coordinate of the bounding box.
        ymin: The top left y-coordinate of the bounding box.
        xmax: The bottom right x-coordinate of the bounding box.
        ymax: The bottom right y-coordinate of the bounding box.
    :return: iou = float[0,1] siendo el % de overlap
    """
    assert bb1[0] < bb1[2]
    assert bb1[1] < bb1[3]
    assert bb2[0] < bb2[2]
    assert bb2[1] < bb2[3]

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:

        iou = 0.0

    else:
        # The intersection of two axis-aligned bounding boxes is always an
        # axis-aligned bounding box
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # compute the area of both AABBs
        bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
        bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
        assert iou >= 0.0
        assert iou <= 1.0

    return iou

#%%
def Calc_Histrogram(path):
    """
    Calcula los histogramas de una lista de imagenes RGB dado el path donde se encuentran
    :param path: direccion donde se encuentra la lista de imagenes RGB
    :return: (hist_vect_blue, hist_vect_green, hist_vect_red), histogramas en int32 de cada canal (B, G, R)
    """

    ImagesList = os.listdir(path)

    hist_vect_blue = []
    hist_vect_green = []
    hist_vect_red = []

    for idx,recorte in enumerate(ImagesList):
        img = cv2.imread(path + recorte,1) #leer imagen en RGB

        hist_blue = cv2.calcHist([img],[0],None,[256],[0,256])
        hist_green = cv2.calcHist([img],[1],None,[256],[0,256])
        hist_red = cv2.calcHist([img],[2],None,[256],[0,256])

        hist_blue_int = hist_blue.astype(int) #pasar de float32 a int32
        hist_green_int = hist_green.astype(int) #pasar de float32 a int32
        hist_red_int = hist_red.astype(int) #pasar de float32 a int32

        hist_vect_blue.append(hist_blue_int)
        hist_vect_green.append(hist_green_int)
        hist_vect_red.append(hist_red_int)

        if os.path.isdir(path):
            # skip directories
            continue

#    hist_vect_int = [i.astype(int) for i in hist_vect] #convertir de float32 a int32
#    return hist_vect_int

    return hist_vect_blue, hist_vect_green, hist_vect_red #lists of int32 numpy arrays

#%%
def Calc_Histogram_Simple(image, mascara):
    """
    Calcula los histogramas (B, G y R) de una imagen RGB dada la imagen con su path y su mascara
    :param img: imagen RGB en .bmp
    :param masks: imagen en escala de grises con valores 0 o 255
    :return: (hist_blue, hist_green, hist_red), histogramas en int32 de cada canal (B, G, R)
    """
    img = cv2.imread(image,1) #leer imagen en RGB
    mask = cv2.imread(mascara,0) #leer imagen en escala de grises

    hist_blue = cv2.calcHist([img],[0],mask,[256],[0,256])
    hist_green = cv2.calcHist([img],[1],mask,[256],[0,256])
    hist_red = cv2.calcHist([img],[2],mask,[256],[0,256])

    hist_blue_int = hist_blue.astype(int) #pasar de float32 a int32
    hist_green_int = hist_green.astype(int) #pasar de float32 a int32
    hist_red_int = hist_red.astype(int) #pasar de float32 a int32

    return hist_blue_int, hist_green_int, hist_red_int #histograms int32

#%%
def Calc_Histogram_Simple_Norm(image, mascara, boolean):
    """
    Calcula los histogramas (B, G y R) de una imagen RGB dada la imagen y su mascara NORMALIZADOS!
    :param img: ruta a la imagen / imagen RGB
    :param masks: ruta a la mascara / mask en escala de grises con valores 0 o 255
    :param boolean: boolean=True -> rutas de img y masks, boolean=False -> img en RGB y masks en grayscale
    :return: (hist_blue, hist_green, hist_red), histogramas en float32 de cada canal (B, G, R)
    """
    if boolean == True:
        img = cv2.imread(image,1) #leer imagen en RGB
        mask = cv2.imread(mascara,0) #leer imagen en escala de grises
    else:
        img = image
        mask = mascara

    module = cv2.countNonZero(mask)

    hist_blue = cv2.calcHist([img],[0],mask,[256],[0,256])
    hist_green = cv2.calcHist([img],[1],mask,[256],[0,256])
    hist_red = cv2.calcHist([img],[2],mask,[256],[0,256])

    b_copy = np.zeros((256,1),dtype=np.float32)
    g_copy = np.zeros((256,1),dtype=np.float32)
    r_copy = np.zeros((256,1),dtype=np.float32)

    for idx,bit in enumerate(zip(np.nditer(hist_blue), np.nditer(hist_green), np.nditer(hist_red))):

        b_copy[idx,0] = int(bit[0]) / module
        g_copy[idx,0] = int(bit[1]) / module
        r_copy[idx,0] = int(bit[2]) / module

    return b_copy, g_copy, r_copy #histograms float32 y sumatorio = 1

#%%

def Calc_DistanceMatrix(vect1, vect2):
    """
    Calcula la matriz de distancias euclideas entre 2 vectores dados elemento a elemento
    :param vect1: vector de caractaristicas 1 (vector de N histogramas)
    :param vect2: vector de caracteristicas 2 (vector de N histogramas)
    :return: matrizDistancias: matriz de distancias obtenida
    """
    tamaño_1 = len(vect1)
    tamaño_2 = len(vect2)

    if tamaño_1 != tamaño_2:
        print("error en tamaños de vectores")
        return
    else:
        matrizDistancias = np.empty((tamaño_1,tamaño_1))

#        matrizDistancias = distance_matrix(vect1, vect2) #He probado con solo un vector de (256,1) que
        # es un unico histograma y tampoco funciona! --> NO FUNCIONA!

        i = 0
        for i in range(tamaño_1):
            for j in range(tamaño_2):
            #        matrizDistancias[i,j] = cv2.compareHist(hist_vect[i][0],hist_vect[j][0],cv2.HISTCMP_BHATTACHARYYA)
                matrizDistancias[i,j] = distance.euclidean(vect1[i],vect2[j])
            j = 0
#            print(i)

#        plt.matshow(matrizDistancias)
#        plt.show()

        return matrizDistancias

#%%

def Calc_Distance(hist1, hist2):
    """
    Calcula la distancia euclidea entre dos histogramas simples
    :param hist1: histograma 1
    :param hist2: histograma 2
    :return: dist_value: valor en float32 de la distancia calculada
    """
    tamaño_1 = len(hist1)
    tamaño_2 = len(hist2)

    if tamaño_1 != tamaño_2:
        print("error en tamaños de vectores")
        return
    else:
        dist_value = distance.euclidean(hist1, hist2)

    return dist_value


#%%

def Calc_HOG(path):
    """
    Calcula los HOG de una lista de imagenes en escala de grises dado el path donde se encuentran
    :param path: ruta a la carpeta donde estan las imagenes
    :return: HOG_vect: lista con los HOG de las imagenes del path dado en formato float32 np array
    """

    ImagesList = os.listdir(path)

    hog = cv2.HOGDescriptor()

    winStride = (8,8)
    padding = (8,8)
#    locations = ((10,20),)

    HOG_vect = []

    for idx,recorte in enumerate(ImagesList):
        img = cv2.imread(path + recorte,0) #leer imagen en escala de grises
        Hog_des = hog.compute(img,winStride,padding,) #si pongo el locations me devuelve vectores con todo 0s
        HOG_vect.append(Hog_des)

        if os.path.isdir(path):
            # skip directories
            continue

    return HOG_vect #list of float32 numpy arrays


#%%
from matplotlib import pyplot as plt


def plot_matrix(DistMatrix, ImagesList, color, name, savingpath, vmin, vmax, show=True, save=True):
    """
    Plotea una matriz de distancias
    :param DistMatrix: matriz de distancias en formato float64
    :param color: barra de colores --> ejemplos: plt.cm.(autumn, gray, cool, winter, hsv, ...)
    :param name: nombre de la figura y formato --> ejemplo: "Matrix_B_pr2.png"
    :param path: ruta donde guardar el plot
    :param show: si es True muestra el plot
    :return: None
    """
    C = DistMatrix
#    f = plt.figure(figsize=(8,6)) #Figure dimension (width, height) in inches
#    ax = f.add_axes([0.005, 0.1, 0.8, 0.8]) #The dimensions [left, bottom, width, height] of the new axes
#    plt.title(name, fontsize=20)
#    axcolor = f.add_axes([0.8, 0.1, 0.03, 0.8])

    fig, ax = plt.subplots()
    im = ax.matshow(C, cmap=color, vmin=vmin, vmax=vmax)
#    im = ax.matshow(C, cmap=cm.gist_rainbow, norm=LogNorm(vmin=0.01, vmax=1))
#    im = ax.matshow(C, cmap=color, norm=Normalize(), vmin=0, vmax=400)

    ax.set_title(name, fontsize=20, loc='left', pad=100)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(ImagesList)))
    ax.set_yticks(np.arange(len(ImagesList)))

    # ... and label them with the respective list entries (esto las escribe todas --> demasiada info)
#    ax.set_xticklabels(ImagesList)
#    ax.set_yticklabels(ImagesList)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")

    #Marco el cambio de persona con lineas
    imgname_ant = ImagesList[0] #inicializo la imagen anterior
    Imglist_reducida = []

    for idx,imgname in enumerate(ImagesList):

        if imgname[0:3] != imgname_ant[0:3]:
            plt.axhline(y=(idx-0.5), color='r', linestyle='-') #le resto 0.5 para que no me corte los pixels
            plt.axvline(x=(idx-0.5), color='r', linestyle='-') #(ver en graficos y se entiende)
            Imglist_reducida.append(imgname)
        else:
            Imglist_reducida.append('')

        imgname_ant = imgname

    ax.set_xticklabels(Imglist_reducida)
    ax.set_yticklabels(Imglist_reducida)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, cmap=color)
    cbar.ax.set_ylabel("Euclidean distances", rotation=-90, va="bottom")

    if show == True:
#        f.show()
        fig.tight_layout()
        plt.show()
    else:
        None

    try:
        os.stat(savingpath)
    except:
        os.mkdir(savingpath)

    if save == True:
        plt.savefig(savingpath + name)
    else:
        None

#%%

def Matrix_SuperPixel(ListaImages, Matrix):
    """
    Calcula la media, mediana y desv estandar por persona dada una matriz de entrada con datos (de 6 personas)
    de una misma camara.
    :param ListaImages: lista de nombres de los BB de las personas que han sido calculadas para la Matrix
    :param Matrix: matriz de distancias euclideas de n x m de 6 personas
    :return matrix_mean: matriz con la media de los datos por persona
    :return matrix_median: matriz con la mediana de los datos por persona
    :return matrix_std: matriz con la desviacion de los datos por persona
    """

    p00 = []
    p01 = []
    p02 = []
    p03 = []
    p04 = []
    p05 = []

    for idx,img_name in enumerate(ListaImages):

        if img_name[0:3] == "p00":
            p00.append(idx)
        elif img_name[0:3] == "p01":
            p01.append(idx)
        elif img_name[0:3] == "p02":
            p02.append(idx)
        elif img_name[0:3] == "p03":
            p03.append(idx)
        elif img_name[0:3] == "p04":
            p04.append(idx)
        elif img_name[0:3] == "p05":
            p05.append(idx)
        else:
            print("error?")

    cambios_vector = [p00, p01, p02, p03, p04, p05]
    matrix_mean = np.zeros((len(cambios_vector),len(cambios_vector)))
    matrix_median = np.zeros((len(cambios_vector),len(cambios_vector)))
    matrix_std = np.zeros((len(cambios_vector),len(cambios_vector)))

    for indx,cambios in enumerate(cambios_vector):

        for indy,cambios_2 in enumerate(cambios_vector):

            values=[]

            for i in cambios:

                for j in cambios_2:

                    if Matrix[i,j] != 0: values.append(Matrix[i,j])

            matrix_mean[indx,indy] = np.mean(values)
            matrix_median[indx,indy] = np.median(values)
            matrix_std[indx,indy] = np.std(values)

    return matrix_mean, matrix_median, matrix_std

#%%

import shutil #(Shell Utilities) --> operations with files and directories

def Process_random_images(camnumber, BB_path, BBmasks_path, newfolder, nframes):
    """
    Copia el BB y el BB_mask de cada n frames en una nueva carpeta a elegir.
    :param camnumber: numero de la camara seleccionada --> "c00", "c01", "c02" o "c03"
    :param BB_path: directorio donde se encuentran los BB
    :param BBmasks_path: directorio donde se encuentran los BB_masks
    :param newfolder: ruta donde guardar crear la carpeta y nombre de la misma
    :param nframes: numero de veces que se cogera cada frame
    :return: None
    """

    lista_BB = os.listdir(BB_path + camnumber + "/")
    lista_BBmasks = os.listdir(BBmasks_path + camnumber + "/")

    if not os.path.exists(newfolder):
        os.makedirs(newfolder)

    for idx,i in enumerate(lista_BB):

        if (idx == 0) or (idx % nframes == 0):

            try:
                os.stat(newfolder)
            except:
                os.mkdir(newfolder)
            print("frame number: ", i)

            shutil.copy(BB_path + camnumber + "/" + i, newfolder) #copio el BB seleccionado a la carpeta newfolder

            if lista_BBmasks[idx-1][:-9] == i[:-4]: #quito el formato de imagen (_mask.png y el .bmp)
                #copio el BBmask seleccionado a la carpeta newfolder
                shutil.copy(BBmasks_path + camnumber + "/" + lista_BBmasks[idx-1], newfolder)

            elif lista_BBmasks[idx][:-9] == i[:-4]:
                shutil.copy(BBmasks_path + camnumber + "/" + lista_BBmasks[idx], newfolder)

            elif lista_BBmasks[idx+1][:-9] == i[:-4]:
                shutil.copy(BBmasks_path + camnumber + "/" + lista_BBmasks[idx+1], newfolder)

            else:
                print("error de coincidencia")
                break

#%%

def Calc_MaskRatio(img, boolean):
    """
    Calcula el % de pixels de la mascara respecto al BB
    :param img: boolean=True -> ruta a la mascara, boolean=False -> img en grayscale
    :param boolean: True or False
    :return: ratio: valor del porcentaje en float
    """
    if boolean == True:
        mask = cv2.imread(img,0) #leer imagen en escala de grises
    else:
        mask = img

    module = cv2.countNonZero(mask)
    ratio = module / np.size(mask)

    return ratio

def Calc_AspectRatio(img, boolean):
    """
    Calcula el ratio de altura/anchura de la mascara
    :param img: boolean=True -> ruta a la mascara, boolean=False -> img en grayscale
    :param boolean: True or False
    :return: ret: booleano para saber si se ha ejecutado correctamente
    :return: shape: altura/anchura en float
    """
    if boolean == True:
        mask = cv2.imread(img,0) #leer imagen en escala de grises
    else:
        mask = img

    image,contours,hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 1:
        cnt = contours[0]
        x,y,w,h = cv2.boundingRect(cnt)

        shape = h / w
        ret = True
    else:
        shape = 0
        ret = False

    return ret, shape

#%%

#import shutil #(Shell Utilities) --> operations with files and directories

def Find_ncam_BBs(BBs, Masks, BB_path, BBmasks_path):
    """
    Copia los mismos BBs y BB_masks de todas las camaras, partiendo de los BBs y Masks de una sola cam
    :param BBs: path al conjunto de BBs de partida de una camara
    :param Masks: path al conjunto de masks de partida de una camara
    :param BB_path: path donde estan todos los BBs
    :param BBmasks_path: path donde estan todas las mascaras
    :return: None
    """
    lista_BBs_dados = os.listdir(BBs)
    lista_Masks_dadas = os.listdir(Masks)
    actual_cam = BBs[-4:-1]
#    lista_BBmasks = os.listdir(BBmasks_path + camnumber + "/")

    for ncam in os.listdir(BB_path):

        if ncam != actual_cam:

            for BB_name in lista_BBs_dados:

                BB_new = BB_name.replace(actual_cam, ncam) #sustituyo el string "c0x" por la nueva cam "c0n"

                if not os.path.exists(BBs[:-4] + ncam + "/"):
                    os.makedirs(BBs[:-4] + ncam + "/")
                try:
                    shutil.copy(BB_path + ncam + "/" + BB_new, BBs[:-4] + ncam + "/")
                    #copio el BB seleccionado a la nueva carpeta
                except:
                    print("error coincidencia en: " + BB_new)

            for Mask_name in lista_Masks_dadas:

                Mask_new = Mask_name.replace(actual_cam, ncam) #sustituyo el string "c0x" por la nueva cam "c0n"

                if not os.path.exists(Masks[:-4] + ncam + "/"):
                    os.makedirs(Masks[:-4] + ncam + "/")
                try:
                    shutil.copy(BBmasks_path + ncam + "/" + Mask_new, Masks[:-4] + ncam + "/")
                    #copio el BB seleccionado a la nueva carpeta
                except:
                    print("error coincidencia en: " + Mask_new)

    print("completado")
    return

#%%

def IstheSamePerson(BB_A, Mask_A, BB_B, Mask_B, boolean, functions):
    """
    Compara si dos personas son la misma o no con sus BB y mascaras
    :param BB_A: path completo al BB / img de la persona A
    :param Mask_A: path completo a la mascara / mask de la persona A
    :param BB_B: path completo al BB / img de la persona B
    :param Mask_B: path completo a la mascara / mask de la persona B
    :param boolean: True or False, if True -> rutas images, if False -> images en BGR y masks en grayscale
    :param functions: fd de funciones
    :return score: int [0,7]
    """

    #Miro el ratio de mascara respecto al total de BB
    pA_ratio = functions.Calc_MaskRatio(Mask_A, boolean)
    pB_ratio = functions.Calc_MaskRatio(Mask_B, boolean)

    #Miro la altura/anchura de la mascara
    retA, pA_shape = functions.Calc_AspectRatio(Mask_A, boolean)
    retB, pB_shape = functions.Calc_AspectRatio(Mask_B, boolean)

#    print("Mask porcentaje : %.4f %.4f"%(pA_ratio,pB_ratio))
#    print("Aspect ratio de la mascara:  %.4f %.4f" %(pA_shape, pB_shape))

    #Calculo los histogramas normalizados en los 3 canales
    pA_b, pA_g, pA_r = functions.Calc_Histogram_Simple_Norm(BB_A, Mask_A, boolean)
    pB_b, pB_g, pB_r = functions.Calc_Histogram_Simple_Norm(BB_B, Mask_B, boolean)

    #Distancia de los histogramas por canal
    dist_b = functions.Calc_Distance(pA_b, pB_b)
    dist_g = functions.Calc_Distance(pA_g, pB_g)
    dist_r = functions.Calc_Distance(pA_r, pB_r)

    data = [pA_ratio, pB_ratio, pA_shape, pB_shape, dist_b, dist_g, dist_r]
    data= {
        'pA_ratio': pA_ratio,
        'pB_ratio': pB_ratio,
        'pA_shape': pA_shape,
        'pB_shape': pB_shape,
        'dist_b': dist_b,
        'dist_g': dist_g,
        'dist_r': dist_r
        }

    score = 0

    if pA_ratio > 0.3:
        score = score + 1
    if pB_ratio > 0.3:
        score = score + 1
    if pA_shape > 1.5:
        score = score + 1
    if pB_shape > 1.5:
        score = score + 1
    if dist_b < 0.05:
        score = score + 1
    if dist_g < 0.05:
        score = score + 1
    if dist_r < 0.05:
        score = score + 1

    return score, data


#%%
def PersonsInaFrame_Vall(frame_t, frame_t_menos1, boolean, functions):
    """
    Compara las personas que hay en un frame con las del frame anterior
    :param frame_t: [BB1, Mask1, BB2, Mask2, ..., npersons] en el instante (t)
    :param frame_t_menos1: [BB1, Mask1, BB2, Mask2, ..., npersons] en el instante (t-1)
    :param boolean: True or False, if True -> rutas images, if False -> images en BGR y masks en grayscale
    :param functions: fd de funciones
    :return matrix: matriz del tamaño npersons_t x npersons_t_menos1 en int32
    """
    npersons_t = int(len(frame_t)/2)
    npersons_t_menos1 = int(len(frame_t_menos1)/2)
    matrix = np.zeros((npersons_t, npersons_t_menos1))
    data_results = []

    for i in range(npersons_t):

        for j in range(npersons_t_menos1):

            matrix[i, j], data = functions.IstheSamePerson(frame_t[(2*i)], frame_t[(2*i+1)],
                  frame_t_menos1[(2*j)], frame_t_menos1[(2*j+1)], boolean, functions)

            data_results.append([i, j, data])

    print(matrix)

    return (matrix.astype(int)), data_results


#%%
#Funcion imperfecta si el numero de personas no coincide en la CAM_A y la CAM_B

def PersonsInDifCams_Vall(CAM_A, CAM_B, boolean, functions):
    """
    Compara las personas que hay en dos camaras (A y B) en los instantes de tiempo (t) y (t-1)
    :param CAM_A: ([BB1A, Mask1A, BB2A, Mask2A, ..., npersons][BB1A, Mask1A, BB2A, Mask2A, ..., npersons]) en los instantes (t) y (t-1) respectivamente
    :param CAM_B: ([BB1B, Mask1B, BB2B, Mask2B, ..., npersons][BB1B, Mask1B, BB2B, Mask2B, ..., npersons]) en los instantes (t) y (t-1) respectivamente
    :param boolean: True or False, if True -> rutas images, if False -> images en BGR y masks en grayscale
    :param functions: fd de funciones
    :return: matriz del tamaño npersons x npersons en int32
    """

    matrix_A = functions.PersonsInaFrame_Vall(CAM_A[0], CAM_A[1], boolean, functions)
    matrix_B = functions.PersonsInaFrame_Vall(CAM_B[0], CAM_B[1], boolean, functions)

    print('Matriz A: ', matrix_A, ' matriz B: ', matrix_B)

    matriz = matrix_A + matrix_B

    return matriz

