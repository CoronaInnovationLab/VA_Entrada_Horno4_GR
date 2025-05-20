from harvesters.core import Harvester
from utils import conectar_camara
import datetime
import cv2 as cv
import numpy as np
import os
import time
import sys


def preparar_img(img):

    # Convertir la imagen a formato BGR para OpenCV
    img = cv.cvtColor(img, cv.COLOR_BAYER_BG2BGR)

    #-------------------------------------------------------------------------------------------------
    # Resize
    recorte = cv.resize(img, (768,576), interpolation=cv.INTER_CUBIC)

    img_final = recorte
    
    return img_final

# ----------------------------------------------------------------------------------------------------------
# Conexión camara y toma de imagen
# ----------------------------------------------------------------------------------------------------------
h, ia = conectar_camara('salida 1')
ia.start()

while True:
    # Capturar una imagen de la cámara
    with ia.fetch() as buffer:
        señal = ia.remote_device.node_map.LineStatus.value
        # if señal:
        print(señal)
        # Obtener los datos del buffer
        component = buffer.payload.components[0]
        # Convertir los datos a una imagen numpy
        image = component.data.reshape(component.height, component.width)

        # Preprocesar la imagen
        frame = preparar_img(image)

        cv.imshow('Camera Feed', frame)
        key = cv.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv.imwrite('captura_salida.png', frame)
# Detener Camara
ia.stop()
ia.destroy()
h.reset()   