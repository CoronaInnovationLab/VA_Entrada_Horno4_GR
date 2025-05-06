from harvesters.core import Harvester
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
# Inicializar Harvester
h = Harvester()

cti_file = "C:/Program Files/MATRIX VISION/mvIMPACT Acquire/bin/x64/mvGENTLProducer.cti"

h.add_file(cti_file)

# Actualizar la lista de cámaras disponibles
h.update()

try:
    # Conectar a la segunda cámara disponible
    # print(h._device_info_list[0]['serial_number'])
    ia = h.create(1)
    print(ia.device_access_status) 
    print('Camara conectada')
except Exception as e:
    print('Camara no disponible')
    sys.exit(1)

ia.start()

# while True: 
# Capturar una imagen de la cámara
with ia.fetch() as buffer:
    print('a')
    # Obtener los datos del buffer
    component = buffer.payload.components[0]
    # Convertir los datos a una imagen numpy
    image = component.data.reshape(component.height, component.width)

    # Preprocesar la imagen
    frame = preparar_img(image)

    cv.imshow('Camera Feed', frame)

    k = cv.waitKey(1)
    if k == ord('q'):
        print('Operacion terminada por el usuario.')
        # break
    elif k == ord('s'):
        cv.imwrite('ak.png', frame)
# Detener Camara
ia.stop()
ia.destroy()
h.reset()   
sys.exit()