from harvesters.core import Harvester
import datetime
import cv2 as cv
import numpy as np
import os
import time
import sys

def log(msg):
    print(f'[{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {msg}\n')

def preparar_crudo(img):
    # Convertir la imagen a formato BGR para OpenCV
    img = cv.cvtColor(img, cv.COLOR_BAYER_BG2BGR)

    # se requiere tamaño original
    # img_final = cv.resize(img, (768,576), interpolation=cv.INTER_CUBIC)

    return img

def preparar_img(img):

    # Convertir la imagen a formato BGR para OpenCV
    img = cv.cvtColor(img, cv.COLOR_BAYER_BG2BGR)

    #-------------------------------------------------------------------------------------------------
    # Resize
    recorte = cv.resize(img, (768,576), interpolation=cv.INTER_CUBIC)

    img_final = recorte
    
    return img_final

# ----------------------------------------------------------------------------------------------------------------------
#                                          FRAMES  DEFINITION
# ----------------------------------------------------------------------------------------------------------------------
# Directorio donde se guardarán los fotogramas
output_dir = 'capturas'

if not os.path.exists(output_dir):
    os.makedirs(os.path.join(output_dir, ['fotos', 'salida']))

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
    # Conectar a la primera cámara disponible
    ia = h.create(0)
    log('Camara conectada')
except Exception as e:
    log('Camara no disponible')
    sys.exit(1)

ia.start()

# Inicializar variable para el video writer
guardar_fotos = False

inventario_final: dict = {key: 0 for key in ['Lavamanos', 'OnePiece', 'Taza', 'Tanque', 'Pedestal']}

# Lista de colores en formato RGB
colores = {
    'Taza': (255, 0, 0),    # Rojo
    'OnePiece': (0, 255, 0),    # Verde
    'Lavamanos': (0, 0, 255),    # Azul
    'Tanque': (255, 255, 0),  # Amarillo
    'Pedestal': (255, 0, 255),  # Magenta
    # 'Plafon': (0, 255, 255)   # Cian
}

def print_resultados(frame: np.ndarray, inventario_final: dict, colores: dict) -> None:
    """
    Muestra los resultados en el frame, con el texto de las clases en blanco
    y las cantidades en el color asociado a cada clase.
    """
    x, y = 50, 50  # Posición inicial para el texto
    font_scale = 0.5
    thickness = 2

    for clase, cantidad in inventario_final.items():
        # Texto de clase 
        clase_texto = f"{str(clase)}: "
        cv.putText(frame, clase_texto, (int(x), int(y)), cv.FONT_HERSHEY_SIMPLEX, 
                   font_scale, (255, 255, 255), thickness, cv.LINE_AA)
        
        # Texto de cantidad
        cantidad_texto = str(cantidad)
        
        # Calculo de coordenadas para "concatenar" el texto
        text_size = cv.getTextSize(clase_texto, cv.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        cantidad_x = x + text_size[0]
        cv.putText(frame, cantidad_texto, (int(cantidad_x), int(y)), cv.FONT_HERSHEY_SIMPLEX, 
                   font_scale, colores.get(clase, (255, 255, 255)), thickness, cv.LINE_AA)

        # Incrementar la posición x para la siguiente clase
        x += text_size[0] * 1.5

try:
    while True: 
        # Capturar una imagen de la cámara
        with ia.fetch() as buffer:
            # Obtener los datos del buffer
            component = buffer.payload.components[0]
            # Convertir los datos a una imagen numpy
            image = component.data.reshape(component.height, component.width)

            # Preprocesar la imagen
            frame = preparar_img(image)
            image = preparar_crudo(image)

            cv.imshow('Camera Feed', frame)
            cv.imshow('Crudo', image)

            #Preguntamos si se esta dando la confirmacion de movimiento carro
            if ia.remote_device.node_map.LineStatus.value == True:
                if guardar_fotos:
                    # Generar el nombre del archivo para el fotograma
                    nombre = time.strftime("%Y-%m-%d_%H-%M-%S")
                    frame_filename = os.path.join(output_dir, 'fotos', f'{nombre}.png')
                    # Guardar el fotograma como una imagen PNG
                    cv.imwrite(frame_filename, frame)

                log(f'Foto en: {frame_filename}')

            k = cv.waitKey(1)
            if k == ord('q'):
                log('Operacion terminada por el usuario.')
                break
            elif k == ord('s'):
                cv.imwrite(os.path.join(output_dir, 'fotos', 'ak.png'), frame)
except Exception as e:
    log(f'Error: {e}')
finally:
    # Detener Camara
    ia.stop()
    ia.destroy()
    h.reset()   
    sys.exit()