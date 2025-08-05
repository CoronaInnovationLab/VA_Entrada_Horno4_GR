from utils import conectar_camara
import cv2 as cv
import time
import os

def preparar_img(img):

    # Convertir la imagen a formato BGR para OpenCV
    img = cv.cvtColor(img, cv.COLOR_BAYER_BG2BGR)

    #-------------------------------------------------------------------------------------------------
    # Resize
    img_final = cv.resize(img, (768,576), interpolation=cv.INTER_CUBIC)
  
    return img_final

# ----------------------------------------------------------------------------------------------------------
# Conexión camara y toma de imagen
# ----------------------------------------------------------------------------------------------------------
h, ia = conectar_camara('salida 1')

# Balance de blancos
# ia.remote_device.node_map.BalanceWhiteAuto.value = 'Off'  # 'off' 'On Demand'

balance = {'Red': 1.08001, 'Green': 1, 'Blue':1.91854}

for canal, color in balance.items():
    ia.remote_device.node_map.BalanceRatioSelector.value = canal
    ia.remote_device.node_map.BalanceRatio.value = color

captura_realizada = False
ia.start()
folder = '../00_Data/datos_salida'
contador_trigger = 0

while True:
    señal = ia.remote_device.node_map.LineStatus.value

    # Capturar una imagen de la cámara
    with ia.fetch() as buffer:
        # Obtener los datos del buffer
        component = buffer.payload.components[0]

        # Convertir los datos a una imagen numpy
        img_original_camara = component.data.reshape(component.height, component.width)

        img_original_camara_copy = img_original_camara.copy()

    # Preprocesar la imagen
    frame = preparar_img(img_original_camara_copy)

    if señal:
        if not captura_realizada:
            captura_realizada = True
            contador_trigger +=1
            if contador_trigger == 2:
                timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
                cv.imwrite(os.path.join(folder, f'{timestamp}.png'), frame)
            if contador_trigger == 3:
                contador_trigger = 0

    else:
        if captura_realizada:
            captura_realizada = False

    # Texto de señal en imagen
    cv.putText(frame,f'Señal: {str(señal)}', (30,60), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv.LINE_AA)
    cv.imshow('camera', frame)

    key = cv.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('s'):
        timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
        cv.imwrite(f'captura_salida_{timestamp}.png', frame)


# Detener Camara
ia.stop()
ia.destroy()
h.reset()   
cv.destroyAllWindows()