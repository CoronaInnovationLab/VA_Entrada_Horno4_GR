from utils import conectar_camara
import cv2 as cv
import time



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

while True:
    señal = ia.remote_device.node_map.LineStatus.value

    print(señal)
    if not señal:
        if captura_realizada:
            time.sleep(7)
            cv.destroyAllWindows()

    else:
        captura_realizada = True
        # timestamp = time.strftime('%Y-%m-%d_%H-%M-%S', time.time())
        timestamp = 'asd'
        print('Capturando datos', flush=True)
        
        ia.start()
        # Capturar una imagen de la cámara
        with ia.fetch() as buffer:
            # Obtener los datos del buffer
            component = buffer.payload.components[0]

            # Convertir los datos a una imagen numpy
            img_original_camara = component.data.reshape(component.height, component.width)

            # Crear una copia de la imagen
            img_original_camara_copy = img_original_camara.copy()

        # Detener la camara
        ia.stop()

        # Preprocesar la imagen
        frame = preparar_img(img_original_camara_copy)

        # Showing the image
        # cv.imshow('Camera Feed', frame)
        cv.imwrite(f'{timestamp}.png', frame)
        key = cv.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv.imwrite('captura_salida.png', frame)

# Detener Camara
ia.stop()
ia.destroy()
h.reset()   
# cv.destroyAllWindows()