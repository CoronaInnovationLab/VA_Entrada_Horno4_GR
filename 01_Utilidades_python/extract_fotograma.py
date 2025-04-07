import cv2 as cv
import numpy as np
import time
import os

# Constantes
input_path = 'data/videos'

# Directorio donde se guardarán los fotogramas
out_path = 'data/train_img_por_filtrar'
if not os.path.exists(out_path):
    os.makedirs(out_path)

# Intervalo de fotogramas (por ejemplo, extraer cada 10 fotogramas)
frame_interval = 100

def preparar_img(img):

    # # Convertir la imagen a formato BGR para OpenCV
    # img = cv.cvtColor(img, cv.COLOR_BAYER_BG2BGR)

    # Corregir distorsion
    # Matriz de cámara
    camera_matrix = np.array([[2056, 0, 1025],
                            [0, 2064, 1032],
                            [0, 0, 1]], dtype=np.float32)

    # Coeficientes de distorsión
    dist_coeffs = np.array([-0.35, 0.1, 0.0, 0.0], dtype=np.float32)

    h, w = img.shape[:2]

    # Calcular la nueva matriz de cámara (sin distorsión)
    new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))

    # Corregir la distorsión de la imagen
    dst = cv.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)

    # Recortar la imagen según el ROI (región de interés)
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]

    # -------------------------------------------------------------------------------------------------
    # Resize
    recorte = cv.resize(dst, (768, 576), interpolation=cv.INTER_CUBIC)
    # recorte = cv.resize(img, (768, 576), interpolation=cv.INTER_CUBIC)

    img_final = recorte
    
    return img_final

########################################3
# main

paths = os.listdir(input_path)

for video_path in paths:

    video_path = os.path.join(input_path,video_path)
    # Abrir el video con OpenCV
    cap = cv.VideoCapture(video_path)

    # Inicializar el contador de fotogramas
    frame_number = 0

    while True:
        # Leer un fotograma del video
        ret, frame = cap.read()

        # Si no hay más fotogramas, salir del bucle
        if not ret:
            break
        # Solo guardar el fotograma si es múltiplo del intervalo
        if frame_number % frame_interval == 0 and frame_number >= frame_interval:
            # Generar el nombre del archivo para el fotograma
            frame_filename = os.path.join(out_path, f'{video_path[18:-4]}_{frame_number}.png')

            # preprocesar
            frame = preparar_img(frame)

            # Guardar el fotograma como una imagen PNG
            cv.imwrite(frame_filename, frame)

        # Incrementar el contador de fotogramas
        frame_number += 1

    # Liberar el video y cerrar todas las ventanas
    cap.release()
    print(f'Extracción de fotogramas para el video {video_path} completada.')
