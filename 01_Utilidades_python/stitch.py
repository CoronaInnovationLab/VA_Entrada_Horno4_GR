import cv2 as cv
import os
import numpy as np

def preparar_img(img):
    """
    Aplica ecualización de histograma para mejorar el contraste y uniformizar la iluminación.
    Funciona en imágenes en escala de grises y en imágenes BGR.
    """
    # # Convertir a YUV (luminancia y crominancia)
    # img_yuv = cv.cvtColor(img, cv.COLOR_BGR2YUV)
    
    # # Aplicar la ecualización solo en el canal Y (luminancia)
    # img_yuv[:, :, 0] = cv.equalizeHist(img_yuv[:, :, 0])
    
    # # Convertir de nuevo a BGR
    # img = cv.cvtColor(img_yuv, cv.COLOR_YUV2BGR)

    # Escala de grises
    # img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # # Ecualizar brillo
    # img = cv.equalizeHist(img)

    # Recortar
    img = img[:, 450:1500]

    # rotar
    img = cv.rotate(img,cv.ROTATE_90_COUNTERCLOCKWISE)
    
    return img

def stitch_images(image_paths):
    
    filenames = os.listdir(image_paths)

    # Cargar todas las imágenes a partir de las rutas proporcionadas
    images = [preparar_img(cv.imread(image_paths+'/'+img)) for img in filenames]

    # Verificar si las imágenes se cargaron correctamente
    if not images:
        print("No se proporcionaron imágenes.")
        return None
    
    # Crear el objeto de OpenCV para la costura
    stitcher = cv.Stitcher_create()

    # Procesar la costura de imágenes
    (status, stitched_image) = stitcher.stitch(images)
    # Ajustar los parámetros de costura para imágenes verticales
    # stitcher.setPanoConfidenceThresh(0.8)  
    # stitcher.setWaveCorrection(True)  


    # Verificar si la costura fue exitosa
    if status == cv.Stitcher_OK:
        print("Costura completada.")
        # Mostrar la imagen resultante
        final = cv.resize(stitched_image, (500,400), interpolation=cv.INTER_CUBIC)
        mostrar_resultado(final)
        
    else:
        print("La costura falló, error código:", status)
        # 1: Error si no se encontraron suficientes imágenes.
        # 2: Error de estimación de cámaras.
        # 3: Error al ajustar la homografía.
            # Las imágenes no tienen suficiente solapamiento.
            # No se encontraron suficientes características comunes entre las imágenes.
            # Las imágenes tienen perspectivas demasiado diferentes o distorsiones significativas.

    return None

def mostrar_resultado(imagen_resultante):
    if imagen_resultante is not None:
        # Mostrar la imagen resultante
        cv.imshow('Imagen Final', imagen_resultante)
        cv.waitKey(0)
        cv.destroyAllWindows()
    else:
        print("No se pudo mostrar la imagen resultante.")

if __name__ == "__main__":

    images_path = r'C:\Users\fguerrerot\Documents\Proyecto Entrada Horno\Fotos\Entrada\carro0\r2'
    
    # Realizar la costura de imágenes
    stitched_image = stitch_images(images_path)