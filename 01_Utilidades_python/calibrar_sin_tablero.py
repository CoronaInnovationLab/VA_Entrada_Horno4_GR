import cv2 as cv
import numpy as np
import os

def calibrar():
    # Cargar la imagen con distorsión de ojo de pez
    img = cv.imread(r'C:\Users\fguerrerot\Documents\Proyecto Entrada Horno\Fotos\Entrada\carro0\imagen_28-07_1722187671.jpeg')

    # Matriz de cámara (suponiendo que ya la tienes)
    camera_matrix = np.array([[2056, 0, 1025],
                            [0, 2064, 1032],
                            [0, 0, 1]], dtype=np.float32)

    # Coeficientes de distorsión (ejemplo)
    dist_coeffs = np.array([-0.35, 0.1, 0.0, 0.0], dtype=np.float32)

    # Obtener el tamaño de la imagen
    h, w = img.shape[:2]
    # print(h,w)

    # Calcular la nueva matriz de cámara (sin distorsión)
    new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))

    # Corregir la distorsión de la imagen
    dst = cv.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)

    # Recortar la imagen según el ROI (región de interés)
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    r1 = cv.resize(dst, (500,400), interpolation=cv.INTER_CUBIC)
    # cv.imwrite('1.png', dst)

    # Undistort with Remapping
    mapx, mapy = cv.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, new_camera_matrix, (w,h), 5)
    dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

    # crop the image
    x, y, w, h = roi

    dst = dst[y:y+h, x:x+w]
    # cv.imwrite('2.png', dst)
    r2 = cv.resize(dst, (500,400), interpolation=cv.INTER_CUBIC)

    original = cv.resize(img, (500,400))

    # Mostrar y guardar la imagen corregida
    cv.imshow('r1', r1)
    cv.imshow('r2', r2)
    cv.imshow('original', original)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return new_camera_matrix