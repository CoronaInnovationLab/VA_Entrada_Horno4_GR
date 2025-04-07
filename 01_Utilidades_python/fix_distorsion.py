import cv2 as cv
import numpy as np
import os

def fix(image_paths):
    filenames = os.listdir(image_paths)

    # Cargar todas las imágenes a partir de las rutas proporcionadas
    images = [cv.imread(image_paths+'/'+img) for img in filenames]

    # Verificar si las imágenes se cargaron correctamente
    if not images:
        print("No se proporcionaron imágenes.")
        return None

    # Matriz de cámara (suponiendo que ya la tienes)
    camera_matrix = np.array([[2056, 0, 1025],
                            [0, 2064, 1032],
                            [0, 0, 1]], dtype=np.float32)

    # Coeficientes de distorsión (ejemplo)
    dist_coeffs = np.array([-0.35, 0.1, 0.0, 0.0], dtype=np.float32)
    
    # os.makedirs(image_paths+r'\r1\\')
    # os.makedirs(image_paths+r'\r2\\')

    for num, img in enumerate(images):
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
        cv.imwrite(image_paths + r'\r1\\' +str(num) + '.png', dst)

        # Undistort with Remapping
        mapx, mapy = cv.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, new_camera_matrix, (w,h), 5)
        dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

        # crop the image
        x, y, w, h = roi

        dst = dst[y:y+h, x:x+w]
        dst = cv.resize(dst, (800,600), interpolation=cv.INTER_CUBIC)
        cv.imwrite(image_paths + r'\r2\\' + str(num) + '.png', dst)

    return None

if __name__ == "__main__":
    ruta = r'C:\Users\fguerrerot\Documents\Proyecto Entrada Horno\Fotos\resto'

    fix(ruta)