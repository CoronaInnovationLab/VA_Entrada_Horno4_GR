from scipy.spatial.distance import cdist
from sqlalchemy import create_engine, exc, URL
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import cv2 as cv
import datetime
import math
import time
import os

# ROI Deteccion
roi_x1: int = 56
roi_x2: int = 740
roi_y1: int = 175
roi_y2:int = 450
count_line: int = 270

# Sistema posible colisión
mid: int = 370
lim_izq: int = 355 # -15
lim_der: int = 385 # +15
path_alarma: str = '../00_Data/alarmas'
if not os.path.exists(path_alarma):
    os.makedirs(path_alarma)

# match ref
ref_hole = cv.imread('Ref_Hole.png', 0)
wt, ht = ref_hole.shape[::-1]
distance_mm: int = 150
delta_x_ini: int = 70
delta_x_fin: int = delta_x_ini + 50
threshold_match_ref: float = 0.60

# Lista de colores en formato RGB
#           Rojo,               Verde,                     Azul,                Amarillo,               Magenta
colores = {'Taza':(255, 0, 0), 'Lavamanos':(0, 255, 0), 'Onepiece':(0, 0, 255), 'Tanque':(255, 255, 0), 'Pedestal':(255, 0, 255)}


# Parametros conexion SQL
load_dotenv()
# Connection keys 
server = os.getenv("SERVER")
username = os.getenv("USER_SQL")
password = os.getenv("PASSWORD")
database = os.getenv("DATABASE")
tabla = 'entrada_H4_GR'
# Connecting to the sql database
connection_str = "DRIVER={ODBC Driver 17 for SQL Server};SERVER=%s;DATABASE=%s;UID=%s;PWD=%s;Encrypt=no" % (server, database, username, password)
connection_url = URL.create("mssql+pyodbc", query={"odbc_connect": connection_str})


def log(msg:str):
    print(f'[{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {msg}\n')


def draw_detections(image, box, label, confidence):
    label_confianza = label + " " + str(round(confidence * 100, 2)) + "%"
    color = tuple(colores[label])
    
    # Dibujar bounding box
    x1, y1, x2, y2 = box[:4].astype(int)
    cv.rectangle(image, (x1, y1), (x2, y2), color, 2)

    # Dibujar el label_confianza
    text_size, _ = cv.getTextSize(label_confianza, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    text_x, text_y = x1+3, y1-5
    rect_start, rect_end = (text_x - 5, text_y - text_size[1] - 5), (text_x + text_size[0] + 5, text_y + 5)
    cv.rectangle(image, rect_start, rect_end, color, -1)
    cv.putText(
        image,
        label_confianza,
        (text_x, text_y),
        cv.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2,
    )
    # transparency=0.2
    # # Dibujar la mascara
    # overlay = image.copy()
    # mask = (mask * 255).astype(np.uint8)

    # # resize to (768, 576)
    # mask = cv.resize(mask, (768, 576), interpolation=cv.INTER_CUBIC)

    # # Find contours in the binary mask
    # contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # for contour in contours:
    #     # Approximate the contour to reduce the number of points
    #     refined_mask = cv.approxPolyDP(contour, 0.002 * cv.arcLength(contour, True), True)

    #     # Skip invalid contours (e.g., too small)
    #     if len(refined_mask) < 3:
    #         continue

    #     # Effect of "selection"
    #     cv.polylines(overlay, [refined_mask], isClosed=True, color=color, thickness=3)
    #     cv.fillPoly(overlay, [refined_mask], color)  # Draw mask with primary color


    # # efecto de brillo
    # cv.polylines(overlay, [refined_mask], isClosed=True, color=color, thickness=3)

    # # Combinar mascara y la imagen original
    # image = cv.addWeighted(overlay, transparency, image, 1 - transparency, 0)  
    
    return image


def draw_grids(image, debug=False):
    # ROI Deteccion
    cv.line(image, (roi_x1, roi_y1), (roi_x1, roi_y2), (26, 136, 230), 2) 
    cv.line(image, (roi_x2, roi_y1), (roi_x2, roi_y2), (26, 136, 230), 2) 

    # linea de conteo
    cv.line(image, (roi_x1, count_line), (roi_x2, count_line), (26, 55, 230), 2)

    if debug:
        # Malla/cruz para nivelar y centrar la camara manualmente
        # get image shape
        h, w = image.shape[:2]
        # Sistema anti colision
        cv.line(image, (mid, 0), (mid, h), (26, 136, 230), 2)
        cv.line(image, (lim_izq, 0), (lim_izq, h), (26, 55, 230), 2)
        cv.line(image, (lim_der, 0), (lim_der, h), (26, 55, 230), 2)
        # draw crosshair in the center of the image
        # cv.line(image, (w // 2, 0), (w // 2, h), (0, 255, 0), 2)
        # cv.line(image, (0, h // 2), (w, h // 2), (0, 255, 0), 2)

    return image


def show_inventary(frame, inventario):
    '''
    Mustra los objetos contados en la parte superior isquierda de la pantalla,
    junto con el conteo total de objetos.
    '''
    x, y = 40, 50  # Posición inicial para el texto
    font_scale = 0.5
    thickness = 2

    for clase, cantidad in inventario.items():
        # Texto de clase 
        clase_texto = f"{clase}: "
        cv.putText(frame, clase_texto, (int(x), int(y)), cv.FONT_HERSHEY_SIMPLEX, 
                   font_scale, (255, 255, 255), thickness, cv.LINE_AA)
        
        # Texto de cantidad
        cantidad_texto = str(cantidad)
        
        # Calculo de coordenadas para "concatenar" el texto
        text_size = cv.getTextSize(clase_texto, cv.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        cantidad_x = x + text_size[0]
        cv.putText(frame, cantidad_texto, (int(cantidad_x), int(y)), cv.FONT_HERSHEY_SIMPLEX, 
                   font_scale, colores[clase], thickness, cv.LINE_AA)

        # Incrementar la posición x para la siguiente clase
        x += text_size[0] * 1.5


def preparar_img(img):
    '''
    Preprocesado de la imagen, realiza correccion de ojo de pez y resize a 800x600
    '''
    # Convertir la imagen a formato BGR para OpenCV 
    # Solo cuando se trabaja con la camara
    img = cv.cvtColor(img, cv.COLOR_BAYER_BG2BGR)

    # Corregir distorsion
    # Matriz de cámara (suponiendo que ya la tienes)
    camera_matrix = np.array([[2056, 0, 1025],
                            [0, 2064, 1032],
                            [0, 0, 1]], dtype=np.float32)

    # Coeficientes de distorsión (ejemplo)
    dist_coeffs = np.array([-0.35, 0.1, 0.0, 0.0], dtype=np.float32)

    h, w = img.shape[:2]

    # Calcular la nueva matriz de cámara (sin distorsión)
    new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))

    # Corregir la distorsión de la imagen
    dst = cv.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)

    # Recortar la imagen según el ROI (región de interés)
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]

    #-------------------------------------------------------------------------------------------------
    # Resize
    recorte = cv.resize(dst, (768, 576), interpolation=cv.INTER_CUBIC)

    img_final = recorte

    # ROI para evitar detecciones falsas en la parte superior de la imagen
    # Mascara
    blank = np.zeros(img_final.shape[:2], dtype='uint8')
    # Area seleccion ROI 
    cv.rectangle(blank, (roi_x1, roi_y1), (roi_x2, roi_y2), 255, -1)
    mascara = cv.bitwise_and(img_final,img_final,mask=blank)
    cv.imwrite('mask.png', mascara)
    
    return img_final, mascara


class Tracker:
    def __init__(self, max_distance=50, max_lost=5):
        """
        Inicializa el tracker.
        tracks format: {track_id: {'class_name': name, 'center': (x, y), 'bbox': [x1, y1, x2, y2], 'counted': bool}, .., ...}
        inventario format: {class_id: count, .., ...}
        """
        self.next_id = 0
        self.tracks = {}  # Diccionario para almacenar los objetos rastreados
        self.inventario = {'Lavamanos': 0, 'Taza': 0, 'Onepiece': 0, 'Pedestal': 0, 'Tanque': 0}  # Diccionario para almacenar el inventario final del carro
        self.max_distance = max_distance # Distancia máxima para asociar detecciones
        self.max_lost = max_lost # Número máximo de frames sin detección antes de eliminar el objeto rastreado

    def update(self, detections, labels):
        """
        Actualiza el estado del tracker con las nuevas detecciones.
        """
        if len(detections) == 0:
            # Incrementar el contador de "perdidos" para todos los objetos rastreados
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id]['lost'] += 1
                if self.tracks[track_id]['lost'] > self.max_lost:
                    del self.tracks[track_id]  # Eliminar objetos perdidos
            
            if self.tracks == {}:
                return True
            
            return False

        # Calcular los centros de las detecciones
        detection_centers = np.array(list([(d[0] + d[2]) / 2, (d[1] + d[3]) / 2] for d in detections))

        # Obtener los centros de los objetos rastreados
        track_ids = list(self.tracks.keys())
        track_centers = np.array([self.tracks[track_id]['center'] for track_id in track_ids])
        track_class_names = np.array([self.tracks[track_id]['class_name'] for track_id in track_ids])

        # Asociar detecciones con objetos rastreados usando la distancia
        if len(track_centers) > 0:
            distances = cdist(track_centers, detection_centers)
            row_indices, col_indices = np.unravel_index(np.argsort(distances.ravel()), distances.shape)

            assigned_tracks = set()
            assigned_detections = set()

            for row, col in zip(row_indices, col_indices):
                # Si ya se asigno, continuar
                if row in assigned_tracks or col in assigned_detections:
                    continue
                # Si la distancia es mayor que la máxima, continuar
                if distances[row, col] > self.max_distance:
                    continue
                # Si la clase no coincide, continuar
                if track_class_names[row] != labels[col]:
                    continue
                
                # Actualizar el objeto rastreado con la nueva detección
                track_id = track_ids[row]
                self.tracks[track_id]['center'] = detection_centers[col]
                self.tracks[track_id]['bbox'] = detections[col]
                self.tracks[track_id]['class_name'] = labels[col]
                self.tracks[track_id]['lost'] = 0
                assigned_tracks.add(row)
                assigned_detections.add(col)

            # Incrementar el contador de "perdidos" para los objetos no asignados
            for i, track_id in enumerate(track_ids):
                if i not in assigned_tracks:
                    self.tracks[track_id]['lost'] += 1
                    if self.tracks[track_id]['lost'] > self.max_lost:
                        del self.tracks[track_id]
                        
            # Crear nuevos objetos rastreados para las detecciones no asignadas
            for i, detection in enumerate(detections):
                if i not in assigned_detections:
                    self.tracks[self.next_id] = {
                        'center': detection_centers[i],
                        'bbox': detection,
                        'class_name': labels[i],
                        'lost': 0,
                        'counted': False
                    }
                    self.next_id += 1
        else:
            # Si no hay objetos rastreados, crear nuevos para todas las detecciones
            for detection, label in zip(detections, labels):
                self.tracks[self.next_id] = {
                    'center': ((detection[0] + detection[2]) / 2, (detection[1] + detection[3]) / 2),
                    'bbox': detection,
                    'class_name': label,
                    'lost': 0,
                    'counted': False
                }
                self.next_id += 1

        return False

    
    def contar(self, frame, path_clasificacion):
        '''
        Añade los objetos al inventario final unicamente si cruzan la linea de conteo.
        Se guardan las imagenes de los objetos en la carpeta de clasificacion.
        '''
        alarma_choque = False
        # Obtener los centros de los objetos rastreados
        track_ids = list(self.tracks.keys())
        track_centers = np.array([self.tracks[track_id]['center'] for track_id in track_ids])
        track_counted = np.array([self.tracks[track_id]['counted'] for track_id in track_ids])
        track_class_names = np.array([self.tracks[track_id]['class_name'] for track_id in track_ids])
        track_boxes = np.array([self.tracks[track_id]['bbox'] for track_id in track_ids])

        # Obtener los objetos que cruzan la línea de conteo
        for i, center in enumerate(track_centers):
            if center[1] < count_line and not track_counted[i]:
                # Recorte para clasificación
                x1, y1, x2, y2 = track_boxes[i].astype(int)
                crop = frame[y1:y2,x1:x2]
                # Ruta
                class_name = track_class_names[i]
                if not os.path.exists(os.path.join(path_clasificacion, class_name)):
                    os.makedirs(os.path.join(path_clasificacion, class_name))
                ruta = os.path.join(path_clasificacion, class_name, f'{time.strftime("%Y-%m-%d_%H-%M-%S")}.png') 
                # Guardar
                cv.imwrite(ruta, crop)

                # Añadir al inventario
                self.inventario[track_class_names[i]] += 1

                # Marcar como contado
                self.tracks[track_ids[i]]['counted'] = True

                # Verificar choques, solo tazas y onepiece
                piezas_para_malla_dinamica = ['Taza', 'Onepiece']
                if track_class_names[i] in piezas_para_malla_dinamica:
                    if center[0] < mid:
                        # Izquierda
                        if x2 < lim_izq:
                            alarma_choque = True
                    else:
                        # Derecha
                        if x1 > lim_der:
                            alarma_choque = True

                    if alarma_choque:
                        mascara = frame
                        cv.rectangle(mascara, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv.line(mascara, (mid, y1), (mid, y2), (26, 136, 230), 2)
                        cv.line(mascara, (lim_izq, y1), (lim_izq, y2), (26, 55, 230), 2)
                        cv.line(mascara, (lim_der, y1), (lim_der, y2), (26, 55, 230), 2)
                        ruta_alarma = os.path.join(path_alarma, f'{time.strftime("%Y-%m-%d_%H-%M-%S")}.png')
                        cv.imwrite(ruta_alarma, mascara)

        return alarma_choque


def match_ref_get_relacion_pixel_mm(mascara, x1, y1, x2, y2, centro_x):
    # roi segun orientacion de pieza
    roi_top, roi_bottom, mid_crop = match_ref_get_roi(mascara, x1, y1, x2, y2, centro_x)

    # match ref por hueco 
    pos_ref_top, top_finded = match_ref_get_match(roi_top)
    pos_ref_bottom, bottom_finded = match_ref_get_match(roi_bottom)
    
    # distance_pixel si no se encuentra match
    distance_pixel = 56
    
    # if ambos huecos -> calcular distancia
    if top_finded and bottom_finded:
        distance_pixel = math.dist((x1 + delta_x_ini + pos_ref_top[0], y1 + pos_ref_top[1]),
                                   (x1 + delta_x_ini + pos_ref_bottom[0], y1 + mid_crop + pos_ref_bottom[1]))

    relacion_pixel_mm = distance_pixel / distance_mm
    
    return relacion_pixel_mm


def match_ref_get_roi(mascara, x1, y1, x2, y2, centro_x):
    mid_crop = (y2 - y1) // 2
    # Mirando a la izquierda
    if centro_x < mid:
        roi_top = mascara[y1:y1 + mid_crop, x2 - delta_x_ini:x2 - delta_x_fin]
        roi_bottom = mascara[y1 + mid_crop:y2, x2 - delta_x_ini:x2 - delta_x_fin]
    else:
        roi_top = mascara[y1:y1 + mid_crop, x1 + delta_x_ini:x1 + delta_x_fin ]
        roi_bottom = mascara[y1 + mid_crop:y2, x1 + delta_x_ini:x1 + delta_x_fin ]
    
    return roi_top, roi_bottom, mid_crop

def match_ref_get_match(roi):
    
    res_h1 = cv.matchTemplate(roi, ref_hole, cv.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv.minMaxLoc(res_h1)

    pos_ref = None
    finded = False

    if max_val > threshold_match_ref:
        top_left = max_loc
        bottom_right = (top_left[0] + wt, top_left[1] + ht)

        pos_ref = [0, 0]
        pos_ref[0] = int(((bottom_right[0] - top_left[0]) / 2) + top_left[0])
        pos_ref[1] = int(((bottom_right[1] - top_left[1]) / 2) + top_left[1])

        finded = True

    return pos_ref, finded

    
def save_sql(inventario_final: dict, fecha:str, alarma_choque:bool):
    # formato fecha 'YYYY-MM-DD_hh-mm-ss'
    fecha_format = fecha.split('_') 
    fecha_format[1] = fecha_format[1].replace('-',':')
    nombre = ' '.join(fecha_format)#("%Y-%m-%d %H:%M:%S")
    # Añadir fecha del carro - fecha/hora
    inventario_final['Fecha'] = nombre
    inventario_final['Colision'] = 1 if alarma_choque else 0
    df = pd.DataFrame(inventario_final, index=[inventario_final['Fecha']])
    # log(df)
    
    # Connect to DB
    engine = create_engine(connection_url)
    log(f'Guardando análisis en la tabla {tabla}...')
    try:
        # Inserta el DataFrame completo
        df.to_sql(
            name=tabla,  
            con=engine,
            if_exists='append',  # Opciones: 'fail', 'replace', 'append'
            index=False
        )
        log("Datos insertados correctamente.")
    # except exc.IntegrityError:
    #     log('Los datos ya se encontraban en la Base de Datos.')
    # except Exception as e:
    #     log(f"Error al insertar datos: {e}")
    finally:
        engine.dispose()  # Cierra la conexión