from scipy.spatial.distance import cdist
import numpy as np
import cv2 as cv
import datetime
import time
import os

# Definición de la malla
malla_izq: int = 106
malla_der: int = 622
count_line: int = 270

# Lista de colores en formato RGB
#           Rojo,       Verde,          Azul,       Amarillo,       Magenta
colores = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]


def log(msg:str):
    print(f'[{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {msg}\n')

def draw_detections(image, box, label, confidence):

    label_confianza = str(int(label)) + " " + str(round(confidence * 100, 2)) + "%"
    color = tuple(colores[int(label)])
    
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
    # malla anti choque 
    cv.line(image, (malla_izq, 150), (malla_izq, 400), (26, 136, 230), 2) 
    cv.line(image, (malla_der, 150), (malla_der, 400), (26, 136, 230), 2) 

    # linea de conteo
    cv.line(image, (malla_izq, count_line), (malla_der, count_line), (26, 55, 230), 2)

    if debug:
        # Malla/cruz para nivelar y centrar la camara manualmente
        # get image shape
        h, w = image.shape[:2]
        # draw crosshair in the center of the image
        cv.line(image, (w // 2, 0), (w // 2, h), (0, 255, 0), 2)
        cv.line(image, (0, h // 2), (w, h // 2), (0, 255, 0), 2)

    return image


def show_inventary(frame, inventario):
    '''
    Mustra los objetos contados en la parte superior isquierda de la pantalla,
    junto con el conteo total de objetos.
    '''
    x, y = 50, 50  # Posición inicial para el texto
    font_scale = 0.5
    thickness = 2

    for clase, cantidad in inventario.items():
        # Texto de clase 
        clase_texto = f"{str(int(clase))}: "
        cv.putText(frame, clase_texto, (int(x), int(y)), cv.FONT_HERSHEY_SIMPLEX, 
                   font_scale, (255, 255, 255), thickness, cv.LINE_AA)
        
        # Texto de cantidad
        cantidad_texto = str(cantidad)
        
        # Calculo de coordenadas para "concatenar" el texto
        text_size = cv.getTextSize(clase_texto, cv.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        cantidad_x = x + text_size[0]
        cv.putText(frame, cantidad_texto, (int(cantidad_x), int(y)), cv.FONT_HERSHEY_SIMPLEX, 
                   font_scale, colores[int(clase)], thickness, cv.LINE_AA)

        # Incrementar la posición x para la siguiente clase
        x += text_size[0] * 2


def preparar_img(img):
    '''
    Preprocesado de la imagen, realiza correccion de ojo de pez y resize a 800x600
    '''

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
    
    return img_final


class Tracker:
    def __init__(self, max_distance=50, max_lost=5):
        """
        Inicializa el tracker.
        tracks format: {track_id: {'class_id': id, 'center': (x, y), 'bbox': [x1, y1, x2, y2], 'counted': bool}, .., ...}
        inventario format: {class_id: count, .., ...}
        """
        self.next_id = 0
        self.tracks = {}  # Diccionario para almacenar los objetos rastreados
        self.inventario = {}  # Diccionario para almacenar el inventario final del carro
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
                return False
            
            return True

        # Calcular los centros de las detecciones
        detection_centers = np.array(list([(d[0] + d[2]) / 2, (d[1] + d[3]) / 2] for d in detections))

        # Obtener los centros de los objetos rastreados
        track_ids = list(self.tracks.keys())
        track_centers = np.array([self.tracks[track_id]['center'] for track_id in track_ids])
        track_class_ids = np.array([self.tracks[track_id]['class_id'] for track_id in track_ids])

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
                if track_class_ids[row] != labels[col]:
                    continue
                
                # Actualizar el objeto rastreado con la nueva detección
                track_id = track_ids[row]
                self.tracks[track_id]['center'] = detection_centers[col]
                self.tracks[track_id]['bbox'] = detections[col]
                self.tracks[track_id]['class_id'] = labels[col]
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
                        'class_id': labels[i],
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
                    'class_id': label,
                    'lost': 0,
                    'counted': False
                }
                self.next_id += 1

    
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
        track_class_ids = np.array([self.tracks[track_id]['class_id'] for track_id in track_ids])
        track_boxes = np.array([self.tracks[track_id]['bbox'] for track_id in track_ids])

        # Obtener los objetos que cruzan la línea de conteo
        for i, center in enumerate(track_centers):
            if center[1] < count_line and not track_counted[i]:
                # Recorte para clasificación
                x1, y1, x2, y2 = track_boxes[i].astype(int)
                crop = frame[y1:y2,x1:x2]
                # Ruta
                class_name = str(int(track_class_ids[i]))
                if not os.path.exists(os.path.join(path_clasificacion, class_name)):
                    os.makedirs(os.path.join(path_clasificacion, class_name))
                ruta = os.path.join(path_clasificacion, class_name, f'{time.strftime("%Y-%m-%d_%H-%M-%S")}.png') 
                # Guardar
                cv.imwrite(ruta, crop)

                # Añadir al inventario
                if track_class_ids[i] in self.inventario:
                    self.inventario[track_class_ids[i]] += 1
                else:
                    self.inventario[track_class_ids[i]] = 1

                # Marcar como contado
                self.tracks[track_ids[i]]['counted'] = True

                # Verificar choques
                if x1 < malla_izq or (x2) > malla_der:
                    alarma_choque = True

        return alarma_choque
    
    
def save_sql(inventario_final: dict, fecha:str):
    global alarma_choque
    # formato fecha 'YYYY-MM-DD hh:mm:ss'
    fecha_format = fecha.split('_') #("%Y-%m-%d %H:%M:%S")
    fecha_format[1] = fecha_format[1].replace('-',':')
    nombre = ' '.join(fecha_format)
    # Añadir fecha del carro - fecha/hora
    inventario_final['fecha'] = nombre
    inventario_final['colision'] = 1 if alarma_choque else 0
    df = pd.DataFrame(inventario_final, index=[inventario_final['fecha']])
    log(df)
    
    # Connect to DB
    engine = create_engine(connection_url)
    print(f'Guardando análisis en la tabla {tabla}...')
    try:
        # Inserta el DataFrame completo
        df.to_sql(
            name=tabla,  
            con=engine,
            if_exists='append',  # Opciones: 'fail', 'replace', 'append'
            index=False
        )
        print("Datos insertados correctamente.")
    except exc.IntegrityError:
        log('Los datos ya se encontraban en la Base de Datos.')
    except Exception as e:
        log(f"Error al insertar datos: {e}")
    finally:
        engine.dispose()  # Cierra la conexión