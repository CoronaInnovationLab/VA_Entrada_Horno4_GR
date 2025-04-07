import cv2
import os
import time
import numpy as np

# ------------------------------------------------------------------------------------------------ 
# variables
# ------------------------------------------------------------------------------------------------
consecutivo: int = 0
objetos_actuales: list[object] = []
inventario:list[object] = []
inventario_final: dict = dict(
    [
        ('Lavamanos', 0),
        ('Tazas', 0),
        ('One_Pieces', 0),
        ('Pedestales', 0),
        ('Tanques', 0)
    ]
)

clasificacion: dict = dict(
    [
        (0, 'Lavamanos'),
        (1, 'Tazas'),
        (2, 'One_Pieces'),
        (3, 'Pedestales'),
        (4, 'Tanques')
    ]
)

# Ruta del video
path: str = os.path.join('data/videos', '2024-10-29_12-13-26.mp4')

path_clasificacion: str = 'data/clasificacion'

# Abrir el video con OpenCV
video = cv2.VideoCapture(path)

# Verificar si el video se abrió correctamente
if not video.isOpened():
    print("Error al abrir el video")
    exit()

rois_seleccionadas = False

count_line = 230 # 270 es bien - 70 para out

checkout_line_y = count_line - 40 

# ------------------------------------------------------------------------------------------------
# funciones y objetos
# ------------------------------------------------------------------------------------------------
class Pieza:

    def __init__(self, class_id: int, frame, roi):
        global consecutivo
        self.id: int = consecutivo
        self.class_id: int = class_id
        self.tracker = cv2.TrackerCSRT_create()

        consecutivo += 1
        self.tracker.init(frame, roi)
        objetos_actuales.append(self)

    def contar(self, frame, x:int, y:int, w:int, h:int):
        inventario_final[clasificacion[self.class_id]] += 1 
        inventario.append(self)
        # Recorte para clasificación
        print(x,y,w,h)
        crop = frame[y:y+h,x:x+w]
        # Ruta
        ruta = os.path.join(path_clasificacion, clasificacion[self.class_id], f'{time.strftime("%Y-%m-%d_%H-%M-%S")}.png') 
        # Guardar
        cv2.imwrite(ruta, crop)

    def check_out(self):
        objetos_actuales.remove(self)

    def __str__(self) -> str:
        return f'objeto: {self.id} con clase {clasificacion[self.class_id]}. '


def print_resultados() -> str:
    texto = ''
    for clase, cantidad in inventario_final.items():
        texto += str(clase) + ': ' + str(cantidad) + '.   '
    return texto

def selec_roi(frame) -> tuple:
    # Mascara
    blank = np.zeros(frame.shape[:2], dtype='uint8')
    # Area seleccion ROI
    cv2.rectangle(blank, (50, count_line+10), (750, count_line + 250), 255, -1)
    masked = cv2.bitwise_and(frame,frame,mask=blank)
    # Seleccion manual de ROI
    roi = cv2.selectROI("Selecciona los objetos a rastrear y presiona Enter", masked, fromCenter=False, showCrosshair=True)
    return roi

# ------------------------------------------------------------------------------------------------
# loop principal
# ------------------------------------------------------------------------------------------------
while True:
    # Leer un fotograma del video
    success, frame = video.read()

    # Si no hay más fotogramas, salir del bucle
    if not success:
        break

    # Volver a seleccionar ROI's cuando no haya objetos que seguir
    if len(objetos_actuales) == 0:
        rois_seleccionadas = False
    
    if not rois_seleccionadas:
        while True:
            # Seleccionar ROI's iniciales
            roi = selec_roi(frame)
            if roi == (0, 0, 0, 0):  # Salir si no se selecciona un área válida
                break
            # Definir clase 
            clase = input('Clase del objeto: \n')
            # Crear objeto
            nueva_pieza = Pieza(int(clase), frame, roi)

        # Cambiar el estado para evitar volver a seleccionar ROIs
        rois_seleccionadas = True
        cv2.destroyAllWindows()

    # Recorrer trackers activos
    for objeto in objetos_actuales:
        # Verificar restreo de objeto
        success, box = objeto.tracker.update(frame)
        (x, y, w, h) = [int(v) for v in box]
        
        # Conteo de inventario y recorte de ROI
        # primero el conteo para evitar guardar imgs con boxes
        if y < count_line and objeto not in inventario:  # El objeto cruzo la línea
            objeto.contar(frame, x, y, w, h)

        # Dibujar y procesar check_out
        if y > checkout_line_y:  # El objeto está por debajo de la línea
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            objeto.check_out()


    # Dibujar la línea de check-out
    cv2.line(frame, (50, checkout_line_y), (750, checkout_line_y), (255, 0, 0), 2)

    # Dibujar linea de conteo
    cv2.line(frame, (50, count_line), (750, count_line), (0, 0, 255), 2)

    # Mostrar resultados
    texto = print_resultados()
    cv2.putText(frame, texto, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (255,0,0), 2, cv2.LINE_AA)
    
    # mostrar el fotograma
    cv2.imshow('track manual', frame)
    
    # Control programa
    key = cv2.waitKey(1) & 0xFF
    # Salir del loop si se presiona 'q'
    if key == ord('q'):
        break        

# Liberar el video y cerrar todas las ventanas
video.release()
cv2.destroyAllWindows()