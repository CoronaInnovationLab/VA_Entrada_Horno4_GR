# LEE Y ANALIZA CON LA IA UN VIDEO YA GRABADO

import halcon as ha
import numpy as np
import cv2 as cv
import datetime
import time
import os

# ******************************************************
# Configuraciones iniciales
# ******************************************************

debug_mode: bool = True
batch_size_inference: int = 1 # este script solo soporta = 1
espera_entre_frames = 0.1
gap: int = 5 # +- pixeles para realizar el recorte de img clasificacion
min_confidence: float  = 0.90

# Evaluar el dispositivo disponible (GPU o CPU)
devices = ha.query_available_dl_devices(["runtime", "runtime"], ["gpu", "cpu"])

if len(devices) == 0:
    raise RuntimeError("No supported device found to continue this example.")

device = devices[0]

# Ruta del modelo
model_path: str  = 'best_model.hdl'

##### FUNCIONAMIENTO CON VIDEO
# Ruta
path_videos:str = 'data/videos'
path: str = os.path.join('data', 'crudo_2025-01-10_13-09-55.mp4')
path_clasificacion: str = 'data/clasificacion'
path_temporal: str  = 'data/temporal'

# Abrir el video con OpenCV
video = cv.VideoCapture(path)

# Verificar si el video se abrió correctamente
if not video.isOpened():
    print("Error al abrir el video")
    exit()

rois_seleccionadas: bool = False

# Línea de conteo
count_line: int = 270

# Linea check Out
checkout_line: int = count_line - 40

# limite izquierdo
malla_izq: int = 106

# limite derecho
malla_der: int = 662

# ******************************************************
# Parametros del modelo
# ******************************************************

# Leer el modelo entrenado
model = ha.read_dl_model(model_path)

# Obtener nombres de clases
class_names = ha.get_dl_model_param(model, "class_names")

# Configurar el modelo
ha.set_dl_model_param(model, "batch_size", batch_size_inference)
ha.set_dl_model_param(model, "device", device)

# Inicializar variables 
consecutivo: int = 0
objetos_actuales: list[object] = []
inventario:list[object] = []
inventario_final: dict = {key: 0 for key in class_names}
# Lista de colores en formato RGB
colores = {
    'taza': (255, 0, 0),    # Rojo
    'onepiece': (0, 255, 0),    # Verde
    'lavamanos': (0, 0, 255),    # Azul
    'tanque': (255, 255, 0),  # Amarillo
    'pedestal': (255, 0, 255),  # Magenta
    # 'Plafon': (0, 255, 255)   # Cian
}

#############################
# Funciones  y objetos personalizados
#############################

def log(msg:str):
    if debug_mode:
        print(f'[{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {msg}\n')


def gen_dl_sample(batch:ha.HObject) -> ha.HHandle:
    '''
    Generar DL_Sample en un formato "halcon-friendly":
    '''
    sample = [ha.from_python_dict({'image': image}) for image in batch]
    sample = tuple(sample)

    return sample


def preparar_img(img):
    '''
    Preprocesado de la imagen, realiza correccion de ojo de pez y resize a 800x600
    '''

    # Convertir la imagen a formato BGR para OpenCV
    # img = cv.cvtColor(img, cv.COLOR_BAYER_BG2BGR)

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


def get_ia_input(frame, ruta_salida:str = path_temporal):
    '''
    Crea una imagen de las mismas dimenciones pero con la ROI designada, la guarda en el directorio designado
    para leerla con las librerias de halcon
    '''
    # Crear carpeta de salida si no existe
    if not os.path.exists(ruta_salida):
        os.makedirs(ruta_salida)

    # ROI
    # Mascara
    blank = np.zeros(frame.shape[:2], dtype='uint8')
    # Area seleccion ROI # mejor resultado borte inferior = (250, 270) en 800x600
    # en size halcon best = 240
    cv.rectangle(blank, (50, count_line + 15), (735, count_line + 240), 255, -1)
    masked = cv.bitwise_and(frame,frame,mask=blank)

    if debug_mode:
        cv.imshow('IA input', masked)

    # Guardar la imagen en la carpeta de salida
    output_path = os.path.join(ruta_salida, 'temporal.png')
    cv.imwrite(output_path, masked)

    # leer imagen con halcon
    image_batch = ha.read_image(output_path)
    
    # preprocesar halcon, img byte a real
    image_batch = ha.convert_image_type(image_batch,'real')
    log(image_batch)

    # borrar temporal de la carpeta de salida
    os.remove(output_path)

    return image_batch

def get_boundingbox(frame, model) -> list[list]:
    '''
    Realiza la inferencia.\n
    :return boxes: lista de listas con el box y clase de cada objeto detectado ej:
    [row1, row2, col1, col2, class_id]
    [['10.5', '20.7', '30', '40', '1'], ['10.5', '20.7', '30', '40', '2']] 
    '''
    # preprocesar frame, transformar a halcon image batch
    image_batch = get_ia_input(frame)

    # Crear muestras de DL
    dl_samples = gen_dl_sample(image_batch)
    
    # Aplicar modelo
    dl_results = ha.apply_dl_model(model, dl_samples,[])

    # Convertir a diccionarios para facilidad de uso
    dl_result = ha.as_python_dict(dl_results[0])
    log(dl_result)
    
    # Obtener bounding boxes
    inference_items = ["bbox_row1","bbox_row2","bbox_col1","bbox_col2","bbox_class_id"]
    detected_objs = len(dl_result["bbox_row1"])

    boxes = [[dl_result[item][i] for item in inference_items] for i in range(detected_objs) if dl_result['bbox_confidence'][i] > min_confidence]

    return boxes


# def print_resultados() -> str:
#     texto = ''
#     for clase, cantidad in inventario_final.items():
#         texto += str(clase) + ': ' + str(cantidad) + '.   '
#     return texto

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


class Pieza:

    def __init__(self, class_id: int, frame, roi):
        global consecutivo
        self.id: int = consecutivo
        self.class_name: str = class_names[class_id]
        self.tracker = cv.TrackerCSRT_create()
        self.color = colores[self.class_name]

        consecutivo += 1
        self.tracker.init(frame, roi)
        objetos_actuales.append(self)

    def contar(self):
        inventario_final[self.class_name] += 1 
        inventario.append(self)


    def check_out(self, frame, x:int, y:int, w:int, h:int):
        global alarma_choque
        objetos_actuales.remove(self)

        # Recorte para clasificación
        crop = frame[y-gap:y+h+gap,x-gap:x+w+gap]
        # Ruta
        ruta = os.path.join(path_clasificacion, self.class_name, f'{time.strftime("%Y-%m-%d_%H-%M-%S")}.png') 

        # Guardar
        cv.imwrite(ruta, crop)

    def __str__(self) -> str:
        return f'objeto: {self.id} con clase {self.class_name}. '


#############################
# Main
#############################
carro_completo = 0
# Inicializar variable para el video writer
fourcc = cv.VideoWriter_fourcc(*'h264')# mp4v
nombre = time.strftime("%Y-%m-%d_%H-%M-%S")
video_filename =os.path.join(path_videos, f'cv_tracker_{nombre}.mp4')
video_writer = cv.VideoWriter(video_filename, fourcc, 8, (768, 576))

while True:
    # Leer un fotograma del video
    success, frame = video.read()

    # Si no hay más fotogramas, salir del bucle
    if not success:
        break

    frame = preparar_img(frame)

    # Volver a seleccionar ROI's cuando no haya objetos que seguir
    if len(objetos_actuales) == 0:
        rois_seleccionadas = False
    
    if not rois_seleccionadas:
        #
        # Inferencia
        boxes = get_boundingbox(frame, model)

        # Cambiar tamaño de la imagen a 768x576
        frame = cv.resize(frame, (768, 576), interpolation=cv.INTER_CUBIC)

        # terminacion de programa
        if len(boxes) == 0:
            carro_completo += 1
            if carro_completo > 3:
                break

        else:
            for bound_box in boxes:
                # ej: box = ['10.5', '20.7', '30', '40', '1']
                r1, r2, c1, c2, clase = map(int, bound_box)
                nueva_pieza = Pieza(clase, frame, (c1, r1, c2-c1, r2-r1))
        
        rois_seleccionadas = True

    # Recorrer trackers activos
    for objeto in objetos_actuales:
        # Verificar restreo de objeto
        success, box = objeto.tracker.update(frame)
        (x, y, w, h) = [int(v) for v in box]
        
        # Conteo de inventario y recorte de ROI
        if y < count_line and objeto not in inventario:  # El objeto cruzo la línea
            objeto.contar()

        # Dibujar y procesar check_out
        if y > checkout_line:  # El objeto está por debajo de la línea
            cv.rectangle(frame, (x, y), (x + w, y + h), objeto.color, 2)
        else:
            objeto.check_out(frame, x, y, w, h)

   # malla anti choque 
    cv.line(frame, (malla_izq, 150), (malla_izq, 400), (26, 136, 230), 2) 
    cv.line(frame, (malla_der, 150), (malla_der, 400), (26, 136, 230), 2) 

    # Dibujar la línea de check-out
    cv.line(frame, (malla_izq, checkout_line), (malla_der, checkout_line), (26, 100, 230), 2)

    # Dibujar linea de conteo
    cv.line(frame, (malla_izq, count_line), (malla_der, count_line), (26, 55, 230), 2)

    # Mostrar resultados
    texto = print_resultados(frame, inventario_final, colores)
    
    # mostrar el fotograma
    cv.imshow('track IA Halcon', frame)

    video_writer.write(frame)
    
    # Control programa
    key = cv.waitKey(1) & 0xFF
    # Salir del loop si se presiona 'q'
    if key == ord('q'):
        break        

    time.sleep(espera_entre_frames)

# Liberar el video y cerrar todas las ventanas
print(texto)
video.release()
video_writer.release()
cv.destroyAllWindows()