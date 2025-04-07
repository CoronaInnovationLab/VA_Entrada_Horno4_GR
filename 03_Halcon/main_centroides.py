# FUNCIONAMIENTO EN FOREVER CON LA SEÑAL DIGITAL DE LA CAMARA
# EXTRAS
# GUARDA INVENTARIO DE CADA CARRO EN DB 
# ANALISIS EN VIVO
# GUARDA VIDEO ANALIZADO
from harvesters.core import Harvester
from sqlalchemy import create_engine, exc, URL
from dotenv import load_dotenv
import halcon as ha
import pandas as pd
import numpy as np
import cv2 as cv
import datetime
import time
import sys
import os

# ******************************************************
# Configuraciones iniciales
# ******************************************************

debug_mode: bool = True
batch_size_inference: int = 1 # este script solo soporta = 1
espera_entre_frames = 0.2
gap: int = 5 # +- pixeles para realizar el recorte de img clasificacion
min_confidence: float  = 0.90

# Evaluar el dispositivo disponible (GPU o CPU)
devices = ha.query_available_dl_devices(["runtime", "runtime"], ["gpu", "cpu"])

if len(devices) == 0:
    raise RuntimeError("No supported device found to continue this example.")

device = devices[0]

# Ruta del modelo
model_path: str  = 'best_model.hdl'

# Rutas
path_videos:str = 'data/videos'
path_clasificacion: str = 'data/clasificacion'
path_temporal: str  = 'data/temporal'

rois_seleccionadas: bool = False
alarma_choque: bool = False

# Línea de conteo
# en 800x600 : 250 (encima de linea amarilla)
count_line: int = 270

# en 800x600 : 110 - 690
# limite izquierdo
malla_izq: int = 106 # 110

# limite derecho
malla_der: int = 662 # 690

# ******************************************************
# Parametros conexion SQL
# ******************************************************
load_dotenv()
# Connection keys 
server = os.getenv("SERVER")
username = os.getenv("USER_SQL")
password = os.getenv("PASSWORD")
database = os.getenv("DATABASE")
tabla = 'entrada_H4_GR'
# Connecting to the sql database
connection_str = "DRIVER={ODBC Driver 18 for SQL Server};SERVER=%s;DATABASE=%s;UID=%s;PWD=%s;Encrypt=no" % (server, database, username, password)
connection_url = URL.create("mssql+pyodbc", query={"odbc_connect": connection_str})

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
    
    return img_final


def get_ia_input(frame, ruta_salida:str = path_temporal):
    '''
    Crea una imagen de las mismas dimenciones pero con la ROI designada, la guarda en el directorio designado
    para leerla con las librerias de halcon
    '''
    # Crear carpeta de salida si no existe
    if not os.path.exists(ruta_salida):
        os.makedirs(ruta_salida)

    output_path = os.path.join(ruta_salida, 'temporal.png')
    cv.imwrite(output_path, frame)

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


def calculate_centroid(c1: int, r1: int, c2: int, r2: int) -> list:
    """
    Calcula centroide segun el bounding box proporcionado
    """
    centroid_x = float((c1 + c2) / 2)
    centroid_y = float((r1 + r2) / 2)

    return [centroid_x, centroid_y]


def print_resultados(frame: np.ndarray, inventario_final: dict, colores: dict):
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


class Pieza:

    def __init__(self, class_id: int, frame, roi):
        global consecutivo
        self.id: int = consecutivo
        self.class_name: str = class_names[class_id]
        self.centroid: list[float] = calculate_centroid(*roi)
        self.color = colores[self.class_name]
        # self.contado: bool = False

        consecutivo += 1
        objetos_actuales.append(self)
        self.show(frame,frame, *roi)

    def contar(self, frame, x:int, y:int, w:int, h:int):
        global alarma_choque
        inventario_final[self.class_name] += 1 
        inventario.append(self)
        objetos_actuales.remove(self)

        # Recorte para clasificación
        crop = frame[y-gap:h+gap,x-gap:w+gap]
        # Ruta
        ruta = os.path.join(path_clasificacion, self.class_name, f'{time.strftime("%Y-%m-%d_%H-%M-%S")}.png') 
        
        # Guardar
        cv.imwrite(ruta, crop)

        # Verificar choques
        if x < malla_izq or (x+w) > malla_der:
            alarma_choque = True

    def update(self, frame, frame_copy, centroid: list, x:int, y:int, w:int, h:int):
        self.centroid = centroid
        self.show(frame, frame_copy, x, y, w, h)

    def show(self, frame, frame_copy, x:int, y:int, w:int, h:int):
        # Conteo de inventario
        if self.centroid[1] < count_line and self not in inventario:  # El objeto cruzo la línea
            self.contar(frame_copy, x, y, w, h)
        else:
            # El objeto está por debajo de la línea
            cv.rectangle(frame, (x, y), (w, h), self.color, 2)


def is_old_object(objeto: Pieza, tracked_centroid: list, tracked_class_id: int) -> bool:
    """
    Compara el centroide del nuevo objeto detectado con el viejo para determinar si es el mismo
    """
    # Recuperar info del objeto viejo
    old = False
    class_name = objeto.class_name
    centroid = objeto.centroid
    tracked_class_name = class_names[tracked_class_id]
    
    # Distancia
    distance = ((tracked_centroid[0] - centroid[0])**2 + (tracked_centroid[1] - centroid[1])**2) ** 0.5
    log(f'tracked_centroid: {tracked_centroid}, centroid: {centroid}, distance: {distance}')
    if (distance < 80) and (tracked_class_name == class_name):
        old = True
        log('old') 
    return old

#############################
# Main
#############################
# Inicializar Harvester
harves = Harvester()

cti_file = "C:/Program Files/MATRIX VISION/mvIMPACT Acquire/bin/x64/mvGENTLProducer.cti"

harves.add_file(cti_file)

# Actualizar la lista de cámaras disponibles
harves.update()

try:
    # Conectar a la primera cámara disponible
    ia = harves.create(0)
    log('Camara conectada')
except Exception as e:
    log('Camara no disponible')
    sys.exit(1)

ia.start()

# Inicializar variable para el video writer
video_writer = None
video_iniciado = False
nombre = ''
salto_fotograma = 10

while True:
    # Capturar una imagen de la cámara
    with ia.fetch() as buffer:
        # # Obtener los datos del buffer
        # component = buffer.payload.components[0]
        # # Convertir los datos a una imagen numpy
        # image = component.data.reshape(component.height, component.width)

        # # Preprocesar la imagen
        # frame = preparar_img(image)
        # cv.imshow('Camera Feed', frame)

        #Preguntamos si no se esta dando la confirmacion de movimiento carro
        if ia.remote_device.node_map.LineStatus.value==False:
            carro_completo = 0
        #Preguntamos si se esta dando la confirmacion de movimiento carro
        if ia.remote_device.node_map.LineStatus.value == True and carro_completo < 3:
            # inicializar videowriter
            if not video_iniciado:
                fourcc = cv.VideoWriter_fourcc(*'h264')# mp4v
                nombre = time.strftime("%Y-%m-%d_%H-%M-%S")
                video_filename =os.path.join(path_videos, f'centroide_{nombre}.mp4')
                video_writer = cv.VideoWriter(video_filename, fourcc, 2, (768, 576))
                video_iniciado = True
                
            # Procesar fotograma cada x     
            if numero_fotograma % salto_fotograma != 0:
                numero_fotograma +=1
                continue

            log(f'fotograma: {numero_fotograma}')
            
            # Obtener los datos del buffer
            component = buffer.payload.components[0]
            # Convertir los datos a una imagen numpy
            image = component.data.reshape(component.height, component.width)

            # Preprocesar la imagen
            frame = preparar_img(image)

            # Inferencia
            boxes = get_boundingbox(frame, model)

            # Terminacion de programa por carro completo
            if len(boxes) == 0:
                carro_completo += 1
                numero_fotograma += 1
                log('Sin detecciones')
                continue

            # Copia de frame para mostrar
            frame_copy = frame.copy()
            
            for bound_box in boxes:
                # ej: box = ['10.5', '20.7', '30', '40', '1']
                r1, r2, c1, c2, clase = map(int, bound_box)
                
                # Calcular centroide
                log(bound_box)
                centroid = calculate_centroid(c1,r1,c2,r2)

                # Comparar
                is_old = False
                for objeto in objetos_actuales:
                    is_old = is_old_object(objeto, centroid, clase)
                    if is_old:
                        # Actualizar posicion del objeto
                        objeto.update(frame, frame_copy, centroid, c1,r1,c2,r2)
                        break
                # Crear nuevo objeto en caso de ser nuevo
                if not is_old and r1 > count_line:
                    nueva_pieza = Pieza(clase, frame, [c1, r1, c2, r2])

            # malla anti choque 
            cv.line(frame, (malla_izq, 150), (malla_izq, 400), (26, 136, 230), 2) 
            cv.line(frame, (malla_der, 150), (malla_der, 400), (26, 136, 230), 2) 

            # Dibujar linea de conteo
            cv.line(frame, (malla_izq, count_line), (malla_der, count_line), (26, 55, 230), 2)

            # Mostrar resultados
            print_resultados(frame, inventario_final, colores)
            
            # mostrar el fotograma
            cv.imshow('track IA Halcon', frame)     

            # Guardar el fotograma en el archivo de video
            video_writer.write(frame)

            numero_fotograma +=1

            time.sleep(espera_entre_frames)

        # Liberar el video y cerrar todas las ventanas
        if carro_completo == 3:
            video_writer.release()
            video_writer = None
            video_iniciado = False
            cv.destroyAllWindows()
            #
            save_sql(inventario_final, nombre)
            # reiniciar variables de inventario
            consecutivo = 0
            numero_fotograma = 0
            objetos_actuales = []
            inventario = []
            inventario_final = {key: 0 for key in class_names}
            alarma_choque = False
            #
            # +1 carro para que no entre al condicional
            carro_completo +=1
            log('Carro Completo')

        k = cv.waitKey(1)
        if k == ord('q'):
            log('Operacion terminada por el usuario.')
            break
        elif k == ord('s'):
            cv.imwrite(os.path.join('data', 'ak.png'), frame)