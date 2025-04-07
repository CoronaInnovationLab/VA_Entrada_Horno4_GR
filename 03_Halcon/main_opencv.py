# FUNCIONAMIENTO EN FOREVER CON LA SEÑAL DIGITAL DE LA CAMARA
# EXTRAS
# GUARDA INVENTARIO DE CADA CARRO EN DB 
# ANALISIS EN VIVO
# GUARDA VIDEO ANALIZADO
from harvesters.core import Harvester
from sqlalchemy import create_engine, exc, URL
import halcon as ha
import pandas as pd
import numpy as np
import cv2 as cv
import datetime
import time
import os
import sys

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

##### FUNCIONAMIENTO FOREVER
# Ruta
path_videos:str = 'data/videos'
path_clasificacion: str = 'data/clasificacion'
path_temporal: str  = 'data/temporal'

rois_seleccionadas: bool = False
alarma_choque: bool = False

# Línea de conteo
# en 800x600 : 250 (encima de linea amarilla)
count_line: int = 270#280

# Linea check Out
checkout_line: int = count_line - 40 #mejor resultado = 40

# en 800x600 : 110 - 690
# limite izquierdo
malla_izq: int = 106 # 110

# limite derecho
malla_der: int = 662 # 690

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

    # ha._numpy_image_type_to_halcon() #### probar esto
    #
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


def save_sql(inventario_final: dict, fecha:str):
    # formato fecha 'YYYY-MM-DD hh:mm:ss'
    fecha_format = fecha.split('_') #("%Y-%m-%d %H:%M:%S")
    fecha_format[1] = fecha_format[1].replace('-',':')
    nombre = ' '.join(fecha_format)
    # Añadir nombre del carro - fecha/hora
    inventario_final['fecha'] = nombre
    df = pd.DataFrame(inventario_final, index=[inventario_final['fecha']])
    log(df)
    # Connection keys 
    server = os.environ.get("SERVER")
    username = os.environ.get("USER_SQL")
    password = os.environ.get("PASSWORD")
    database = os.environ.get("DATABASE")
    tabla = 'entrada_H4_GR'

    # Connecting to the sql database
    connection_str = "DRIVER={ODBC Driver 18 for SQL Server};SERVER=%s;DATABASE=%s;UID=%s;PWD=%s;Encrypt=no" % (server, database, username, password)
    connection_url = URL.create("mssql+pyodbc", query={"odbc_connect": connection_str})
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

        # Verificar choques
        if x < malla_izq or (x+w) > malla_der:
            alarma_choque = True

    def __str__(self) -> str:
        return f'objeto: {self.id} con clase {self.class_name}. '


#############################
# Main
#############################
# ----------------------------------------------------------------------------------------------------------
# Conexión camara y toma de imagen
# ----------------------------------------------------------------------------------------------------------
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

while True:
    # Capturar una imagen de la cámara
    with ia.fetch() as buffer:
        # Obtener los datos del buffer
        component = buffer.payload.components[0]
        # Convertir los datos a una imagen numpy
        image = component.data.reshape(component.height, component.width)

        # Preprocesar la imagen
        frame = preparar_img(image)
        cv.imshow('Camera Feed', frame)

        #Preguntamos si no se esta dando la confirmacion de movimiento carro
        if ia.remote_device.node_map.LineStatus.value==False:
            carro_completo = 0
        #Preguntamos si se esta dando la confirmacion de movimiento carro
        if ia.remote_device.node_map.LineStatus.value == True and carro_completo < 3:
            # inicializar videowriter
            if not video_iniciado:
                fourcc = cv.VideoWriter_fourcc(*'h264')# mp4v
                nombre = time.strftime("%Y-%m-%d_%H-%M-%S")
                video_filename =os.path.join(path_videos, f'{nombre}.mp4')
                video_writer = cv.VideoWriter(video_filename, fourcc, 8, (768, 576))
                video_iniciado = True

            # Volver a seleccionar ROI's cuando no haya objetos que seguir
            if len(objetos_actuales) == 0:
                rois_seleccionadas = False
            
            if not rois_seleccionadas:
                #
                # Inferencia
                boxes = get_boundingbox(frame, model)

                # Terminacion de programa por carro completo
                if len(boxes) == 0:
                    carro_completo += 1

                else:
                    for bound_box in boxes:
                        # ej: box = ['10.5', '20.7', '30', '40', '1']
                        r1, r2, c1, c2, clase = map(int, bound_box)
                        nueva_pieza = Pieza(clase, frame, (c1, r1, c2-c1, r2-r1))
                
                rois_seleccionadas = True

            # Recorrer trackers activos
            frame_copy = frame.copy()
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
                    objeto.check_out(frame_copy, x, y, w, h)
    
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

            # Guardar el fotograma en el archivo de video
            if video_writer is not None:
                video_writer.write(frame)

            time.sleep(espera_entre_frames)
        
        # Liberar el video y cerrar todas las ventanas
        if carro_completo == 3:
            log(texto)
            video_writer.release()
            video_writer = None
            video_iniciado = False
            cv.destroyAllWindows()
            #
            save_sql(inventario_final, nombre)
            # reiniciar variables de inventario
            consecutivo = 0
            objetos_actuales = []
            inventario = []
            inventario_final = {key: 0 for key in class_names}
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