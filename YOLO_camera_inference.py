from utils import draw_detections, draw_grids, show_inventary, preparar_img, Tracker, log
from harvesters.core import Harvester
from sqlalchemy import create_engine, exc, URL
from dotenv import load_dotenv
from ultralytics import YOLO
import cv2 as cv
import torch
import sys
import os

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
# Configuraciones iniciales
# ******************************************************
model_path: str = "runs/train/Entrada_H4_YOLO_V1/weights/best.pt" 
crop_path: str = "data/crops"
video_name: str = "crudo_2025-02-28_13-01-09.mp4"

video_path: str = os.path.join('data/videos', video_name)
output_path: str = 'data/videos/inferencias'
save_video: bool = True

# ******************************************************
# Configuracion camara y guardado de video
# ******************************************************
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
if not os.path.exists(output_path):
    os.makedirs(output_path)

# ******************************************************
# Parametros del modelo
# ******************************************************
model = YOLO(model_path)
class_names = model.names
min_confidence: float  = 0.90
min_iou: float = 0.45
alarma_choque = False

device = 0 if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Inicializar tracker
tracker = Tracker()

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
                output_path = os.path.join(output_path, f"{video_name[:-4]}_inference.mp4")
                fps = 30
                width = int(768)
                height = int(576)
                fourcc = cv.VideoWriter_fourcc(*'mp4v')
                out = cv.VideoWriter(output_path, fourcc, fps, (width, height))

            # Obtener los datos del buffer
            component = buffer.payload.components[0]
            # Convertir los datos a una imagen numpy
            image = component.data.reshape(component.height, component.width)

            # Preprocesar la imagen
            frame = preparar_img(frame)

            # Inferir
            results = model.predict(source=frame, conf=min_confidence, iou=min_iou, device=device)
            
            # Procesar resultados
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                labels = result.boxes.cls.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()

                # Actualizar el inventario
                fin = tracker.update(boxes, labels)
                print(f"Total objects: {len(tracker.tracks)}")

                alarma_choque = tracker.contar(frame, crop_path) or alarma_choque

                # mostrar el inventario
                show_inventary(frame, tracker.inventario)

                # Dibujar detecciones
                for box, label, confidence in zip(boxes, labels, confidences):
                    frame = draw_detections(frame, box, label, confidence)

            if fin:
                video_writer.release()
                video_writer = None
                video_iniciado = False
                cv.destroyAllWindows()
                #
                save_sql(inventario_final, nombre)

                # reinicio de vars
                break

            #mostrar bordes
            frame = draw_grids(frame)

            # Ver en tiempo real
            cv.imshow("YOLOv8 Inference", frame)
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                break
                
            print(f"Alarma de choque: {alarma_choque}")