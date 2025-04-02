from utils import draw_detections, draw_grids, show_inventary, preparar_img, Tracker, save_sql, log
from harvesters.core import Harvester
from ultralytics import YOLO
import numpy as np
import cv2 as cv
import torch
import time
import sys
import os

# ******************************************************
# Configuraciones iniciales
# ******************************************************
model_path: str = "runs/train/Entrada_H4_YOLO_V1/weights/best.pt" 
crop_path: str = "data/crops"

video_path: str = 'data/videos'
output_path: str = 'data/videos/inferencias'
video_iniciado: bool = False
carro_completo: bool = False

if not os.path.exists(output_path):
    os.makedirs(output_path)

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

# ******************************************************
# Parametros del modelo
# ******************************************************
model = YOLO(model_path)
class_names = model.names
min_confidence: float  = 0.90
min_iou: float = 0.45

device = 0 if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

while True:
    # Capturar una imagen de la cámara
    with ia.fetch() as buffer:
        #Preguntamos si no se esta dando la confirmacion de movimiento carro
        if ia.remote_device.node_map.LineStatus.value==False:
            carro_completo = False
        #Preguntamos si se esta dando la confirmacion de movimiento carro
        if ia.remote_device.node_map.LineStatus.value == True and not carro_completo:
            # Obtener los datos del buffer
            component = buffer.payload.components[0]

            # Convertir los datos a una imagen numpy
            image = component.data.reshape(component.height, component.width)

            # inicializar variables
            if not video_iniciado:

                alarma_choque = False
                tracker = Tracker()
                video_name: str = time.strftime("%Y-%m-%d_%H-%M-%S")

                output_inference_path = os.path.join(output_path, f"{video_name}_inference.mp4")
                output_raw_path = os.path.join(video_path, f"{video_name}_raw.mp4")

                fps = 25
                fourcc = cv.VideoWriter_fourcc(*'mp4v')
                fourccraw = cv.VideoWriter_fourcc(*'h264')

                inference = cv.VideoWriter(output_inference_path, fourcc, fps, (768, 576))
                raw = cv.VideoWriter(output_raw_path, fourccraw, fps, (component.width, component.height))

                video_iniciado = True

            # Preprocesar la imagen
            frame, roi = preparar_img(image)

            # Inferir
            results = model.predict(source=roi, conf=min_confidence, iou=min_iou, device=device)
            
            # Procesar resultados
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                labels = result.boxes.cls.cpu().numpy()
                #convertir labels de float a su equivalente en string
                labels = np.array([class_names[int(label)] for label in labels])
                confidences = result.boxes.conf.cpu().numpy()

                # Actualizar el inventario
                carro_completo = tracker.update(boxes, labels)
                
                alarma_choque = tracker.contar(frame, crop_path) or alarma_choque

                # mostrar el inventario
                show_inventary(frame, tracker.inventario)

                # Dibujar detecciones
                for box, label, confidence in zip(boxes, labels, confidences):
                    frame = draw_detections(frame, box, label, confidence)

            #mostrar bordes
            frame = draw_grids(frame)

            # Guardar fotogramas
            raw.write(image)
            inference.write(frame)

            # Ver en tiempo real
            cv.imshow("YOLOv8 Inference", frame)
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            if carro_completo:
                inference.release()
                inference = None

                raw.release()
                raw = None
                
                video_iniciado = False
                cv.destroyAllWindows()
                #
                log(f"Alarma de choque: {alarma_choque}")
                # save_sql(tracker.inventario, video_name, alarma_choque)
                log(tracker.inventario)

                del tracker