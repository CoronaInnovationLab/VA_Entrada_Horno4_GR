import halcon as h
import numpy as np
import cv2 as cv
import datetime
import os

global debug_mode

# ******************************************************
# Configuraciones iniciales
# ******************************************************

debug_mode = False#True

# Evaluar el dispositivo disponible (GPU o CPU)
devices = h.query_available_dl_devices(["runtime", "runtime"], ["gpu", "cpu"])

if len(devices) == 0:
    raise RuntimeError("No supported device found to continue this example.")

device = devices[0]

# Ruta del modelo
model_path = 'best_model.hdl'

# Ruta de imágenes de prueba
image_dir = 'img_prueba_00'
preproces_dir = 'preprocesadas'
batch_size_inference = 3

# Línea de conteo
count_line = 260

# ******************************************************
# Parametros del modelo
# ******************************************************

# Leer el modelo entrenado
model = h.read_dl_model(model_path)

# Obtener nombres y clases
dataset_info = {
    "class_names": h.get_dl_model_param(model, "class_names"),
    "class_ids": h.get_dl_model_param(model, "class_ids"),
}

# Configurar el modelo
h.set_dl_model_param(model, "batch_size", batch_size_inference)
h.set_dl_model_param(model, "device", device)

#############################
# Funciones personalizadas
#############################

def log(msg:str):
    if debug_mode:
        print(f'[{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {msg}\n')


def gen_dl_sample(batch:h.HObject) -> h.HHandle:
    '''
    Generar DL_Sample en un formato "halcon-friendly":
    '''
    sample = [h.from_python_dict({'image': image}) for image in batch]
    sample = tuple(sample)

    return sample


def preprocess(ruta:str, ruta_salida:str):
    '''
    Preprocesa las imagenes y las guarda en el directorio designado
    '''
    # Crear carpeta de salida si no existe
    if not os.path.exists(ruta_salida):
        os.makedirs(ruta_salida)

    image_files = os.listdir(ruta)
    for img_dir in image_files:
        # leer imagen
        input_path = os.path.join(ruta, img_dir)
        img = cv.imread(input_path)

        # Cambiar tamaño de la imagen a 768x576
        resized_image = cv.resize(img, (768, 576), interpolation=cv.INTER_CUBIC)

        # Guardar la imagen en la carpeta de salida
        output_path = os.path.join(ruta_salida, img_dir)
        cv.imwrite(output_path, resized_image)

    log(f"Preprocesamiento completado. Imágenes guardadas en {ruta_salida}")

#############################
# Main
#############################

# preprocesar imagenes
preprocess(image_dir, preproces_dir)

# Obtener path de imágenes
image_paths = os.listdir(preproces_dir)

# Inicializar variables
count_objects = 0
tracked_objects = []
new_object_id = 1
inventario_final = []
obj_contados = []

# Loop para procesar lotes
for batch_index in range(len(image_paths) // batch_size_inference):
    batch = image_paths[batch_index * batch_size_inference:(batch_index + 1) * batch_size_inference]
    batch = [preproces_dir+'/'+i for i in batch]

    # leer imagenes con halcon
    image_batch = h.read_image(batch)

    # preprocesar halcon, img byte a real
    image_batch = h.convert_image_type(image_batch,'real')
    log(image_batch)

    # Crear muestras de DL
    dl_samples = gen_dl_sample(image_batch)
    
    # Aplicar modelo
    dl_results = h.apply_dl_model(model, dl_samples,[])

    for sample_index, (dl_sample, dl_result) in enumerate(zip(dl_samples, dl_results)):
        current_frame_objects = []

        # Convertir a diccionarios para facilidad de uso
        dl_result = h.as_python_dict(dl_result)
        dl_sample = h.as_python_dict(dl_sample)
        
        # Obtener bounding boxes
        detected_bbox = {
            "row1": dl_result["bbox_row1"],
            "row2": dl_result["bbox_row2"],
            "col1": dl_result["bbox_col1"],
            "col2": dl_result["bbox_col2"],
            "class_ids": dl_result["bbox_class_id"]
        }

        for i in range(len(detected_bbox["class_ids"])):
            # Calcular centroides y clase
            centroid_x = (detected_bbox["col1"][i] + detected_bbox["col2"][i]) / 2
            centroid_y = (detected_bbox["row1"][i] + detected_bbox["row2"][i]) / 2
            class_id = detected_bbox["class_ids"][i]
        
            # Verificar rastreo previo
            object_tracked = False
            for tracked in tracked_objects:
                tracked_centroid = tracked["centroid"]
                tracked_id = tracked["id"]
                tracked_class_id = tracked["idClase"]

                # Distancia
                distance = ((tracked_centroid[0] - centroid_x)**2 + (tracked_centroid[1] - centroid_y)**2) ** 0.5

                if distance < 80 and tracked_class_id == class_id:
                    tracked["centroid"] = [centroid_x, centroid_y]
                    object_tracked = True
                    break
            
            # No rastreado previamente
            if not object_tracked:
                new_tracked_object = {
                    "id": new_object_id,
                    "centroid": [centroid_x, centroid_y],
                    "idClase": class_id
                }

                tracked_objects.append(new_tracked_object)
                object_id = new_object_id
                new_object_id += 1
            else:
                object_id = tracked["id"]

            current_frame_objects.append(object_id)

            # Verificar si cruzó línea de conteo
            if detected_bbox["row1"][i] < count_line and object_id not in obj_contados:
                count_objects += 1
                inventario_final.append(class_id)
                obj_contados.append(object_id)

                # Guardar imagen recortada
                save_dir = f"imgClasificacion00/{dataset_info['class_names'][class_id]}"
                sample_image = dl_sample['image']
                cropped_image = h.crop_part(sample_image,
                    detected_bbox["row1"][i],
                    detected_bbox["col1"][i],
                    detected_bbox["col2"][i] - detected_bbox["col1"][i],
                    detected_bbox["row2"][i] - detected_bbox["row1"][i]
                )

                file_name = f"{save_dir}/pieza_{object_id}.png"
                cropped_image = h.convert_image_type(cropped_image,'byte')
                h.write_image(cropped_image, "png", 0, file_name)

        # Limpiar objetos rastreados no presentes
        tracked_objects = [tracked for tracked in tracked_objects if tracked["id"] in current_frame_objects]

        # Mostrar resultados
        element_info = "".join([
            f"{dataset_info['class_names'][i]}={inventario_final.count(i)}, "
            for i in dataset_info['class_ids'] if i in inventario_final
        ])

        print(f"Frame {sample_index}: {element_info} Total: {count_objects}")

# Finalizar
print(f"Conteo total: {count_objects}")