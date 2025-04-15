# ----------------------------------------------------------------------------------------------------------------------
# Library
import os
import time
import math
import numpy as np
import datetime
import threading
import cv2
from keras.models import load_model
from harvesters.core import Harvester
from pymodbus.client import ModbusTcpClient


# ----------------------------------------------------------------------------------------------------------------------
#                                       MATCHING DEFINITION
# ----------------------------------------------------------------------------------------------------------------------
# Ref Matching folders
source_dir = '.\\data'
shape_path = os.path.join(source_dir, 'Model_Shape')

# Matching: Read the shape model from file
template1 = cv2.imread(shape_path + "/Ref_1.png", 0)
template2 = cv2.imread(shape_path + "/Ref_2.png", 0)
template3 = cv2.imread(shape_path + "/Ref_3.png", 0)

templateHole = cv2.imread(shape_path + "/Ref_Hole.png", 0)

# Threshold variable
threshold_match_ref = 0.9
threshold_match_hol = 0.8

# ----------------------------------------------------------------------------------------------------------------------
#                                       MATCHING DEFINITION 0584
# ----------------------------------------------------------------------------------------------------------------------
# Template for the 3151 ref
template_0584 = cv2.imread(shape_path + "/Ref_0584.png", 0)

# Threshold variable
threshold_match_0584 = 0.75

# ROI Match REF 0584
y_init_ref_0584 = 500
delta_y_ref_0584 = 600
x_init_ref_0584 = 1300
delta_x_ref_0584 = 420

# ----------------------------------------------------------------------------------------------------------------------
#                                       MATCHING DEFINITION 3008
# ----------------------------------------------------------------------------------------------------------------------
# Template for the 3151 ref
template_3008 = cv2.imread(shape_path + "/Ref_3008.png", 0)

# Threshold variable
threshold_match_3008 = 0.85

# ROI Match REF 3008
y_init_ref_3008 = 500
delta_y_ref_3008 = 600
x_init_ref_3008 = 1300
delta_x_ref_3008 = 420

# ----------------------------------------------------------------------------------------------------------------------
#                                       ROI CONFIGURATION
# ROI coordinate Y
y_init_1 = 770
y_init_2 = 60
y_init_3 = 1510
delta_y = 210

# ROI coordinate X
x_init_1 = 70
x_init_2_3 = 980
delta_x = 200

# ROI coordinate Y
y_init_hole_1 = 500
y_init_hole_2 = 800
delta_y_hole = 300

# ROI coordinate X
x_init_hole_1_2 = 1100
delta_x_hole = 300
# ----------------------------------------------------------------------------------------------------------------------
#                                       CALIBRATION
# ----------------------------------------------------------------------------------------------------------------------
# Distance hole for pixel to mm
distance_hole_mm = 150

# Variables para calibracion de angulo y ejes
offset_Angle = 0
offset_Z = 0
offset_Y = 0

# Función para salvar una imagen en una ruta establecida
def save_image(directory, filename, image):
    if not os.path.exists(directory):
        os.makedirs(directory)
    cv2.imwrite(directory + filename, image)


# Función que devuelve fecha y hora actual
def get_date_time():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# Función que abre archivo y realiza append al final
def write_log(directory, message):
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(directory + "log.txt", "a") as f:
        f.write(f"{get_date_time()} {message}\n")


# Función para convertir enteros con signo a sin signo
def signed_to_unsigned(n, byte_count):
    return int.from_bytes(n.to_bytes(byte_count, 'little', signed=True), 'little', signed=False)


# Función que realiza el template matching
def template_matching(img, template, threshold, y, delta_y, x, delta_x):
    # ------------------------------------------------------------------------------------------------------------------
    # Template Size
    # ------------------------------------------------------------------------------------------------------------------
    wt, ht = template.shape[::-1]
    # ------------------------------------------------------------------------------------------------------------------
    # ROI definition
    # ------------------------------------------------------------------------------------------------------------------
    # Matching: Build the ROI for searching the reference
    roi = img[y:y + delta_y, x:x + delta_x]

    # ------------------------------------------------------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------------------------------------------------------
    res_h1 = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res_h1)

    if max_val > threshold:
        print(f'El match tiene una precisión de {max_val}', flush=True)
        top_left = max_loc
        bottom_right = (top_left[0] + wt, top_left[1] + ht)

        pos_ref = [0, 0]
        pos_ref[0] = int(((bottom_right[0] - top_left[0]) / 2) + top_left[0])
        pos_ref[1] = int(((bottom_right[1] - top_left[1]) / 2) + top_left[1])

        return pos_ref, roi, False, max_val
    else:
        pos_ref = [0, 0]
        print(f'No se encontro un match para el template el valor hallado fue de {max_val}', flush=True)
        return pos_ref, roi, True, max_val


def angle_center_line(pos_ref1, pos_ref2, x, y1, y2):
    # Variable initiation
    pos_center = [0, 0]

    # Center definition
    pos_center[0] = int(((pos_ref2[0] + x - pos_ref1[0] + x) / 2) + pos_ref1[0])
    pos_center[1] = int(((pos_ref2[1] + y2 - pos_ref1[1] + y1) / 2) + pos_ref1[1])

    # Angle definition
    angle_hole_rad = np.arcsin(((pos_ref1[0] + x) - (pos_ref2[0] + x)) /
                               math.hypot((pos_ref1[0] + x) - (pos_ref2[0] + x), (pos_ref1[1] + y1) -
                                          (pos_ref2[1] + y2)))
    angle_hole = np.degrees(angle_hole_rad)
    angle_hole = 90 + angle_hole

    return angle_hole, pos_center


def config_genienano():
    # Instantiate a Harvester object
    h = Harvester()

    # Load a GenTL Producer; you can load many more if you want to:
    gentl_file = r"C:\Program Files\MATRIX VISION\mvIMPACT Acquire\bin\x64\mvGenTLProducer.cti"
    h.add_file(gentl_file)
    # print(h.files)

    # Enumerate the available devices that GenTL Producers can handle:
    h.update()
    # print(h.device_info_list)
    # ------------------------------------------------------------------------------------------------------------------
    # Select a target device and create an ImageAcquire object that controls the device
    ia = h.create({'serial_number': camera_serial_number})  # If we want to be more specific
    # ------------------------------------------------------------------------------------------------------------------
    # Manipulating GenICam Feature Nodes
    # print(dir(ia.remote_device.node_map))

    # Camera Temperature
    cam_temp = ia.remote_device.node_map.DeviceTemperature.value
    print(f"La temperatura de la camara es {cam_temp}", flush=True)

    return h, ia


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def conexion_camara(h, ia):
    print('Ready!!!', flush=True)

    # Instanciar la conexión al PLC
    client_trigger = ModbusTcpClient(ipaddress_plc)  # IP del PLC
    client_trigger.connect()

    # Inicialización variables
    z_position = 0
    y_position = 0
    toilet_angle = 0
    counter = 0

    flag_noref = False
    flag_sintapa = False
    flag_nohole = False
    flag_noidentify = False
    flag_empty = False

    while True:
        try:
            # Read trigger
            trigger = client_trigger.read_holding_registers(trigger_address, 1)
            try:
                # Verificación Valor Trigger
                valor_trigger = trigger.registers[0]
            except AttributeError:
                # Escribir mensaje en el archivo de registro
                write_log('./log/', "Registro trigger no tiene valor")
                valor_trigger = 0

        except:
            # Escribir mensaje en el archivo de registro
            write_log('./log/', "No se pudo leer registro trigger de PLC")
            valor_trigger = 0

            # Wait
            time.sleep(0.1)

        # Print Trigger
        # print(f'Fecha: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} valor trigger {valor_trigger}')

        # Taking a picture when the trigger is 1
        if valor_trigger == 1:
            # ----------------------------------------------------------------------------------------------------------
            # Conexión camara y toma de imagen
            # ----------------------------------------------------------------------------------------------------------
            ia.start()

            # Camera Temperature
            cam_temp = ia.remote_device.node_map.DeviceTemperature.value
            print(f"La temperatura de la camara es {cam_temp}", flush=True)

            with ia.fetch() as buffer:
                # Let's create an alias of the 2D image component:
                component = buffer.payload.components[0]
                img_original_camara = component.data.reshape(component.height, component.width,
                                                             int(component.num_components_per_pixel))

                # Crear una copia de la imagen
                img_procesada = img_original_camara.copy()
                img_original = img_original_camara.copy()

                # Detener Camara
                ia.stop()  # Stop image acquisition

            # ----------------------------------------------------------------------------------------------------------
            # Image Analysis
            # ----------------------------------------------------------------------------------------------------------
            # Resizing the image
            img = cv2.resize(img_procesada, (360, 320))

            # Showing the image
            # cv2.imshow('Image', img)

            print('Taking a picture and Analyzing with AI', flush=True)
            print(f'Fecha: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', flush=True)

            # ------------------------------------------------------------------------------------------------------
            #                                               Initialization
            # ------------------------------------------------------------------------------------------------------
            text_angulo = ""
            text_z = ""
            text_y = ""
            # ------------------------------------------------------------------------------------------------------

            # ------------------------------------------------------------------------------------------------------
            # IA identification
            # ------------------------------------------------------------------------------------------------------
            # Rescale
            img_scale = img / 255

            # Convert single image to a batch.
            img_arr = np.array([img_scale])

            # IA Model
            prediction_classes = model.predict(img_arr)
            result_ia = ref_dict[np.argmax(prediction_classes)]
            probability_ia = np.round(prediction_classes[0][np.argmax(prediction_classes, axis=-1)[0]], 3)

            # ------------------------------------------------------------------------------------------------------
            # Alarm NO IDENTIFICADA
            # ------------------------------------------------------------------------------------------------------
            if probability_ia < threshold_ia:
                message = f"Referencia No Identificada la probabilidad fue de {probability_ia}"
                flag_noidentify = True

                # Numero Programa PLC
                numero_programa = 0
            else:
                # Numero Programa PLC
                numero_programa = ref_dict_programas[result_ia]

                flag_noidentify = False
                flag_empty = False
                # --------------------------------------------------------------------------------------------------
                # Alarm EMPTY
                # --------------------------------------------------------------------------------------------------
                if result_ia == "Empty":
                    message = "No hay taza"
                    flag_empty = True
                # --------------------------------------------------------------------------------------------------
                # Alarm SIN TAPA
                # --------------------------------------------------------------------------------------------------
                elif result_ia == "Sin_Tapa":
                    message = "La referencia no tiene tapa"
                    flag_sintapa = True

                else:
                    # --------------------------------------------------------------------------------------------------
                    # Reference recognition CODIGO MOMENTANEO
                    # --------------------------------------------------------------------------------------------------
                    print('Match 0584', flush=True)
                    _, _, flag_0584, precision_0584 = template_matching(img_original, template_0584,
                                                                        threshold_match_0584,
                                                                        y_init_ref_0584, delta_y_ref_0584,
                                                                        x_init_ref_0584, delta_x_ref_0584)

                    print('Match 3008', flush=True)
                    _, _, flag_3008, precision_3008 = template_matching(img_original, template_3008,
                                                                        threshold_match_3008,
                                                                        y_init_ref_3008, delta_y_ref_3008,
                                                                        x_init_ref_3008, delta_x_ref_3008)

                    if not flag_3008 and precision_3008 > precision_0584:
                        result_ia = '3008'

                    flag_sintapa = False

                    # ----------------------------------------------------------------------------------------------
                    # Reference Matching
                    # ----------------------------------------------------------------------------------------------
                    # Ref1 matching
                    pos_ref1, roi_ref1, flag_noref1, _ = template_matching(img_original, template1, threshold_match_ref,
                                                                           y_init_1, delta_y, x_init_1, delta_x)

                    # Ref2 matching
                    pos_ref2, roi_ref2, flag_noref2, _ = template_matching(img_original, template2, threshold_match_ref,
                                                                           y_init_2, delta_y, x_init_2_3, delta_x)

                    # Ref3 matching
                    pos_ref3, roi_ref3, flag_noref3, _ = template_matching(img_original, template3, threshold_match_ref,
                                                                           y_init_3, delta_y, x_init_2_3, delta_x)

                    # Calculation if reference where found
                    if flag_noref1 is False and flag_noref2 is False and flag_noref3 is False:
                        angle_ref, pos_ref_line = angle_center_line(pos_ref2, pos_ref3, x_init_2_3, y_init_2, y_init_3)
                        flag_noref = False
                    else:
                        # ------------------------------------------------------------------------------------------
                        # Alarm NO REF | FALLA ILUMINACION
                        # ------------------------------------------------------------------------------------------
                        flag_noref = True
                        message = "No se encontraron las referencias, FALLA ILUMINACION"

                        # Roi display
                        # roi_ref = cv2.hconcat([roi_ref1, roi_ref2, roi_ref3])
                        # cv2.imshow('Roi Ref', roi_ref)
                    # --------------------------------------------------------------------------------------------------
                    # Hole matching
                    # --------------------------------------------------------------------------------------------------
                    # Hole matching 1
                    pos_ref_h1, roi_hole1, flag_nohole1, _ = template_matching(img_procesada, templateHole,
                                                                               threshold_match_hol, y_init_hole_1,
                                                                               delta_y_hole, x_init_hole_1_2,
                                                                               delta_x_hole)

                    # Hole matching 2
                    pos_ref_h2, roi_hole2, flag_nohole2, _ = template_matching(img_procesada, templateHole,
                                                                               threshold_match_hol, y_init_hole_2,
                                                                               delta_y_hole, x_init_hole_1_2,
                                                                               delta_x_hole)

                    # Calculation if both hole where found
                    if flag_nohole1 is False and flag_nohole2 is False:
                        angle_hole, pos_ref_hole = angle_center_line(pos_ref_h1, pos_ref_h2, x_init_hole_1_2,
                                                                     y_init_hole_1, y_init_hole_2)
                        flag_nohole = False
                    else:
                        # ------------------------------------------------------------------------------------------
                        # Alarm NO HOLE | Falla de coordenadas
                        # ------------------------------------------------------------------------------------------
                        flag_nohole = True
                        message = "No se encontraron los huecos del asiento | Falla de coordenadas"

                        # Roi display
                        # roi_hole = cv2.hconcat([roi_hole1, roi_hole2])
                        # cv2.imshow('Roi Hole', roi_hole)
                    # ----------------------------------------------------------------------------------------------
                    # Position and Angle
                    # ----------------------------------------------------------------------------------------------
                    if flag_nohole is False and flag_noref is False:
                        distance_pixel = math.hypot(pos_ref_h1[0] - pos_ref_h2[0],
                                                    (pos_ref_h1[1] + y_init_hole_1) - (pos_ref_h2[1] + y_init_hole_2))

                        if flag_0584 is False and precision_0584 > precision_3008:
                            if distance_pixel >= 266:
                                result_ia = "0584"
                            else:
                                result_ia = "0582"

                            print(f"La distancia en Pixeles es {distance_pixel}", flush=True)

                        # Numero Programa PLC
                        numero_programa = ref_dict_programas[result_ia]

                        message = (f"La imagen es una taza referencia {result_ia} con una probabilidad de "
                                   f"{probability_ia}")

                        # calculation of millimeters per pixel in fixing holes
                        millimeters_per_pixel = distance_hole_mm / distance_pixel
                        toilet_angle = (angle_ref - angle_hole) * 10

                        # Ajuste de X -> Y &  Y -> Z para coincidir con el robot
                        y_position = ((pos_ref_hole[0] - pos_ref_line[0]) * millimeters_per_pixel)
                        z_position = ((pos_ref_line[1] - pos_ref_hole[1]) * millimeters_per_pixel)

                        # Ajuste de calibracion con respecto a BOA 1600
                        toilet_angle = toilet_angle + offset_Angle
                        z_position = z_position + offset_Z
                        y_position = y_position + offset_Y

                        # Message configuration
                        text_angulo = "El angulo de la pieza es de {} grados".format(int(toilet_angle))
                        text_z = "Posicion en z es {} mm.".format(int(np.ceil(z_position)))
                        text_y = "Posicion en y es {} mm.".format(int(np.ceil(y_position)))

                        # ------------------------------------------------------------------------------------------
                        # Alarm ANGULO MAYOR AL PERMITIDO
                        # ------------------------------------------------------------------------------------------
                        # if toilet_angle > 5 or toilet_angle < -5:
                        #     flag_angulo_mayor = True
                        #     print("Angulo mayor al permitido")

                        # ------------------------------------------------------------------------------------------
                        # Imagen modification
                        # ------------------------------------------------------------------------------------------
                        # Reference Matching
                        img_procesada = cv2.circle(img_procesada, (pos_ref_line[0], pos_ref_line[1]), radius=0,
                                                   color=(255, 255, 255), thickness=20)
                        img_procesada = cv2.line(img_procesada, (pos_ref2[0] + x_init_2_3,
                                                                 pos_ref2[1] + y_init_2), (pos_ref3[0] + x_init_2_3,
                                                                                           pos_ref3[1] + y_init_3),
                                                 (255, 255, 255), 6)

                        # Hole Matching
                        img_procesada = cv2.circle(img_procesada, (pos_ref_hole[0], pos_ref_hole[1]), radius=0,
                                                   color=(255, 255, 255), thickness=10)
                        img_procesada = cv2.line(img_procesada, (pos_ref_h1[0] + x_init_hole_1_2,
                                                                 pos_ref_h1[1] + y_init_hole_1),
                                                 (pos_ref_h2[0] + x_init_hole_1_2, pos_ref_h2[1] + y_init_hole_2),
                                                 (255, 255, 255), 6)
                    else:
                        text_angulo = "No se pudo calcular el angulo"
                        text_z = "No se pudo calcular la posicion en Z"
                        text_y = "No se pudo calcular la posicion en Y"

            # ----------------------------------------------------------------------------------------------------------
            # Message Output
            # ----------------------------------------------------------------------------------------------------------
            counter += 1

            if counter == 65000:
                counter = 0

            print(message, flush=True)
            print(text_angulo, flush=True)
            print(text_z, flush=True)
            print(text_y, flush=True)
            print("\n", flush=True)

            # Putting the text in the image
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img_procesada, message, (5, 1600), font, 1.2, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img_procesada, text_angulo, (5, 1635), font, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(img_procesada, text_z, (5, 1670), font, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(img_procesada, text_y, (5, 1705), font, 1.2, (255, 255, 255), 3, cv2.LINE_AA)

            # Roi drawing
            cv2.rectangle(img_procesada, (x_init_hole_1_2, y_init_hole_1), (x_init_hole_1_2 + delta_x_hole,
                                                                            y_init_hole_1 + delta_y_hole), 255, 2)
            cv2.rectangle(img_procesada, (x_init_hole_1_2, y_init_hole_2), (x_init_hole_1_2 + delta_x_hole,
                                                                            y_init_hole_2 + delta_y_hole), 255, 2)

            # Resizing the image
            img_procesada_resize = cv2.resize(img_procesada, (360 * 2, 320 * 2))

            # ----------------------------------------------------------------------------------------------------------
            # Saving the image with the recognition label on the name
            # ----------------------------------------------------------------------------------------------------------
            if capture_image is True:
                if probability_ia >= threshold_ia:
                    save_image(f'./results/{result_ia}/Procesada/',
                               f'test_Genie_{result_ia}_{int(time.time())}.png', img_procesada_resize)

                    save_image(f'./results/{result_ia}/Raw/',
                               f'test_Genie_{result_ia}_{int(time.time())}.png', img_original)
                else:
                    save_image(f'./results/Low_Confident/{result_ia}/Procesada/',
                               f'test_Genie_{result_ia}_{int(time.time())}.png', img_procesada_resize)

                    save_image(f'./results/Low_Confident/{result_ia}/Raw/',
                               f'test_Genie_{result_ia}_{int(time.time())}.png', img_original)

                # Alarmas
                if flag_empty is True:  # Sin pieza
                    save_image(f'./results/Alarmas/Empty/Procesada/',
                               f'test_Genie_{result_ia}_{int(time.time())}.png', img_procesada_resize)

                    save_image(f'./results/Alarmas/Empty/Raw/',
                               f'test_Genie_{result_ia}_{int(time.time())}.png', img_original)

                elif flag_nohole is True:  # Falla de coordenadas
                    save_image(f'./results/Alarmas/Coordenadas/Procesada/',
                               f'test_Genie_{result_ia}_{int(time.time())}.png', img_procesada_resize)

                    save_image(f'./results/Alarmas/Coordenadas/Raw/',
                               f'test_Genie_{result_ia}_{int(time.time())}.png', img_original)

                elif flag_sintapa is True:  # falta de tapa
                    save_image(f'./results/Alarmas/Falta_Tapa/',
                               f'test_Genie_{result_ia}_{int(time.time())}.png', img_procesada_resize)

                elif flag_noref is True:  # iluminacion
                    save_image(f'./results/Alarmas/iluminacion/',
                               f'test_Genie_{result_ia}_{int(time.time())}.png', img_procesada_resize)
            # ----------------------------------------------------------------------------------------------------------
            # Lectura de ModbusTcp
            # ----------------------------------------------------------------------------------------------------------
            # Instanciar la conexión
            client = ModbusTcpClient(ipaddress_plc)  # IP del PLC
            client.connect()

            # Variable que recoge banderas de alarmas
            flag_alarmas = flag_noref * 2 ** 0 + flag_sintapa * 2 ** 1 + flag_nohole * 2 ** 2 + flag_noidentify * 2 ** 3
            client.write_register(alarma_address, flag_alarmas)

            # Escribir los registros de la dirección 0, 1 registro, unidad 1
            client.write_register(programa_address, int(numero_programa))
            client.write_register(posz_address, signed_to_unsigned(int(np.ceil(z_position)), 2))
            client.write_register(posy_address, signed_to_unsigned(int(np.ceil(y_position)), 2))
            client.write_register(angulo_addres, signed_to_unsigned(int(np.ceil(toilet_angle)), 2))
            client.write_register(contador_address, int(counter))

            # Variable que confirma el envio de datos
            flag_confirmacion_imagen = 0 * 2 ** 0 + 1 * 2 ** 1 + 0 * 2 ** 2 + 0 * 2 ** 3
            client.write_register(confirmacion_address, flag_confirmacion_imagen)

            # Cerrar la conexión para evitar choques de red
            client.close()

            # Limpiar registros
            numero_programa = 0
            z_position = 0
            y_position = 0
            toilet_angle = 0

            time.sleep(1)


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def main():
    # Keep alive thread
    thread = threading.Thread(target=keep_alive, daemon=True, args=(keep_alive_cont,))
    thread.start()

    # Conexión Camara
    h, ia = config_genienano()
    conexion_camara(h, ia)


# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
