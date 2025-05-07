device_access_status = {
    0: "0, DEVICE_ACCESS_STATUS_UNKNOWN - Estado desconocido",
    1: "1, DEVICE_ACCESS_STATUS_READWRITE - Acceso de lectura y escritura (control total)",
    2: "2, DEVICE_ACCESS_STATUS_READONLY - Acceso solo de lectura",
    3: "3, DEVICE_ACCESS_STATUS_NOACCESS - Sin acceso al dispositivo",
    4: "4, DEVICE_ACCESS_STATUS_BUSY - El dispositivo ya está en uso por otro proceso",
    5: "5, DEVICE_ACCESS_STATUS_OPEN_READONLY - Abierto en modo solo lectura por este proceso",
    6: "6, DEVICE_ACCESS_STATUS_OPEN_READWRITE - Abierto con control total por este proceso"
}

class DeviceAccessStatusError(Exception):
    """Exception raised for access status != 1:
    | Código | Constante                             | Significado                                                          |
    | ------ | ------------------------------------- | -------------------------------------------------------------------- |
    | `0`    | `DEVICE_ACCESS_STATUS_UNKNOWN`        | Estado desconocido                                                   |
    | `1`    | `DEVICE_ACCESS_STATUS_READWRITE`      | Acceso de lectura y escritura (control total)                        |
    | `2`    | `DEVICE_ACCESS_STATUS_READONLY`       | Acceso solo de lectura (no se puede iniciar adquisición de imágenes) |
    | `3`    | `DEVICE_ACCESS_STATUS_NOACCESS`       | Sin acceso al dispositivo                                            |
    | `4`    | `DEVICE_ACCESS_STATUS_BUSY`           | El dispositivo ya está en uso por otro proceso                       |
    | `5`    | `DEVICE_ACCESS_STATUS_OPEN_READONLY`  | El dispositivo está abierto en modo solo lectura por este proceso    |
    | `6`    | `DEVICE_ACCESS_STATUS_OPEN_READWRITE` | El dispositivo está abierto con control completo por este proceso    |


    Attributes:
        message
    """
    def __init__(self, estado):
        self.message = f'Estado de acceso de la camara no valido: {device_access_status[estado]}'
        super().__init__(self.message)