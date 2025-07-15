from pymodbus.client import ModbusTcpClient
import time

ipaddress_plc = '10.126.64.112'
# %MW40 a %MW49
alarm_address = 40


client = ModbusTcpClient(ipaddress_plc)  # IP del PLC
client.connect()


# Aquí toca agregar la parte de enviar al PLC
# client.write_register(alarm_address, 0)

# Read trigger
alarma = client.read_holding_registers(alarm_address, 2)
print(alarma.decode)