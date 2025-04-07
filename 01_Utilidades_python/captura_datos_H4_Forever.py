from subprocess import Popen
import time
import datetime

def log(msg):
    print(f'\n[{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {msg}')

filename = 'captura_datos_H4.py'
time_sleep = 10
while True:
    log(f'Iniciando {filename}')
    p = Popen(f"python {filename} 1>>out.log 2>>err.log", shell=True)
    p.wait()
    log(f'Reconexion automatica en {time_sleep} minutos')
    time.sleep(60 * time_sleep)