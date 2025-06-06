from subprocess import Popen
import sys
import datetime

filename = 'YOLO_camera_inference.py'
while True:
    print(f'\nDate: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print("Starting " + filename)
    p = Popen("python " + filename , shell=True)
    p.wait()