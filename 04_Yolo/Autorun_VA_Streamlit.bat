set env_path="C:\Users\horno4pg\OneDrive - Corona\Documentos\VA_Entrada_Horno4_GR"
call %env_path%\env\Scripts\activate.bat
call python "C:\Users\horno4pg\OneDrive - Corona\Documentos\VA_Entrada_Horno4_GR\04_Yolo\camera_inference_forever.py" --server.port 8535
pause