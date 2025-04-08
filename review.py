# PREVISUALIZACION DE BIBLIOTECA DE VIDEOS
# ANALISIS DE INVENTARIOS DESDE DB 

from sqlalchemy import create_engine, exc, URL
import plotly.graph_objects as go
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import datetime
import pyodbc
import math
import time
import os


# ******************************************************
# Configuraciones iniciales
# ******************************************************

debug_mode: bool = True
path: str = '00_Data/videos/inferencias'
RECURRENCIA = 'recurrencia [Min]'

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
connection_str = "DRIVER={ODBC Driver 17 for SQL Server};SERVER=%s;DATABASE=%s;UID=%s;PWD=%s;Encrypt=no" % (server, database, username, password)
connection_url = URL.create("mssql+pyodbc", query={"odbc_connect": connection_str})


def log(msg:str):
    if debug_mode:
        print(f'[{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {msg}\n')


def get_sql(sel_dia_ini, sel_dia_fin) -> pd.DataFrame:
    # Connection keys 
    conn = create_engine(connection_url)
    df = pd.DataFrame()
    try:
        # Execute the query
        with conn.begin() as connection:
            QUERY = "SELECT * FROM [{}].[dbo].[{}] WHERE (Fecha BETWEEN '{}' AND '{}')"
            df = pd.read_sql_query(QUERY.format(database, tabla, sel_dia_ini, sel_dia_fin), connection)
    except (exc.TimeoutError, pyodbc.OperationalError):
        log("La consulta ha superado el tiempo límite.")
    finally:
        conn.dispose()  # Close the connection

    return df

#############################
# Main
#############################
st.title('Inventario')

# Selector de periodo
st.subheader("2) Selección de Periodo a Analizar")
c1, c2 = st.columns(2)

sel_dia_ini = c1.date_input("Seleccione el día inicial", 
                            datetime.date.today() - datetime.timedelta(days=1), 
                            key="dia_ini")
sel_dia_fin = c1.date_input("Seleccione el día final", datetime.date.today(), key="dia_fin")

if sel_dia_fin <= sel_dia_ini:
    c2.error("Recuerda seleccionar una Fecha inicial anterior a la Fecha final!!!")
    st.stop()
elif sel_dia_fin > datetime.date.today():
    c2.error("Recuerda que la Fecha final no puede ser superior a la Fecha actual")
    st.stop()
else:
    c2.info("Analizaras un periodo de tiempo de " + str((sel_dia_fin - sel_dia_ini).days + 1) + " días.")
        

# Get data
inventario = get_sql(sel_dia_ini,sel_dia_fin)
inventario.sort_values(by='Fecha',inplace=True)
inventario[RECURRENCIA] = inventario['Fecha']- inventario['Fecha'].shift()
inventario[RECURRENCIA] = inventario[RECURRENCIA].iloc[1:].apply(lambda x: math.ceil(x.seconds/60))

# Mostrar
st.table(inventario[['Fecha', 'Lavamanos', 'Onepiece', 'Pedestal', 'Tanque', 'Taza', RECURRENCIA, 'Colision']])

# Resumen de totales
class_names = ['Lavamanos', 'Onepiece', 'Pedestal', 'Tanque', 'Taza']
totales = inventario[class_names].sum(axis=0)
totales: dict = {key: total for key, total in zip(class_names, totales)}

st.info(f'Total: {totales}')

# Grafica recurrencia
fig = go.Figure(
    go.Bar(
        x=inventario["Fecha"],
        y=inventario["recurrencia [Min]"],
        name='asd'
    )
)

# # Grafica 2
frecuencia = inventario.groupby('recurrencia [Min]')[RECURRENCIA].count().reset_index(name="frecuencia")

fig2 = go.Figure(
    go.Bar(
        x=frecuencia["recurrencia [Min]"],
        y=frecuencia["frecuencia"],
    )
)

fig.update_layout(title_text='Periodo de frecuencia entrada de carros')
fig2.update_layout(title_text='Distribucion de frecuencias.')
st.plotly_chart(fig)
st.plotly_chart(fig2)

# Obtener videos
path_videos = os.listdir(path)

# Título y selector de video
st.title('Preview')
video_selector = st.selectbox('Seleccione el video deseado', path_videos, index=0)
path_video = os.path.join(path, video_selector)

# Contenedor para el video
prev = st.empty()

prev.video(path_video, format="video/mp4")

# Botón de descarga
with open(path_video, "rb") as f:
    video_bytes = f.read()
st.download_button(
    label="Descargar Video",
    data=video_bytes,
    file_name=video_selector,
    mime="video/mp4",
)