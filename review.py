# PREVISUALIZACION DE BIBLIOTECA DE VIDEOS
# ANALISIS DE INVENTARIOS DESDE DB 


# TODO
# Encabezado e intro

# Tabla por defecto escondida y boton descarga


from sqlalchemy import create_engine, exc, URL
from plotly.subplots import make_subplots
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
POR_DIA = 'Por día'
POR_RANGO_DIA = 'Por rango de días'
clases = ['Lavamanos', 'Onepiece', 'Pedestal', 'Tanque', 'Taza']


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


def add_day(day, add=1):
    """
    Función agrega o quita dias, teniendo en cuenta inicio de mes e inicio de año\n
    :param day: "2021-02-01"  EN STRING
    :param add: numero de dias a operar
    
    :return fin_date: día con los días sumados o restados en STR al día ingresado
    """
    l_day_n = [int(x) for x in day.split("-")]
    ini_date = datetime.date(l_day_n[0], l_day_n[1], l_day_n[2])
    fin_date = ini_date + datetime.timedelta(days=add)

    return str(fin_date)


def get_sql(sel_dia_ini:str = '2025-04-03', sel_dia_fin:str = '2025-04-04') -> pd.DataFrame:
    hora = ' 06:00:00.000'

    # Agregar hora para seleccion correcta de dia segun el turno
    sel_dia_ini += hora
    sel_dia_fin += hora

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
        

# Función para asignar turnos
def asignar_turno(fecha):
    # turnos 6-2 (1:59:59.9999999), 2-10 , 10-6
    hora = fecha.time()

    if datetime.time(6, 0) <= hora < datetime.time(14, 0):
        return 'Turno 1'
    elif datetime.time(14, 0) <= hora < datetime.time(22, 0):
        return 'Turno 2'
    else:
        return 'Turno 3'


# Mapear los valores a cuadros coloreados (Unicode cuadrados)
def convertir_a_cuadro(val):
    return '🟥' if val == 1 else '🟩'

def graficar_apilado_dias(df):
    fig = go.Figure()
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    df_grouped = df.groupby('Dia')[clases].sum().reset_index()

    for clase in clases:
        fig.add_trace(get_barra_apilada(df_grouped, clase))

    fig.update_layout(
        barmode='stack',
        height=400,
        width=900,
        title_text="Resumen inventario por Dias.",
        template='plotly_white'
    )

    return fig

def graficar_apilado_turnos(lista_dfs):
    fig = make_subplots(
        rows=1, cols=3,
        shared_yaxes=True,
        subplot_titles=['Turno 1','Turno 2','Turno 3']
    )

    for i, df in enumerate(lista_dfs):
        df['Fecha'] = pd.to_datetime(df['Fecha'])
        df_grouped = df.groupby('Dia')[clases].sum().reset_index()

        for clase in clases:
            fig.add_trace(get_barra_apilada(df_grouped, clase), row=1, col=i+1)

    fig.update_layout(
        barmode='stack',
        height=400,
        width=900,
        title_text="Resumen inventario por turno.",
        template='plotly_white'
    )

    return fig


def get_barra_apilada(df_grouped, clase):
    
    return go.Bar(
        x=df_grouped['Dia'],
        y=df_grouped[clase],
        name=clase,
        showlegend=True
    )

#############################
# Main
#############################
st.title('Inventario')

# Selector de periodo
st.subheader("Selección de Periodo a Analizar")
col1, col2, col3 = st.columns(3)
with col1:
    sel_fecha = st.radio("¿Que periodo de tiempo desea analizar?",
                            (POR_DIA, POR_RANGO_DIA), key="fecha")

    # Descargar nuevamente flag
    flag_download = False

with col2:
    # Opciones por día
    if sel_fecha == POR_DIA:
        sel_dia_ini = st.date_input("¿Que dia desea analizar?", datetime.date.today() -
                                    datetime.timedelta(days=1), key="dia")
        if sel_dia_ini >= datetime.date.today():
            st.error("Recuerda que el día seleccionado no puede ser superior a la día actual")
            st.stop()
        st.info("Analizaras el día " + str(sel_dia_ini))

        sel_dia_fin = ''

    # Opciones por rango de días
    elif sel_fecha == POR_RANGO_DIA:
        sel_dia_ini = st.date_input("Seleccione el día inicial", datetime.date.today() -
                                    datetime.timedelta(days=1), key="dia_ini")
        sel_dia_fin = col3.date_input("Seleccione el día final", datetime.date.today(), key="dia_fin")

        if sel_dia_fin <= sel_dia_ini:
            st.error("Recuerda seleccionar una fecha inicial anterior a la fecha final!!!")
            st.stop()
        elif sel_dia_fin > datetime.date.today():
            st.error("Recuerda que la fecha final no puede ser superior a la fecha actual")
            st.stop()
        else:
            st.info("Analizaras un periodo de tiempo de " + str((sel_dia_fin - sel_dia_ini).days + 1) + " días.")      

# Get data
descargar = col1.button("Graficar")
if descargar is True:
    # Descargando la información
    start_time = time.time()
    with st.spinner('Descargando la información...'):
        if sel_dia_fin == '':
            sel_dia_fin = add_day(str(sel_dia_ini))

        # Consultar datos    
        inventario = get_sql(str(sel_dia_ini),str(sel_dia_fin))

        if inventario.empty:
            st.info('No hay datos del periodo seleccionado.')
            st.stop()
        
        # Resumen de tiempo
        col3.success("Consulta realizada en %s sec" % round((time.time() - start_time),2))
        
        # Organizar - Calcular datos
        inventario.sort_values(by='Fecha',inplace=True)
        inventario[RECURRENCIA] = inventario['Fecha']- inventario['Fecha'].shift()
        inventario[RECURRENCIA] = inventario[RECURRENCIA].iloc[1:].apply(lambda x: math.ceil(x.seconds/60))
    
        # Aplicar estilo a colision
        inventario['Colision'] = inventario['Colision'].apply(convertir_a_cuadro)

        # Mostrar tabla
        st.dataframe(inventario[['Fecha', 'Lavamanos', 'Onepiece', 'Pedestal', 'Tanque', 'Taza', RECURRENCIA, 'Colision']], 
                    use_container_width=True, hide_index=True,)
    
    # Crear nueva columna con el turno
    inventario['Turno'] = inventario['Fecha'].apply(asignar_turno)

    # Separar la fecha
    inventario['Dia'] = inventario['Fecha'].dt.date

    # grafica de piezas totales, filtros barra apilada
    st.plotly_chart(graficar_apilado_dias(inventario))
    
    # detalle dia : barra apilada por hora individual
    lista_dias = [sel_dia_ini + datetime.timedelta(days=d) for d in range((sel_dia_fin - sel_dia_ini).days + 1)] 
    st.selectbox('Selecccionar dia.',lista_dias)
    # st.plotly_chart(get_barra_apilada())

    #-----
    turno1 = inventario[inventario['Turno'] == 'Turno 1']
    turno2 = inventario[inventario['Turno'] == 'Turno 2']
    turno3 = inventario[inventario['Turno'] == 'Turno 3']

    # analisis por turnos: barra apilada
    st.plotly_chart(graficar_apilado_turnos([turno1,turno2,turno3]))




    
    
    
    # Frecuencia de entrada carro : boxplot por dia

# # Resumen de totales
# class_names = ['Lavamanos', 'Onepiece', 'Pedestal', 'Tanque', 'Taza']
# totales = inventario[class_names].sum(axis=0)
# totales: dict = {key: total for key, total in zip(class_names, totales)}
# st.info(f'Total: {totales}')

# # Grafica recurrencia
# fig = go.Figure(
#     go.Bar(
#         x=inventario["Fecha"],
#         y=inventario["recurrencia [Min]"],
#         name='asd'
#     )
# )

# # # Grafica 2
# frecuencia = inventario.groupby('recurrencia [Min]')[RECURRENCIA].count().reset_index(name="frecuencia")

# fig2 = go.Figure(
#     go.Bar(
#         x=frecuencia["recurrencia [Min]"],
#         y=frecuencia["frecuencia"],
#     )
# )

# fig.update_layout(title_text='Periodo de frecuencia entrada de carros')
# fig2.update_layout(title_text='Distribucion de frecuencias.')
# st.plotly_chart(fig)
# st.plotly_chart(fig2)

# # Obtener videos
# path_videos = os.listdir(path)

# # Título y selector de video
# st.title('Preview')
# video_selector = st.selectbox('Seleccione el video deseado', path_videos, index=0)
# path_video = os.path.join(path, video_selector)

# # Contenedor para el video
# prev = st.empty()

# prev.video(path_video, format="video/mp4")

# # Botón de descarga
# with open(path_video, "rb") as f:
#     video_bytes = f.read()
# st.download_button(
#     label="Descargar Video",
#     data=video_bytes,
#     file_name=video_selector,
#     mime="video/mp4",
# )