# PREVISUALIZACION DE BIBLIOTECA DE VIDEOS
# ANALISIS DE INVENTARIOS DESDE DB 

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

# Settings page
st.set_page_config(page_title='Inventario H4', page_icon='./assets/logo_corona.png', layout="wide")

# ******************************************************
# Configuraciones iniciales
# ******************************************************
path: str = '00_Data/videos/inferencias'
RECURRENCIA = 'recurrencia [Min]'
POR_DIA = 'Por día'
POR_RANGO_DIA = 'Por rango de días'
clases = ['Lavamanos', 'Onepiece', 'Pedestal', 'Tanque', 'Taza']
color_por_clase = {
    'Lavamanos': 'red',
    'Onepiece': 'blue',
    'Pedestal': 'green',
    'Tanque': 'orange',
    'Taza': 'purple'
}
media_recurrencia:int = 15
limite_recurrencia:int = 30


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
        st.error("La consulta ha superado el tiempo límite.")
        st.stop
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


def graficar_apilado_dias(df):
    fig = go.Figure()
    df_grouped = df.groupby('Dia_operativo')[clases].sum().reset_index()

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


def graficar_apilado_horas(df,dia):
    fig = go.Figure()
    df_grouped = df.loc[df['Dia_operativo']==dia]
    df_grouped = df_grouped.groupby('Hora')[clases].sum().reset_index()

    for clase in clases:
        fig.add_trace(get_barra_apilada(df_grouped, clase, group_by='Hora'))

    fig.update_layout(
        barmode='stack',
        height=400,
        width=900,
        title_text="Resumen inventario dia " + str(dia) + ".",
        template='plotly_white',
        
    )

    return fig


def graficar_apilado_turnos(lista_dfs):
    fig = make_subplots(
        rows=1, cols=3,
        shared_yaxes=True,
        subplot_titles=['Turno 1','Turno 2','Turno 3']
    )

    for i, df in enumerate(lista_dfs):
        df_grouped = df.groupby('Dia_operativo')[clases].sum().reset_index()

        for clase in clases:
            fig.add_trace(get_barra_apilada(df_grouped, clase, (i==0)), row=1, col=i+1)

    fig.update_layout(
        barmode='stack',
        height=400,
        width=900,
        title_text="Resumen inventario por turno.",
        template='plotly_white'
    )

    return fig


def get_barra_apilada(df_grouped, clase, show_legend = True, group_by = 'Dia_operativo'):
    
    return go.Bar(
        x=df_grouped[group_by],
        y=df_grouped[clase],
        name=clase,
        legendgroup=clase,
        showlegend=show_legend,
        marker=dict(color=color_por_clase.get(clase, 'gray'))
    )


#############################
# Main
#############################

# Encabezado
st.image('./assets/IOT_COMPLETO.jpg')
st.divider()

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

# Descargar la información
start_time = time.time()
with st.spinner('Descargando la información...'):
    if sel_dia_fin == '':
        sel_dia_fin = add_day(str(sel_dia_ini))
    else:
        sel_dia_fin = add_day(str(sel_dia_fin))

    # Consultar datos    
    inventario = get_sql(str(sel_dia_ini),str(sel_dia_fin))

    if inventario.empty:
        st.info('No hay datos del periodo seleccionado.')
        st.stop()
    
    # Resumen de tiempo
    col3.success("Consulta realizada en %s sec" % round((time.time() - start_time),2))
    
    # Organizar - Calcular datos
    # Ordenar por fecha
    inventario.sort_values(by='Fecha',inplace=True)
    # Calcular recurrencia
    inventario[RECURRENCIA] = inventario['Fecha']- inventario['Fecha'].shift()
    inventario[RECURRENCIA] = inventario[RECURRENCIA].iloc[1:].apply(lambda x: math.ceil(x.seconds/60))
    inventario[RECURRENCIA] = inventario[RECURRENCIA].iloc[1:].apply(lambda x: media_recurrencia if x > limite_recurrencia else x)
    # Aplicar estilo a colision
    inventario['Colision'] = inventario['Colision'].apply(lambda x: '🟥' if x == 1 else '🟩')
    # Crear nueva columna con el turno
    inventario['Turno'] = inventario['Fecha'].apply(asignar_turno)
    # Separar la fecha y hora
    # 'Día_operativo' ajustado a las 6:00 AM
    inventario['Dia_operativo'] = (inventario['Fecha'] - pd.Timedelta(hours=6)).dt.date
    inventario['Hora'] = inventario['Fecha'].dt.hour
    # inventario['Hora'] = inventario['Fecha'].dt.strftime('%Y-%m-%d %H')

    # Mostrar tabla
    st.dataframe(inventario[['Fecha', 'Dia_operativo', 'Lavamanos', 'Onepiece', 'Pedestal', 'Tanque', 'Taza', RECURRENCIA, 'Colision']], 
                use_container_width=True, hide_index=True,)


with st.expander('Videos', expanded=False):
    # Obtener videos
    path_videos = os.listdir(path)
    # seleccionar video
    video_selector = st.selectbox('Seleccione el video deseado', path_videos, index=0)
    path_video = os.path.join(path, video_selector)
    # Contenedor para el video
    prev = st.empty()
    # mostrar
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

with st.expander('Graficas', expanded=False):
    # grafica de piezas totales, filtros barra apilada
    st.plotly_chart(graficar_apilado_dias(inventario))

    # detalle dia : barra apilada por hora individual
    lista_dias = inventario['Dia_operativo'].unique()
    dia_ampliado = st.selectbox('Selecccionar dia.',lista_dias)

    st.plotly_chart(graficar_apilado_horas(inventario, dia_ampliado))

    #-----
    turno1 = inventario[inventario['Turno'] == 'Turno 1']
    turno2 = inventario[inventario['Turno'] == 'Turno 2']
    turno3 = inventario[inventario['Turno'] == 'Turno 3']

    # analisis por turnos: barra apilada
    st.plotly_chart(graficar_apilado_turnos([turno1,turno2,turno3]))

    # Frecuencia de entrada carro : boxplot por dia
    box_recurrencia_dia = go.Figure()
    box_recurrencia_dia.add_trace(go.Box(
        x=inventario['Dia_operativo'],
        y=inventario[RECURRENCIA],
    ))
    box_recurrencia_dia.update_layout(
            barmode='stack',
            height=400,
            width=900,
            title_text="Frecuencia entrada de carros.",
            template='plotly_white',
            yaxis=dict(
                title=dict(
                    text=RECURRENCIA)
            ),    
    )
    st.plotly_chart(box_recurrencia_dia)