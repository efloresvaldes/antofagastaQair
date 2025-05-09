import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd

import secciones_web.home as home_section
import secciones_web.univariante as univariante_section
import secciones_web.bivariante as bivariante_section
import secciones_web.correlacion as correlacion_section
import secciones_web.qair as qair_section
import utils.cleaning_data_functions as cdf

df = pd.read_csv('../data/qair.csv', delimiter=';')
#Cambiando el nmbre de las columnas del dataframe para mejor comprensión
df.rename(columns={"RV1": "SO2", "RV2": "MP10","RV3": "MP2.5"}, inplace=True)

variables_a_considerar = ['Temperatura Media', 'SO2', 'MP10', 'MP2.5']


# Ajustando el ancho del layout
st.set_page_config(layout="wide")

# Título de la aplicación
st.title("Datos de Calidad del Aire en Antofagasta")

# Crear el menú en la barra lateral
with st.sidebar:
    #st.logo('assets/logo_qair_rect.png', use_column_width='auto')  # Insertar la imagen del logo
    selected = option_menu(
        "Calidad del Aire",
        ["Inicio", "Análisis Univariante", "Análisis Bivariante", "Análisis de Correlación"],
        icons=["house", "bar-chart", "file-bar-graph", "diagram-3"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "5!important", "background-color": "#f0f2f6"},
            "icon": {"color": "black", "font-size": "20px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "0px",
                "color": "black",
            },
            "nav-link-selected": {"background-color": "#77c4fe"},
        },
    )

# Mostrar el contenido correspondiente a la opción seleccionada
if selected == "Inicio":
    home_section.main()
elif selected == "Análisis Univariante":
    univariante_section.main(df, variables_a_considerar)
elif selected == "Análisis Bivariante":
    bivariante_section.main(df, variables_a_considerar)
elif selected == "Análisis de Correlación":
    correlacion_section.main(df, variables_a_considerar)
    """
    elif selected == "Estimación QAir":
    df = cdf.sparseDates(df, 'Fecha')
    qair_section.main(df)
    """

