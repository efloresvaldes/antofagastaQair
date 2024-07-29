import streamlit as st


def main():


    st.write("""
    Esta aplicación te permite analizar datos de calidad del aire. Puedes utilizarla para visualizar métricas importantes y obtener información sobre los niveles de contaminantes en el aire. Esta aplicación trabaja con un dataset previamente limpiado obtenido del sitio web 
    https://sinca.mma.gob.cl/index.php/estacion/index/id/259

    1. **Análisis Univariante**: Se hace un análisis univariante sobre las variables Temperatura y los contaminantes SO2, MP10 y MP2.5; donde se muestran gráficos y estadísticas descriptivas, para su comprensión. 
    2. **Análisis Bivariante**: Se visualizan gráficos para representar par de variables seleccionadas por el usuario.
    3. **Análisis de Correlación**: Se visualiza la matriz de correlación y dos gráficos para mostrar como las variables seleccionadas están correlacionadas 
    


   
    """)

    st.header('Contacto y Soporte')
    st.write("""
    Si tienes alguna pregunta o necesitas soporte adicional, no dudes en ponerte en contacto con nuestro equipo de soporte:

    - **Email**: evaflores583@gmail.com
   
    """)
