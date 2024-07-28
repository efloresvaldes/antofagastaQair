
import streamlit as st
def main():
    st.title('Guía de Uso de la Aplicación de Calidad del Aire')

    st.header('¿Cómo usar esta aplicación?')
    st.write("""
    Esta aplicación te permite analizar datos de calidad del aire. Puedes utilizarla para visualizar métricas importantes y obtener información sobre los niveles de contaminantes en el aire. Aquí tienes una guía rápida para comenzar:

    1. **Carga de Datos**: Primero, carga tu archivo CSV con los datos de calidad del aire.
    2. **Visualización**: Usa las herramientas de visualización para explorar las concentraciones de contaminantes como SO₂, MP10 y MP2.5.
    3. **Análisis**: Realiza análisis descriptivos y gráficos para comprender mejor los datos.
    4. **Modelos**: Puedes entrenar modelos de predicción basados en los datos cargados.
    5. **Conclusiones**: Obtén métricas de evaluación para entender el rendimiento de los modelos.

    Para obtener más información sobre cómo usar cada función específica, consulta la documentación o el tutorial proporcionado.
    """)

    st.header('Imágenes de Calidad del Aire')
    st.write("""
    A continuación, se muestran algunas imágenes representativas de los niveles de calidad del aire en diferentes escenarios. Estas imágenes pueden ayudarte a visualizar las concentraciones de contaminantes y su impacto en la calidad del aire.
    """)

    # Mostrar imágenes relacionadas con la calidad del aire
    st.image('path_to_your_image1.png', caption='Ejemplo de concentración de SO₂', use_column_width=True)
    st.image('path_to_your_image2.png', caption='Ejemplo de concentración de MP10', use_column_width=True)
    st.image('path_to_your_image3.png', caption='Ejemplo de concentración de MP2.5', use_column_width=True)

    st.header('Contacto y Soporte')
    st.write("""
    Si tienes alguna pregunta o necesitas soporte adicional, no dudes en ponerte en contacto con nuestro equipo de soporte:

    - **Email**: soporte@calidadaire.com
    - **Teléfono**: +123 456 7890
    - **Sitio Web**: [www.calidadaire.com](http://www.calidadaire.com)
    """)
