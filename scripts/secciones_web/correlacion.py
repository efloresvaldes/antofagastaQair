import streamlit as st
import seaborn as sns
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main(data, variables):
    # Título de la aplicación
    st.title("Análisis de Correlación")

    # Selección de variables
    selected_variables = st.multiselect("Selecciona variables para el análisis", variables, variables[:4])

    # Selección de tipos de gráficos
    chart_types = ["Heatmap", "Pairplot", "Scatter Matrix"]
    selected_chart_types = st.multiselect("Selecciona los tipos de gráficos", chart_types, chart_types[:2])

    # Función para mostrar la matriz de correlación
    def show_correlation_matrix(df, vars):
        corr_matrix = df[vars].corr()
        st.write("### Matriz de Correlación")
        st.write(corr_matrix)

    # Crear gráficos de correlación para las variables seleccionadas
    if len(selected_variables) >= 2:
        st.subheader("Análisis de Correlación")

        show_correlation_matrix(data, selected_variables)

        for chart_type in selected_chart_types:
            if chart_type == "Heatmap":
                fig, ax = plt.subplots()
                sns.heatmap(data[selected_variables].corr(), annot=True, cmap='coolwarm', ax=ax)
                ax.set_title("Heatmap de la Matriz de Correlación")
                st.pyplot(fig)

            elif chart_type == "Pairplot":
                fig = sns.pairplot(data[selected_variables])
                st.pyplot(fig)

            elif chart_type == "Scatter Matrix":
                fig = px.scatter_matrix(data[selected_variables], title="Scatter Matrix")
                st.plotly_chart(fig)

    # Mostrar los datos en una tabla editable
    st.subheader("Conjunto de Datos")
    st.dataframe(data)