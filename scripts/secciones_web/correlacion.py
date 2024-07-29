import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import utils.cleaning_data_functions as cdf

def main(data,variables):


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

        # Crear contenedor de columnas para los gráficos
        cols = st.columns(2)
        col_index = 0

        for chart_type in selected_chart_types:
            with cols[col_index]:
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

            # Alternar entre las dos columnas
            col_index = (col_index + 1) % 2

    # Mostrar los datos en una tabla editable
    st.subheader("Datos")
    st.dataframe(data)