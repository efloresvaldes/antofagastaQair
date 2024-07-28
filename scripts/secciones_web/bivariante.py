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

def main(data):
    # Título de la aplicación
    st.title("Análisis Bivariante")

    # Selección de variables
    variables = data.columns.tolist()
    selected_variables = st.multiselect("Selecciona dos variables para el análisis", variables, variables[:2])

    # Selección de tipos de gráficos
    chart_types = ["Scatter Plot", "Line Plot", "Histogram 2D", "Density Plot", "Regresión Lineal"]
    selected_chart_types = st.multiselect("Selecciona los tipos de gráficos", chart_types, chart_types[:2])

    # Función para mostrar la regresión lineal
    def plot_regression(x, y, data):
        fig = px.scatter(data, x=x, y=y, title=f"Regresión Lineal de {x} vs {y}")
        X = sm.add_constant(data[x])  # Agregar una constante para la intersección
        model = sm.OLS(data[y], X).fit()
        predictions = model.predict(X)
        fig.add_trace(go.Scatter(x=data[x], y=predictions, mode='lines', name='Regresión Lineal'))

        # Mostrar estadísticas del modelo
        st.write(f"### Resultados de la Regresión Lineal entre {x} y {y}")
        st.write(model.summary())

        st.plotly_chart(fig)

    # Crear gráficos bivariantes para las variables seleccionadas
    if len(selected_variables) == 2:
        var1, var2 = selected_variables
        st.subheader(f"Análisis Bivariante entre {var1} y {var2}")

        # Crear contenedor de columnas para los gráficos
        cols = st.columns(2)
        col_index = 0

        for chart_type in selected_chart_types:
            with cols[col_index]:
                if chart_type == "Scatter Plot":
                    fig = px.scatter(data, x=var1, y=var2, title=f"Scatter Plot de {var1} vs {var2}")
                    st.plotly_chart(fig)

                elif chart_type == "Line Plot":
                    fig = px.line(data, x=var1, y=var2, title=f"Line Plot de {var1} vs {var2}")
                    st.plotly_chart(fig)

                elif chart_type == "Histogram 2D":
                    fig = px.density_heatmap(data, x=var1, y=var2, title=f"Histogram 2D de {var1} vs {var2}")
                    st.plotly_chart(fig)

                elif chart_type == "Density Plot":
                    fig = px.density_contour(data, x=var1, y=var2, title=f"Density Plot de {var1} vs {var2}")
                    st.plotly_chart(fig)

                elif chart_type == "Regresión Lineal":
                    plot_regression(var1, var2, data)

            # Alternar entre las dos columnas
            col_index = (col_index + 1) % 2

    # Mostrar los datos en una tabla editable
    st.subheader("Datos")
    st.dataframe(data)