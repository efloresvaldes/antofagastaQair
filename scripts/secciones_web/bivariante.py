import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def main(data, variables):
    # Título de la aplicación
    st.title("Análisis Bivariante")

    # Selección de variables
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

        for chart_type in selected_chart_types:
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

    # Mostrar los datos en una tabla editable
    st.subheader("Conjunto de Datos")
    st.dataframe(data)