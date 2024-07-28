import streamlit as st
import seaborn as sns
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import utils.graphics_functions as gf
def main(data,variables):

    # Título de la aplicación
    st.title("Análisis Univariante")

    # Selección de variables

    selected_variables = st.multiselect("Selecciona hasta 4 variables", variables, variables[:4])

    # Selección de tipos de gráficos

    chart_types = ["Histograma", "Boxplot", "Violinplot", "Histograma Ajustado", "Q-Q Plot"]
    selected_chart_types = st.multiselect("Selecciona los tipos de gráficos", chart_types, chart_types)


    # Función para mostrar estadísticas descriptivas
    def show_statistics(df, var):
        st.write(f"### Estadísticas descriptivas de {var}")
        stats = df[var].describe()
        st.write(stats)
    show_statistics(data, selected_variables)

    # Crear gráficos para cada variable seleccionada
    for var in selected_variables:
        st.subheader(f"Análisis de {var}")



        # Crear contenedor de columnas para los gráficos
        cols = st.columns(2)
        col_index = 0

        for chart_type in selected_chart_types:
            with cols[col_index]:
                if chart_type == "Histograma":

                    fig, ax = plt.subplots()
                    sns.histplot(data[var], kde=True, ax=ax)
                    ax.set_title(f"Histograma de {var}")
                    st.pyplot(fig)

                elif chart_type == "Boxplot":
                    fig, ax = plt.subplots()
                    sns.boxplot(y=data[var], ax=ax)
                    ax.set_title(f"Boxplot de {var}")
                    st.pyplot(fig)

                elif chart_type == "Violinplot":
                    fig, ax = plt.subplots()
                    sns.violinplot(y=data[var], ax=ax)
                    ax.set_title(f"Violinplot de {var}")
                    st.pyplot(fig)

                elif chart_type == "Histograma Ajustado":
                    fig, ax = plt.subplots()
                    sns.histplot(data[var], kde=False, ax=ax)
                    mu, std = stats.norm.fit(data[var])
                    xmin, xmax = plt.xlim()
                    x = np.linspace(xmin, xmax, 100)
                    p = stats.norm.pdf(x, mu, std)
                    ax.plot(x, p, 'k', linewidth=2)
                    title = f"Histograma Ajustado de {var} (μ={mu:.2f}, σ={std:.2f})"
                    ax.set_title(title)
                    st.pyplot(fig)

                elif chart_type == "Q-Q Plot":
                    fig, ax = plt.subplots()
                    stats.probplot(data[var], dist="norm", plot=ax)
                    ax.set_title(f"Q-Q Plot de {var}")
                    st.pyplot(fig)

            # Alternar entre las dos columnas
            col_index = (col_index + 1) % 2

    # Mostrar los datos en una tabla editable
    st.subheader("Datos")
    st.dataframe(data)