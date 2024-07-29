import streamlit as st
import seaborn as sns
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import utils.graphics_functions as gf


def main(data, variables):
    # Título de la aplicación
    st.title("Análisis Univariante")

    # Selección de variables
    selected_variables = st.multiselect("Selecciona las variables", variables)

    # Selección de tipos de gráficos
    chart_types = ["Histograma", "Boxplot", "Violinplot", "Histograma Ajustado", "Q-Q Plot"]
    selected_chart_types = st.multiselect("Selecciona los tipos de gráficos", chart_types, chart_types)

    # Función para mostrar estadísticas descriptivas
    def show_statistics(df, var):
        st.write(f"### Estadísticas descriptivas de {var}")
        stats_descript = df[var].describe()
        st.write(stats_descript)

    if selected_variables:
        show_statistics(data, selected_variables)

    # Crear gráficos para cada variable seleccionada
    for var in selected_variables:
        st.subheader(f"Análisis de {var}")

        # Calcular el número de filas necesarias para los subplots
        num_plots = len(selected_chart_types)
        num_cols = 2
        num_rows = (num_plots + num_cols - 1) // num_cols  # División entera redondeando hacia arriba

        # Crear la figura y los ejes
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 5 * num_rows))
        axes = axes.flatten()  # Asegurarse de que axes es una matriz unidimensional

        for i, chart_type in enumerate(selected_chart_types):
            ax = axes[i]
            if chart_type == "Histograma":
                sns.histplot(data[var], kde=True, ax=ax)
                ax.set_title(f"Histograma de {var}")

            elif chart_type == "Boxplot":
                sns.boxplot(y=data[var], ax=ax)
                ax.set_title(f"Boxplot de {var}")

            elif chart_type == "Violinplot":
                sns.violinplot(y=data[var], ax=ax)
                ax.set_title(f"Violinplot de {var}")

            elif chart_type == "Histograma Ajustado":
                sns.histplot(data[var], kde=False, ax=ax)
                mu, std = stats.norm.fit(data[var])
                xmin, xmax = ax.get_xlim()
                x = np.linspace(xmin, xmax, 100)
                p = stats.norm.pdf(x, mu, std)
                ax.plot(x, p, 'k', linewidth=2)
                title = f"Histograma Ajustado de {var} (μ={mu:.2f}, σ={std:.2f})"
                ax.set_title(title)

            elif chart_type == "Q-Q Plot":
                stats.probplot(data[var], dist="norm", plot=ax)
                ax.set_title(f"Q-Q Plot de {var}")

        # Eliminar cualquier eje no utilizado
        if len(axes) > len(selected_chart_types):
            for j in range(len(selected_chart_types), len(axes)):
                fig.delaxes(axes[j])

        st.pyplot(fig)

    # Mostrar los datos en una tabla editable
    st.subheader("Conjunto de Datos")
    st.dataframe(data)

