import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from typeguard import typechecked


@typechecked
def plot_histograma(df: pd.DataFrame, columna: str):
    """
    Dibuja un histograma para la columna especificada en el DataFrame.

    Parámetros:
    df (pd.DataFrame): El DataFrame que contiene los datos.
    columna (str): El nombre de la columna para la cual se creará el histograma.

    Retorna:
    None
    """
    if columna not in df.columns:
        raise ValueError(f"La columna '{columna}' no existe en el DataFrame.")

    plt.figure(figsize=(10, 6))
    sns.histplot(df[columna], kde=False, bins=30)
    plt.title(f'Histograma de {columna}')
    plt.xlabel(columna)
    plt.ylabel('Frecuencia')
    plt.show()


@typechecked
def plot_boxplot(df: pd.DataFrame, columna: str):
    """
    Dibuja un boxplot para la columna especificada en el DataFrame.

    Parámetros:
    df (pd.DataFrame): El DataFrame que contiene los datos.
    columna (str): El nombre de la columna para la cual se creará el boxplot.

    Retorna:
    None
    """
    if columna not in df.columns:
        raise ValueError(f"La columna '{columna}' no existe en el DataFrame.")

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x=columna)
    plt.title(f'Boxplot de {columna}')
    plt.xlabel(columna)
    plt.show()


@typechecked
def plot_violinplot(df: pd.DataFrame, columna: str):
    """
    Dibuja un violinplot para la columna especificada en el DataFrame.

    Parámetros:
    df (pd.DataFrame): El DataFrame que contiene los datos.
    columna (str): El nombre de la columna para la cual se creará el violinplot.

    Retorna:
    None
    """
    if columna not in df.columns:
        raise ValueError(f"La columna '{columna}' no existe en el DataFrame.")

    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df, x=columna)
    plt.title(f'Violinplot de {columna}')
    plt.xlabel(columna)
    plt.show()


@typechecked
def plot_histograma_ajustado(df: pd.DataFrame, columna: str):
    """
    Dibuja un histograma ajustado con una línea de ajuste KDE para la columna especificada en el DataFrame.

    Parámetros:
    df (pd.DataFrame): El DataFrame que contiene los datos.
    columna (str): El nombre de la columna para la cual se creará el histograma ajustado.

    Retorna:
    None
    """
    if columna not in df.columns:
        raise ValueError(f"La columna '{columna}' no existe en el DataFrame.")

    plt.figure(figsize=(10, 6))
    sns.histplot(df[columna], kde=True, bins=30)
    plt.title(f'Histograma Ajustado de {columna}')
    plt.xlabel(columna)
    plt.ylabel('Frecuencia')
    plt.show()


@typechecked
def plot_qq_plot(df: pd.DataFrame, columna: str):
    """
    Dibuja un Q-Q plot para la columna especificada en el DataFrame.

    Parámetros:
    df (pd.DataFrame): El DataFrame que contiene los datos.
    columna (str): El nombre de la columna para la cual se creará el Q-Q plot.

    Retorna:
    None
    """
    if columna not in df.columns:
        raise ValueError(f"La columna '{columna}' no existe en el DataFrame.")

    data = df[columna].dropna()
    if data.empty:
        raise ValueError(f"La columna '{columna}' no tiene datos válidos para generar el Q-Q plot.")

    plt.figure(figsize=(10, 6))
    stats.probplot(data, dist="norm", plot=plt)
    plt.title(f'Q-Q Plot de {columna}')
    plt.show()
