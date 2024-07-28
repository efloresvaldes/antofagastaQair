import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Optional, Union
from typeguard import typechecked

@typechecked
def eliminar_duplicados(
    df: pd.DataFrame,
    columnas: Optional[List[str]] = None,
    mantener: str = 'first'
) -> pd.DataFrame:
    """
    Elimina elementos duplicados en un DataFrame de pandas.

    Parámetros:
    df (pandas.DataFrame): El DataFrame del que se eliminarán los duplicados.
    columnas (list[str], opcional): Lista de columnas según las cuales se deben identificar los duplicados.
                                     Si no se proporciona, se utilizarán todas las columnas.
    mantener (str, opcional): Qué duplicado mantener:
                              'first' - Mantener el primer duplicado (default).
                              'last' - Mantener el último duplicado.
                               False - Eliminar todos los duplicados.

    Retorna:
    pandas.DataFrame: El DataFrame sin los elementos duplicados.
    """

    # Verificación de tipos
    if not isinstance(df, pd.DataFrame):
        raise TypeError("El parámetro 'df' debe ser un DataFrame de pandas.")
    if columnas is not None and not isinstance(columnas, list):
        raise TypeError("El parámetro 'columnas' debe ser una lista de strings o None.")
    if columnas is not None and not all(isinstance(col, str) for col in columnas):
        raise TypeError("Todos los elementos en 'columnas' deben ser strings.")
    if not isinstance(mantener, str) or mantener not in {'first', 'last', False}:
        raise ValueError("El parámetro 'mantener' debe ser 'first', 'last' o False.")

    # Verificar si hay duplicados
    duplicados = df.duplicated().sum()
    if duplicados == 0:
        return df

    # Eliminar duplicados
    try:
        if columnas:
            df_sin_duplicados = df.drop_duplicates(subset=columnas, keep=mantener)
        else:
            df_sin_duplicados = df.drop_duplicates(keep=mantener)
    except Exception as e:
        raise RuntimeError(f"Error al eliminar duplicados: {e}")

    return df_sin_duplicados


@typechecked
def imputar_valores_faltantes(
    df: pd.DataFrame,
    estrategia: str = 'media',
    valor: Optional[Union[int, float]] = None,
    columnas: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Imputa valores faltantes en un DataFrame de pandas.

    Parámetros:
    df (pandas.DataFrame): El DataFrame en el que se imputarán los valores faltantes.
    estrategia (str, opcional): La estrategia de imputación ('media', 'mediana', 'moda', 'constante').
                                Default es 'media'.
    valor (opcional): El valor constante a utilizar si la estrategia es 'constante'.
    columnas (list[str], opcional): Lista de columnas a imputar. Si no se proporciona, se imputarán todas las columnas.

    Retorna:
    pandas.DataFrame: El DataFrame con los valores faltantes imputados.
    """
    if columnas is None:
        columnas = df.columns.tolist()

    if not isinstance(columnas, list):
        raise TypeError("El parámetro 'columnas' debe ser una lista de strings o None.")
    if not all(isinstance(col, str) for col in columnas):
        raise TypeError("Todos los elementos en 'columnas' deben ser strings.")
    if not isinstance(estrategia, str) or estrategia not in {'media', 'mediana', 'moda', 'constante'}:
        raise ValueError("Estrategia no reconocida. Utilice 'media', 'mediana', 'moda' o 'constante'.")
    if estrategia == 'constante' and valor is None:
        raise ValueError("Debe proporcionar un valor constante para la estrategia 'constante'.")

    for columna in columnas:
        if columna not in df.columns:
            raise ValueError(f"La columna '{columna}' no existe en el DataFrame.")

        if estrategia == 'media':
            impute_value = df[columna].mean()
        elif estrategia == 'mediana':
            impute_value = df[columna].median()
        elif estrategia == 'moda':
            impute_value = df[columna].mode()[0]
        elif estrategia == 'constante':
            impute_value = valor

        df[columna] = df[columna].fillna(impute_value)

    return df


@typechecked
def plot_outliers(
    df: pd.DataFrame,
    columna: str,
    title: str,
    eje: plt.Axes
) -> None:
    """
    Grafica los outliers en una columna de un DataFrame usando un boxplot.

    Parámetros:
    df (pandas.DataFrame): El DataFrame que contiene los datos.
    columna (str): El nombre de la columna para la cual se graficarán los outliers.
    title (str): Título del gráfico.
    eje (matplotlib.axes.Axes): Eje en el que se graficará el boxplot.

    Retorna:
    None
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("El parámetro 'df' debe ser un DataFrame de pandas.")
    if not isinstance(columna, str):
        raise TypeError("El parámetro 'columna' debe ser un string.")
    if columna not in df.columns:
        raise ValueError(f"La columna '{columna}' no existe en el DataFrame.")
    if not isinstance(title, str):
        raise TypeError("El parámetro 'title' debe ser un string.")
    if not isinstance(eje, plt.Axes):
        raise TypeError("El parámetro 'eje' debe ser una instancia de matplotlib.axes.Axes.")

    try:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x=columna, ax=eje)
        eje.set_title(title)
        eje.set_xlabel(columna)
    except Exception as e:
        raise RuntimeError(f"Error al graficar los outliers: {e}")


@typechecked
def eliminar_outliers(
    df: pd.DataFrame,
    columna: str
) -> pd.DataFrame:
    """
    Elimina outliers en una columna de un DataFrame usando el rango intercuartílico (IQR).

    Parámetros:
    df (pandas.DataFrame): El DataFrame que contiene los datos.
    columna (str): El nombre de la columna de la cual se eliminarán los outliers.

    Retorna:
    pandas.DataFrame: El DataFrame sin outliers en la columna especificada.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("El parámetro 'df' debe ser un DataFrame de pandas.")
    if not isinstance(columna, str):
        raise TypeError("El parámetro 'columna' debe ser un string.")
    if columna not in df.columns:
        raise ValueError(f"La columna '{columna}' no existe en el DataFrame.")

    try:
        Q1 = df[columna].quantile(0.25)
        Q3 = df[columna].quantile(0.75)
        IQR = Q3 - Q1

        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR

        df_sin_outliers = df[(df[columna] >= limite_inferior) & (df[columna] <= limite_superior)]
    except Exception as e:
        raise RuntimeError(f"Error al eliminar outliers: {e}")

    return df_sin_outliers


@typechecked
def sparseDates(
    df: pd.DataFrame,
    columna_fecha: str
) -> pd.DataFrame:
    """
    Separa una columna fecha en sus componentes.

    Parámetros:
    df (pandas.DataFrame): El DataFrame que contiene los datos.
    columna_fecha (str): El nombre de la columna que pueda convertirse en una fecha.

    Retorna:
    pandas.DataFrame: El DataFrame con las columnas de componentes de fecha.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("El parámetro 'df' debe ser un DataFrame de pandas.")
    if not isinstance(columna_fecha, str):
        raise TypeError("El parámetro 'columna_fecha' debe ser un string.")
    if columna_fecha not in df.columns:
        raise ValueError(f"La columna '{columna_fecha}' no existe en el DataFrame.")

    try:
        df['fecha'] = pd.to_datetime(df[columna_fecha])
        df['year'] = df['fecha'].dt.year
        df['mes'] = df['fecha'].dt.month
        df['day'] = df['fecha'].dt.day
    except Exception as e:
        raise RuntimeError(f"Error al separar la columna fecha: {e}")

    return df
