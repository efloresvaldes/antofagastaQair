import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from typeguard import typechecked
from typing import List, Dict, Union

@typechecked
def entrenar_modelo_regresion(
    df: pd.DataFrame,
    target: str,
    features: List[str]
) -> LinearRegression:
    """
    Entrena un modelo de regresión lineal (simple o múltiple) basado en las características dadas.

    Parámetros:
    df (pd.DataFrame): DataFrame que contiene los datos.
    target (str): El nombre de la columna objetivo (variable dependiente).
    features (List[str]): Lista de nombres de columnas de características (variables independientes).

    Devuelve:
    LinearRegression: El modelo de regresión entrenado.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("El parámetro 'df' debe ser un DataFrame de pandas.")
    if not isinstance(target, str):
        raise TypeError("El parámetro 'target' debe ser un string.")
    if not isinstance(features, list) or not all(isinstance(f, str) for f in features):
        raise TypeError("El parámetro 'features' debe ser una lista de strings.")

    if target not in df.columns:
        raise ValueError(f"La columna objetivo '{target}' no existe en el DataFrame.")
    if not all(f in df.columns for f in features):
        raise ValueError("Una o más columnas de características no existen en el DataFrame.")

    X = df[features]
    y = df[target]

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Crear el modelo de regresión lineal
    model = LinearRegression()

    # Entrenar el modelo
    model.fit(X_train, y_train)

    # Predecir en el conjunto de prueba
    y_pred = model.predict(X_test)

    # Evaluar el modelo
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")

    # Visualizar los resultados si es un modelo de regresión simple
    if len(features) == 1:
        plt.scatter(X_test, y_test, color='blue')
        plt.plot(X_test, y_pred, color='red', linewidth=2)
        plt.xlabel(features[0])
        plt.ylabel(target)
        plt.title('Regresión Lineal Simple')
        plt.show()

    return model


@typechecked
def entrenar_modelo_randomforest(
    df: pd.DataFrame,
    target: str,
    features: List[str]
) -> RandomForestRegressor:
    """
    Entrena un modelo de RandomForestRegressor basado en las características dadas.

    Parámetros:
    df (pd.DataFrame): DataFrame que contiene los datos.
    target (str): El nombre de la columna objetivo (variable dependiente).
    features (List[str]): Lista de nombres de columnas de características (variables independientes).

    Devuelve:
    RandomForestRegressor: El modelo de RandomForestRegressor entrenado.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("El parámetro 'df' debe ser un DataFrame de pandas.")
    if not isinstance(target, str):
        raise TypeError("El parámetro 'target' debe ser un string.")
    if not isinstance(features, list) or not all(isinstance(f, str) for f in features):
        raise TypeError("El parámetro 'features' debe ser una lista de strings.")

    if target not in df.columns:
        raise ValueError(f"La columna objetivo '{target}' no existe en el DataFrame.")
    if not (f in df.columns for f in features):
        raise ValueError("Una o más columnas de características no existen en el DataFrame.")

    X = df[features]
    y = df[target]

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Crear el modelo de RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Realizar predicciones
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Error Cuadrático Medio (MSE): {mse}')
    print(f'Error Absoluto Medio (MAE): {mae}')
    print(f'Coeficiente de Determinación (R²): {r2}')

    return model


@typechecked
def regresion_lineal_polinomica(
    data: pd.DataFrame,
    características: List[str],
    objetivo: str,
    grado: int = 2,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Union[np.ndarray, float]]:
    """
    Realiza una regresión lineal polinómica en el conjunto de datos proporcionado.

    Parámetros:
    - data (pd.DataFrame): DataFrame que contiene las características y la variable objetivo.
    - características (List[str]): Lista de nombres de las columnas a usar como características.
    - objetivo (str): Nombre de la columna que contiene la variable objetivo.
    - grado (int): Grado del polinomio para transformar las características (por defecto 2).
    - test_size (float): Proporción del conjunto de datos a usar para pruebas (por defecto 0.2).
    - random_state (int): Semilla para la reproducibilidad (por defecto 42).

    Retorna:
    - dict: Un diccionario con los resultados del modelo (coeficientes, intercepto, métricas de evaluación).
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("El parámetro 'data' debe ser un DataFrame de pandas.")
    if not isinstance(características, list) or not all(isinstance(c, str) for c in características):
        raise TypeError("El parámetro 'características' debe ser una lista de strings.")
    if not isinstance(objetivo, str):
        raise TypeError("El parámetro 'objetivo' debe ser un string.")
    if not isinstance(grado, int) or grado <= 0:
        raise TypeError("El parámetro 'grado' debe ser un entero positivo.")
    if not isinstance(test_size, float) or not (0 < test_size < 1):
        raise TypeError("El parámetro 'test_size' debe ser un número flotante entre 0 y 1.")
    if not isinstance(random_state, int):
        raise TypeError("El parámetro 'random_state' debe ser un entero.")

    if objetivo not in data.columns:
        raise ValueError(f"La columna objetivo '{objetivo}' no existe en el DataFrame.")
    if not all(c in data.columns for c in características):
        raise ValueError("Una o más columnas de características no existen en el DataFrame.")

    X = data[características]
    y = data[objetivo]

    # Crear características polinómicas
    poly = PolynomialFeatures(degree=grado)
    X_poly = poly.fit_transform(X)

    # Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=test_size, random_state=random_state)

    # Crear y entrenar el modelo
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Realizar predicciones
    y_pred = model.predict(X_test)

    # Calcular métricas de evaluación
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Error Cuadrático Medio (MSE): {mse}')
    print(f'Error Absoluto Medio (MAE): {mae}')
    print(f'Coeficiente de Determinación (R2): {r2}')

    return {
        'coeficientes': model.coef_,
        'intercepto': model.intercept_,
        'mse': mse,
        'mae': mae,
        'r2': r2
    }
