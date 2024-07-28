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


    # Preparar los datos
    data['Fecha'] = pd.to_datetime(data['Fecha'])
    data['DayOfYear'] = data['Fecha'].dt.dayofyear

    X = data[['Temperatura Media', 'DayOfYear']]
    y_so2 = data['SO2']
    y_mp10 = data['MP10']
    y_mp25 = data['MP2.5']


    # Entrenar modelos
    def train_model(X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model


    model_so2 = train_model(X, y_so2)
    model_mp10 = train_model(X, y_mp10)
    model_mp25 = train_model(X, y_mp25)


    # Predecir valores futuros
    def predict_future(model, temp, start_date, days):
        predictions = []
        for i in range(days):
            future_date = start_date + timedelta(days=i)
            day_of_year = future_date.timetuple().tm_yday
            pred = model.predict([[temp, day_of_year]])[0]
            predictions.append((future_date, pred))
        return predictions


    # Calcular el Índice de Calidad del Aire (QAI)
    def calculate_qai(so2, mp10, mp25):
        # Simplificación del cálculo de QAI
        qai_so2 = np.clip(so2 / 2, 0, 500)
        qai_mp10 = np.clip(mp10 / 2, 0, 500)
        qai_mp25 = np.clip(mp25 / 2, 0, 500)
        return max(qai_so2, qai_mp10, qai_mp25)


    # Crear la aplicación Streamlit
    st.title("Predicción de la Calidad del Aire")

    # Input de temperatura y fecha
    temp = st.number_input("Temperatura media (°C)", value=20.0)
    start_date = st.date_input("Fecha de inicio", value=datetime.now())

    # Mostrar predicciones
    days_to_predict = [1, 3, 5, 7]
    results = []
    for days in days_to_predict:
        pred_so2 = predict_future(model_so2, temp, start_date, days)
        pred_mp10 = predict_future(model_mp10, temp, start_date, days)
        pred_mp25 = predict_future(model_mp25, temp, start_date, days)

        for i in range(days):
            qai = calculate_qai(pred_so2[i][1], pred_mp10[i][1], pred_mp25[i][1])
            results.append({
                'Día': pred_so2[i][0],
                'SO2': pred_so2[i][1],
                'MP10': pred_mp10[i][1],
                'MP2.5': pred_mp25[i][1],
                'QAI': qai
            })

    results_df = pd.DataFrame(results)
    st.write(results_df)

    # Visualizar resultados
    st.subheader("Visualización de las Predicciones")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(results_df['Día'], results_df['SO2'], label='SO2')
    ax.plot(results_df['Día'], results_df['MP10'], label='MP10')
    ax.plot(results_df['Día'], results_df['MP2.5'], label='MP2.5')
    ax.plot(results_df['Día'], results_df['QAI'], label='QAI', linestyle='--')
    ax.legend()
    plt.xlabel("Fecha")
    plt.ylabel("Concentración / QAI")
    plt.title("Predicciones de SO2, MP10, MP2.5 y QAI")
    st.pyplot(fig)

