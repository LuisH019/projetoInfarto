import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import datetime as dt

df = pd.read_csv('bhd.csv')

df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')

df['Change %'] = df['Change %'].str.replace('%', '').astype(float)

df['Price'] = df['Price'].str.replace(',', '').astype(float)

def convert_volume(vol):
    if 'K' in vol:
        return float(vol.replace('K', '')) * 1000
    elif 'M' in vol:
        return float(vol.replace('M', '')) * 1000000
    elif 'B' in vol:
        return float(vol.replace('B', '')) * 1000000000
    return float(vol)

df['Vol.'] = df['Vol.'].apply(convert_volume)

df = df.sort_values('Date').reset_index(drop=True)

dfFiltered = df[df['Date'] > '2024-01-01']

line_chart = px.line(df, x='Date', y='Price', labels={'Price': 'Preço (USD)', 'Date': 'Data'},
              title="Preço do Bitcoin ao longo do tempo")
st.plotly_chart(line_chart)


# Seleção de Recursos e Alvo
X = df[['Date']].astype('int64') / 1e9
y = df['Price']

# Divisão em Conjuntos de Treino e Teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criação e Treinamento do Modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Previsão
y_pred = model.predict(X_test)

# Avaliação do Modelo
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)


st.write(f"Mean Absolute Error (MAE): {mae}")
st.write(f"Mean Squared Error (MSE): {mse}")
st.write(f"Root Mean Squared Error (RMSE): {rmse}")
st.write(f"R²: {r2}")

results = pd.DataFrame({
    'Date': pd.to_datetime(X_test['Date'], unit='s').values,
    'Real Price': y_test.values,
    'Predicted Price': y_pred
})

results = results.sort_values('Date').reset_index(drop=True)

line_chart = px.line(results, x='Date', y=['Real Price', 'Predicted Price'], 
              labels={'value': 'Preço (USD)', 'Date': 'Data'},
              title="Preço real vs Previsão do Bitcoin")
st.plotly_chart(line_chart)