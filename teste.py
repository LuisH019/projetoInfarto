import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from predict import predict

# Carregar o dataset
df = pd.read_csv('C:/Users/Windows 10 Pro/Desktop/corazaum/projetoInfarto/heart_2020_cleaned_copy.csv')
df = df.drop(['Race', 'DiffWalking', 'GenHealth'], axis=1)  # Remover colunas não utilizadas

# Converter para um array NumPy de floats e transpor
dfNumpy = df.astype(float).to_numpy().T

# Parâmetros iniciais
rows, cols = dfNumpy.shape
numEpocas = 100
eta = 0.05
m = rows - 1
N = 30  # Número de neurônios na camada escondida
L = 1

# Vetor de classificação desejada (target)
d = dfNumpy[0]

# Inicializar os pesos aleatoriamente
W1 = np.random.random((N, m + 1))
W2 = np.random.random((L, N + 1))

# Treinamento da rede neural
for i in range(numEpocas):
    E = np.zeros(cols)  # Array para armazenar os erros
    for j in range(cols):
        # Entrada com viés
        Xb = np.hstack((1, dfNumpy[1:, j]))

        # Forward pass
        o1 = sigmoid(W1.dot(Xb))
        o1b = np.insert(o1, 0, 1)  # Inclui o bias
        Y = sigmoid(W2.dot(o1b))

        # Erro e retropropagação
        e = d[j] - Y
        delta2 = e * Y * (1 - Y)
        delta1 = delta2 * W2[:, 1:] * o1 * (1 - o1)

        # Atualização dos pesos
        W1 += eta * np.outer(delta1, Xb)
        W2 += eta * np.outer(delta2, o1b)
        E[j] = (e ** 2) / 2

    # Média dos erros por época
    print(f"Erro médio na época {i + 1}: {E.mean()}")

# Salvar os pesos após o treinamento
np.save('pesos_W1.npy', W1)
np.save('pesos_W2.npy', W2)

# Teste da rede neural com um exemplo
X_novo = np.array([27.1, 1.0, -1.0, -1.0, 0.0, 0.0, 1.0, 10.0, 1.0, 1.0, 9.0, -1.0, -1.0, -1.0])
Y_pred = predict(X_novo, W1, W2)
print("Probabilidade de doença cardíaca:", Y_pred)
print("Predição:", "Doença cardíaca" if Y_pred >= 0.5 else "Sem doença cardíaca")
