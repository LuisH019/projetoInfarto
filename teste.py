import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from predict import predict

# Carregar o dataset fornecido
df = pd.read_csv('C:/Users/Windows 10 Pro/Desktop/corazaum/projetoInfarto/heart_2020_cleaned_copy.csv')
df = df.drop(['Race','DiffWalking', 'GenHealth'], axis=1)
# Converter o DataFrame para um array NumPy de floats
dfNumpy = df.astype(float).to_numpy()[:200000].T  # Trazendo os dados de forma transposta

# Parâmetros iniciais
rows, cols = dfNumpy.shape

numEpocas = 100  # Aumentado para observar a convergência ao longo de mais iterações
eta = 0.05       # Diminuído para um ajuste mais controlado dos pesos
m = rows - 1
N = 30           # Aumentado o número de neurônios na camada escondida para captar mais características
L = 1

# Vetor de classificação desejada (target)
d = dfNumpy[0]  # A primeira linha é o alvo (classificação desejada)

# Inicia aleatoriamente as matrizes de pesos
W1 = np.random.random((N, m + 1))  # N neurônios na camada escondida, m características + viés
W2 = np.random.random((L, N + 1))  # L saídas, N neurônios na camada escondida + viés

# Array para armazenar os erros
E = np.zeros(cols)
Etm = np.zeros(numEpocas)

# Bias
bias = 1

# Entrada do Perceptron
X = dfNumpy[1:]  # A partir da segunda linha, temos as características

# Função de ativação sigmoide
sigmoid = lambda x: 1 / (1 + np.exp(-x))

# Treinamento
for i in range(numEpocas):
    for j in range(cols):
        # Adicionar o bias ao vetor de entrada
        Xb = np.hstack((bias, X[:, j]))

        # Saída da Camada Escondida usando sigmoide
        o1 = sigmoid(W1.dot(Xb))

        # Incluindo o bias na saída da camada escondida
        o1b = np.insert(o1, 0, bias)

        # Saída da rede neural usando sigmoide
        Y = sigmoid(W2.dot(o1b))

        # Erro
        e = d[j] - Y

        # Erro Total
        E[j] = (e.item() ** 2) / 2

        # Retropropagação do erro
        delta2 = e * Y * (1 - Y)  # Derivada da sigmoide
        delta1 = delta2 * W2[:, 1:] * o1 * (1 - o1)  # Retropropagação para a camada escondida

        # Atualização dos pesos
        W1 += eta * np.outer(delta1, Xb)
        W2 += eta * np.outer(delta2, o1b)

    # Calculo da média dos erros por época
    Etm[i] = E.mean()

# Plotar o Erro Médio por época
plt.xlabel("Épocas")
plt.ylabel("Erro Médio")
plt.plot(Etm, color='c')
plt.show()

# Teste da rede neural
Error_Test = np.zeros(cols)

for i in range(cols):
    Xb = np.hstack((bias, X[:, i]))
    o1 = sigmoid(W1.dot(Xb))
    o1b = np.insert(o1, 0, bias)
    Y = sigmoid(W2.dot(o1b))
    print(Y)
    Error_Test[i] = d[i] - Y

print(Error_Test)
print(np.round(Error_Test) - d)  # Exibe erros e a precisão final em relação ao vetor d

# Teste com novo dado
X_novo = np.array([27.1, 1.0, -1.0, -1.0, 0.0, 0.0, 1.0, 10.0, 1.0, 1.0, 9.0, -1.0, -1.0, -1.0])  # Exemplo de entrada com 14 variáveis

# Adiciona o viés a X_novo
X_novo = np.append(X_novo, 1.0)  # Agora X_novo tem 15 elementos (14 variáveis + 1 viés)

# Realiza a predição com o modelo treinado
Y_pred = predict(X_novo, W1, W2)

print("Probabilidade de doença cardíaca:", Y_pred)
print("Predição:", "Doença cardíaca" if Y_pred >= 0.5 else "Sem doença cardíaca")
