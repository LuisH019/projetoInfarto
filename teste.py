import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Carregar o dataset fornecido
df = pd.read_csv('heart_2020_cleaned copy.csv')

# Converter o DataFrame para um array NumPy de floats
dfNumpy = df.astype(float).to_numpy()[:200000].T

# Parâmetros iniciais
rows, cols = dfNumpy.shape

numEpocas = 100  # Aumentado para observar a convergência ao longo de mais iterações
eta = 0.05       # Diminuído para um ajuste mais controlado dos pesos
m = rows - 1
N = 30           # Aumentado o número de neurônios na camada escondida para captar mais características
L = 1

# Vetor de classificação desejada
d = dfNumpy[0]

# Inicia aleatoriamente as matrizes de pesos
W1 = np.random.random((N, m + 1))
W2 = np.random.random((L, N + 1))

# Array para armazenar os erros
E = np.zeros(cols)
Etm = np.zeros(numEpocas)

# Bias
bias = 1

# Entrada do Perceptron
X = dfNumpy[1:]

# Função de ativação sigmoide
sigmoid = lambda x: 1 / (1 + np.exp(-x))

# Treinamento
for i in range(numEpocas):
    for j in range(cols):
        # Adicionar o bias ao vetor de entrada
        Xb = np.hstack((bias, X[:, j]))

        # Saída da Camada Escondida usando sigmoide
        o1 = sigmoid(W1.dot(Xb))

        # Incluindo o bias
        o1b = np.insert(o1, 0, bias)

        # Saída da rede neural usando sigmoide
        Y = sigmoid(W2.dot(o1b))

        # Erro
        e = d[j] - Y

        # Erro Total
        E[j] = (e.transpose().dot(e)) / 2

        # Retropropagação do erro
        delta2 = np.diag(e).dot(Y * (1 - Y))
        vdelta2 = W2.T.dot(delta2)
        delta1 = np.diag(o1b * (1 - o1b)).dot(vdelta2)

        # Atualização dos pesos
        W1 += eta * np.outer(delta1[1:], Xb)
        W2 += eta * np.outer(delta2, o1b)

    # Calculo da média dos erros
    Etm[i] = E.mean()

# Plotar o Erro Médio por época
plt.xlabel("Épocas")
plt.ylabel("Erro Médio")
plt.plot(Etm, color='c')
plt.plot(Etm)
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
print(np.round(Error_Test) - d) # Exibe erros e a precisão final em relação ao vetor d
