import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from predict import predict


df = pd.read_csv('heart_2020_cleaned_copy.csv')
df = df.drop(['Race','DiffWalking', 'GenHealth'], axis=1)

dfNumpy = df.astype(float).to_numpy()[:200000].T

rows, cols = dfNumpy.shape

numEpocas = 100  
eta = 0.05       
m = rows - 1
N = 30           
L = 1

d = dfNumpy[0]  

W1 = np.random.random((N, m + 1))  
W2 = np.random.random((L, N + 1))  

E = np.zeros(cols)
Etm = np.zeros(numEpocas)

bias = 1

X = dfNumpy[1:]  

sigmoid = lambda x: 1 / (1 + np.exp(-x))

for i in range(numEpocas):
    for j in range(cols):
        Xb = np.hstack((bias, X[:, j]))

        o1 = sigmoid(W1.dot(Xb))

        o1b = np.insert(o1, 0, bias)

        Y = sigmoid(W2.dot(o1b))

        e = d[j] - Y

        E[j] = (e ** 2) / 2

        delta2 = e * Y * (1 - Y) 
        delta1 = delta2 * W2[:, 1:] * o1 * (1 - o1) 

        W1 += eta * np.outer(delta1, Xb)
        W2 += eta * np.outer(delta2, o1b)

    Etm[i] = E.mean()

plt.xlabel("Épocas")
plt.ylabel("Erro Médio")
plt.plot(Etm, color='c')
plt.show()

Error_Test = np.zeros(cols)

for i in range(cols):
    Xb = np.hstack((bias, X[:, i]))
    o1 = sigmoid(W1.dot(Xb))
    o1b = np.insert(o1, 0, bias)
    Y = sigmoid(W2.dot(o1b))
    print(Y)
    Error_Test[i] = d[i] - Y

print(Error_Test)
print(np.round(Error_Test) - d)

np.save('w1.npy', W1)
np.save('w2.npy', W2)
# # Teste com novo dado
# X_novo = np.array([27.1, 1.0, -1.0, -1.0, 0.0, 0.0, 1.0, 10.0, 1.0, 1.0, 9.0, -1.0, -1.0, -1.0])  # Exemplo de entrada com 14 variáveis

# # Adiciona o viés a X_novo
# X_novo = np.append(X_novo, 1.0)  # Agora X_novo tem 15 elementos (14 variáveis + 1 viés)

# # Realiza a predição com o modelo treinado
# Y_pred = predict(X_novo, W1, W2)

# print("Probabilidade de doença cardíaca:", Y_pred)
# print("Predição:", "Doença cardíaca" if Y_pred >= 0.5 else "Sem doença cardíaca")


