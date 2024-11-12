import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))  # Função sigmoide

def predict(X_new, W1, W2, bias=1):
    # Adiciona o viés ao vetor de entrada X_new
    Xb = np.append(X_new, bias)  # Agora Xb tem 15 elementos (14 variáveis + 1 viés)
    
    # Saída da Camada Escondida usando sigmoide
    o1 = sigmoid(W1.dot(Xb))  # Produto escalar entre W1 e Xb

    # Inclui o viés na saída da camada escondida
    o1b = np.append(o1, 1.0)  # Agora o1b tem 31 elementos (30 variáveis + 1 viés)

    # Saída final da rede neural usando sigmoide
    Y_pred = sigmoid(W2.dot(o1b))  # Produto escalar entre W2 e o1b

    # Interpreta o resultado como uma probabilidade
    return Y_pred
