import numpy as np
import pandas as pd
from random import randint
import os

def load_weights():
    W1 = np.load('w1.npy')
    W2 = np.load('w2.npy')
    return W1, W2

def sigmoid(x):
    return 1 / (1 + np.exp(-x))  

def predict(X_new, W1, W2, bias=1):
    Xb = np.append(X_new, bias)  
    
    o1 = sigmoid(W1.dot(Xb))  

    o1b = np.append(o1, 1.0)  

    Y_pred = sigmoid(W2.dot(o1b))  

    return Y_pred

df = pd.read_csv('heart_2020_cleaned_copy.csv')
df = df.drop(['Race','DiffWalking', 'GenHealth'], axis=1)

dfNumpy = df.astype(float).to_numpy()

W1, W2 = load_weights()

# i = randint(250000, 300000)

X_novo = dfNumpy[319680]
X_novo = X_novo[1:]

Y_pred = predict(X_novo, W1, W2)

os.system('cls')
# print(i)
print("Probabilidade de doença cardíaca:", Y_pred)
print("Predição:", "Doença cardíaca" if Y_pred >= 0.5 else "Sem doença cardíaca")
