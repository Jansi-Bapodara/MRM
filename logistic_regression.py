import numpy as np
import pandas as pd

df = pd.read_csv("titanic.csv")

target = 'Survived'

df = df.select_dtypes(include=[np.number])

df = df.dropna()

X = df.drop(target, axis=1).values
y = df[target].values.reshape(-1, 1)

X = (X - X.mean(axis=0)) / X.std(axis=0)

m, n = X.shape
w = np.zeros((n, 1))
b = 0
lr = 0.1
n = 1000

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

for i in range(n):
    z = np.dot(X, w) + b
    y_hat = sigmoid(z)
    
    dw = (1/m) * np.dot(X.T, (y_hat - y))
    db = (1/m) * np.sum(y_hat - y)
    w -= lr * dw
    b -= lr * db

y_pred = sigmoid(np.dot(X, w) + b)
y_pred_cls = (y_pred > 0.5).astype(int)
acc = (y_pred_cls == y).mean() * 100
print(f"\nAccuracy: {acc:.2f}%")