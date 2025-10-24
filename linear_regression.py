import numpy as np
import pandas as pd

df = pd.read_csv("CarPrice_Assignment.csv")
df = df[['horsepower', 'price']]
df.dropna(inplace=True)

X = df['horsepower'].values
y = df['price'].values
n = len(X)

w = 0 
b = 0  
lr = 0.0001  
epochs = 205  

for i in range(epochs):
    y_pred = w * X + b
    
    dw = (-2/n) * np.sum(X * (y - y_pred))
    db = (-2/n) * np.sum(y - y_pred)
    
    w = w - lr * dw
    b = b - lr * db


print("\nTraining completed.")
print(f"Final weight (w): {w}")
print(f"Final bias (b): {b}")
