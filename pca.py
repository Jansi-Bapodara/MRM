import numpy as np
import pandas as pd

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        cov = np.cov(X_centered, rowvar=False)

        eigenvalues, eigenvectors = np.linalg.eig(cov)

        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]

        self.components = eigenvectors[:, :self.n_components]

    def transform(self, X):
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


df = pd.read_csv("Titanic.csv")

df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

features = ["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex"]
df_numeric = df[features].dropna()

X = df_numeric.values

X = (X - X.mean(axis=0)) / X.std(axis=0)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print("Principal Components:")
print(pca.components)