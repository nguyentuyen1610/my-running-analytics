# use Gradient Descent for Linear Regression to estimate the linear relationship between running time and calories burned
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data.csv", sep=";")
df["kcal"] = df["kcal"].str.replace(",", ".").astype(float)

X = df["total"].values.reshape(-1, 1)
y = df["kcal"].values.reshape(-1, 1)

# normalize X
X_mean = X.mean()
X_std = X.std()
X_norm = (X - X_mean) / X_std

one = np.ones((X_norm.shape[0], 1))
Xbar = np.concatenate((one, X_norm), axis=1)
m = Xbar.shape[0]

w = np.zeros((2, 1))
learning_rate = 0.01
n_iterations = 5000

for _ in range(n_iterations):
    y_pred = Xbar.dot(w)
    gradients = (2 / m) * Xbar.T.dot(y_pred - y)
    w -= learning_rate * gradients

print("w =", w)

x0 = np.linspace(1600, 2600, 100)
x0_norm = (x0 - X_mean) / X_std
y0 = w[0, 0] + w[1, 0] * x0_norm

plt.plot(X, y, 'ro', label="data")
plt.plot(x0, y0, label="fit line (GD)")
plt.axis([1500, 2700, 330, 480])
plt.xlabel("Total time for 5km (s)")
plt.ylabel("Kcal")
plt.legend()
plt.show()
