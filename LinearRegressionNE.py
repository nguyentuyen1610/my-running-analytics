# use Normal Equation for Linear Regression to estimate the linear relationship between running time and calories burned
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data.csv", sep=";")
df["kcal"] = df["kcal"].str.replace(",", ".").astype(float)

X = df["total"].values.reshape(-1, 1)
y = df["kcal"].values.reshape(-1, 1)

one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis=1)

A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)
w = np.dot(np.linalg.pinv(A), b)

print("w =", w)
w_0 = w[0, 0]
w_1 = w[1, 0]

x0 = np.linspace(1600, 2600, 100)
y0 = w_0 + w_1 * x0

plt.plot(X, y, 'ro', label="data")
plt.plot(x0, y0, label="fit line")
plt.axis([1500, 2700, 330, 480])
plt.xlabel("Total time for 5km (s)")
plt.ylabel("Kcal")
plt.legend()
plt.show()
