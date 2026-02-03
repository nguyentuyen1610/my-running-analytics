# PLA to classify good and bad running sessions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data2.csv", sep=";")

x1 = (df["km1"] + df["km2"]).values
x2 = (df["km4"] + df["km5"]).values
y = df["label"].values

X = np.vstack((np.ones(len(x1)), x1, x2))
y = y.reshape(1, -1)

def h(w, x):
    return np.sign(w.T @ x)

def perceptron(X, y, w_init):
    w = w_init
    N = X.shape[1]

    while True:
        error = 0
        for i in range(N):
            xi = X[:, i].reshape(-1, 1)
            yi = y[0, i]
            if h(w, xi)[0] != yi:
                w = w + yi * xi
                error += 1
        if error == 0:
            break
    return w

np.random.seed(0)
w_init = np.random.randn(3, 1)
w = perceptron(X, y, w_init)

print("Learned weight vector:", w.ravel())
x_line = np.linspace(min(x1), max(x1), 100)
y_line = -(w[0] + w[1] * x_line) / w[2]

plt.scatter(x1[y[0] == 1], x2[y[0] == 1],
            marker="o", label="good run")
plt.scatter(x1[y[0] == -1], x2[y[0] == -1],
            marker="s", label="bad run")
plt.plot(x_line, y_line, "r", label="classify line")
plt.xlabel("First 2km")
plt.ylabel("Last 2km")
plt.legend()
plt.grid(True)
plt.show()
