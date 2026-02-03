# use Logistic Regression to predict the probability of a good running session based on last 2km
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data2.csv", sep=";")

X = (df["km4"] + df["km5"]).values.reshape(-1, 1)
y = (df["label"] == 1).astype(int)

model = LogisticRegression()
model.fit(X, y)

x_line = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
y_prob = model.predict_proba(x_line)[:, 1]

plt.figure(figsize=(8, 5))
plt.scatter(X[y == 1], y[y == 1],
            label="Good run", alpha=0.7)
plt.scatter(X[y == 0], y[y == 0],
            label="Bad run", alpha=0.7)
plt.plot(x_line, y_prob, color="red", label="probability line")
plt.xlabel("Last 2km (s)")
plt.ylabel("Probability of good run")
plt.legend()
plt.grid(True)
plt.show()
