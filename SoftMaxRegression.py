# Softmax Regression model to classify running sessions from total, last 2km, split diff and day diff
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df = pd.read_csv("data3.csv", sep=";")
df["last2km"] = df["km4"] + df["km5"]
df["splitdiff"] = df["km5"] - df["km1"]

X = df[
    [
        "total",
        "last2km",
        "splitdiff",
        "daydiff"
    ]
]
y = df["label"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(
    solver="lbfgs",
    max_iter=1000
)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["bad", "medium", "good"]
    )
)

result = X_test.copy()
result["true"] = y_test.values
result["pred"] = y_pred
result["P_bad"] = y_proba[:, 0]
result["P_medium"] = y_proba[:, 1]
result["P_good"] = y_proba[:, 2]

print("\nPrediction samples:")
print(result.head())
