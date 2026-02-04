# apply Gaussian Naive Bayes to classify running sessions from total, last 2km, day diff
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

df = pd.read_csv("data2.csv", sep = ";")
df["last2km"] = df["km4"] + df["km5"]

X = df[["last2km", "total", "daydiff"]]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y 
)

gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Class prior probabilities:")
print("P(y=0) =", gnb.class_prior_[0])
print("P(y=1) =", gnb.class_prior_[1])
