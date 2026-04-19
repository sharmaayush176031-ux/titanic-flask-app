import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np
import pickle

# ---------------- LOAD DATA ----------------
df = pd.read_csv("train.csv")

# ---------------- KEEP ONLY REQUIRED COLUMNS ----------------
df = df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']]

# ---------------- FORCE CLEANING (MOST IMPORTANT FIX) ----------------

# Convert to numeric (kills hidden bad values)
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df['Fare'] = pd.to_numeric(df['Fare'], errors='coerce')

# Fill missing values
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Fare'] = df['Fare'].fillna(df['Fare'].median())

# Remove ANY remaining bad rows
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

# ---------------- ENCODE SEX ----------------
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])

# ---------------- FEATURES ----------------
X = df[['Pclass', 'Sex', 'Age', 'Fare']]
y = df['Survived']

# FINAL SAFETY CHECK (VERY IMPORTANT)
X = X.fillna(0)
y = y.fillna(0)

# ---------------- SPLIT DATA ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- MODEL 1 ----------------
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

print("Random Forest Accuracy:", rf.score(X_test, y_test))

# ---------------- MODEL 2 ----------------
nb = GaussianNB()
nb.fit(X_train, y_train)

print("Naive Bayes Accuracy:", nb.score(X_test, y_test))

# ---------------- SAVE MODELS ----------------
pickle.dump(rf, open("model_rf.pkl", "wb"))
pickle.dump(nb, open("model_nb.pkl", "wb"))

print("✔ Models saved successfully")
