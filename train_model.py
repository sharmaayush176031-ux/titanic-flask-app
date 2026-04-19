import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import pickle

# ---------------- LOAD DATA ----------------
df = pd.read_csv("train.csv")

# ---------------- SELECT FEATURES ----------------
df = df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']]

# ---------------- ENCODE SEX ----------------
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])

# ---------------- SPLIT X AND Y ----------------
X = df[['Pclass', 'Sex', 'Age', 'Fare']]
y = df['Survived']

# ---------------- IMPUTER (IMPORTANT FIX) ----------------
imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(X)

# ---------------- FINAL SAFETY CHECK ----------------
X = np.nan_to_num(X)

# ---------------- TRAIN TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- RANDOM FOREST ----------------
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

print("Random Forest Accuracy:", rf.score(X_test, y_test))

# ---------------- NAIVE BAYES ----------------
nb = GaussianNB()
nb.fit(X_train, y_train)

print("Naive Bayes Accuracy:", nb.score(X_test, y_test))

# ---------------- SAVE MODELS ----------------
pickle.dump(rf, open("model_rf.pkl", "wb"))
pickle.dump(nb, open("model_nb.pkl", "wb"))

print("✔ Models saved successfully")
