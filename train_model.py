import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import pickle

# ---------------- LOAD DATA ----------------
df = pd.read_csv("train.csv")

# ---------------- KEEP FEATURES ----------------
df = df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']]

# ---------------- ENCODE SEX ----------------
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])

# ---------------- SPLIT ----------------
X = df[['Pclass', 'Sex', 'Age', 'Fare']]
y = df['Survived']

# ---------------- TRAIN TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================================================
# 🔥 PIPELINE 1: RANDOM FOREST
# =========================================================

rf_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", RandomForestClassifier(n_estimators=100, random_state=42))
])

rf_model.fit(X_train, y_train)

print("Random Forest Accuracy:", rf_model.score(X_test, y_test))

# =========================================================
# 🔥 PIPELINE 2: NAIVE BAYES
# =========================================================

nb_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", GaussianNB())
])

nb_model.fit(X_train, y_train)

print("Naive Bayes Accuracy:", nb_model.score(X_test, y_test))

# ---------------- SAVE MODELS ----------------
pickle.dump(rf_model, open("model_rf.pkl", "wb"))
pickle.dump(nb_model, open("model_nb.pkl", "wb"))

print("✔ Models saved successfully")
