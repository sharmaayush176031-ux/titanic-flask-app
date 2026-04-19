import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import pickle

# ---------------- LOAD DATA ----------------
df = pd.read_csv("train.csv")

# ---------------- SELECT FEATURES ----------------
df = df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']]

# ---------------- STRONG CLEANING (IMPORTANT FIX) ----------------
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Fare'] = df['Fare'].fillna(df['Fare'].median())

# EXTRA SAFETY: remove any remaining NaN rows
df = df.dropna()

# ---------------- ENCODE SEX ----------------
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])

# ---------------- SPLIT DATA ----------------
X = df[['Pclass', 'Sex', 'Age', 'Fare']]
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- RANDOM FOREST ----------------
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

print("Random Forest Accuracy:", rf_model.score(X_test, y_test))

# ---------------- NAIVE BAYES ----------------
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

print("Naive Bayes Accuracy:", nb_model.score(X_test, y_test))

# ---------------- SAVE MODELS ----------------
pickle.dump(rf_model, open("model_rf.pkl", "wb"))
pickle.dump(nb_model, open("model_nb.pkl", "wb"))

print("Models saved successfully ✔")
