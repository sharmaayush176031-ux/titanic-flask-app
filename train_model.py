import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import pickle

# ---------------- LOAD DATASET ----------------
df = pd.read_csv("train.csv")  # Kaggle Titanic dataset

# ---------------- SELECT FEATURES ----------------
df = df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']]

# ---------------- HANDLE MISSING VALUES ----------------
df['Age'].fillna(df['Age'].mean(), inplace=True)

# ---------------- ENCODE SEX COLUMN ----------------
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])  # male=1, female=0

# ---------------- SPLIT FEATURES ----------------
X = df[['Pclass', 'Sex', 'Age', 'Fare']]
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- MODEL 1: RANDOM FOREST ----------------
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

rf_acc = rf_model.score(X_test, y_test)
print("Random Forest Accuracy:", rf_acc)

# ---------------- MODEL 2: NAIVE BAYES ----------------
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

nb_acc = nb_model.score(X_test, y_test)
print("Naive Bayes Accuracy:", nb_acc)

# ---------------- SAVE MODELS ----------------
pickle.dump(rf_model, open("model_rf.pkl", "wb"))
pickle.dump(nb_model, open("model_nb.pkl", "wb"))

print("Models saved successfully ✔")
