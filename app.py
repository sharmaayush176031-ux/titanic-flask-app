from flask import Flask, request
import pickle
import numpy as np

app = Flask(__name__)

# ---------------- LOAD TRAINED MODELS ----------------
rf_model = pickle.load(open("model_rf.pkl", "rb"))
nb_model = pickle.load(open("model_nb.pkl", "rb"))

# ---------------- HOME PAGE ----------------
@app.route('/')
def home():
    return """
    <html>
    <head>
        <title>Titanic Survival Predictor</title>
        <style>
            body {
                margin: 0;
                font-family: Arial;
                text-align: center;
                color: white;
                background: url('https://images.unsplash.com/photo-1500375592092-40eb2168fd21?auto=format&fit=crop&w=1600&q=80');
                background-size: cover;
            }

            .overlay {
                background: rgba(0,0,0,0.7);
                height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                flex-direction: column;
            }

            .title {
                font-size: 45px;
                font-weight: bold;
            }

            .btn {
                margin-top: 20px;
                padding: 15px 30px;
                background: #00c6ff;
                color: white;
                border-radius: 10px;
                text-decoration: none;
                font-size: 18px;
            }

            .btn:hover {
                background: #0072ff;
            }
        </style>
    </head>

    <body>
        <div class="overlay">
            <div class="title">🚢 Titanic Survival Prediction</div>
            <a class="btn" href="/predictor">Enter Simulation</a>
        </div>
    </body>
    </html>
    """

# ---------------- INPUT PAGE ----------------
@app.route('/predictor')
def predictor():
    return """
    <html>
    <head>
        <title>Prediction</title>
    </head>

    <body style="text-align:center;font-family:Arial;background:#1e3c72;color:white;">

        <h1>Enter Passenger Details</h1>

        <form action="/predict" method="post">

            <input name="age" placeholder="Age" required><br><br>

            <input name="fare" placeholder="Fare" required><br><br>

            <select name="pclass">
                <option value="1">1st Class</option>
                <option value="2">2nd Class</option>
                <option value="3">3rd Class</option>
            </select><br><br>

            <select name="sex">
                <option value="1">Male</option>
                <option value="0">Female</option>
            </select><br><br>

            <button type="submit">Predict</button>

        </form>

    </body>
    </html>
    """

# ---------------- PREDICTION ----------------
@app.route('/predict', methods=['POST'])
def predict():
    age = float(request.form['age'])
    fare = float(request.form['fare'])
    pclass = int(request.form['pclass'])
    sex = int(request.form['sex'])

    # Convert input for ML model
    data = np.array([[pclass, sex, age, fare]])

    # Predictions
    rf_pred = rf_model.predict(data)[0]
    nb_pred = nb_model.predict(data)[0]

    result = "✅ Survived" if rf_pred == 1 else "❌ Not Survived"

    return f"""
    <html>
    <body style="text-align:center;font-family:Arial;background:#0f172a;color:white;padding-top:100px;">

        <h1>{result}</h1>

        <h2>Random Forest: {rf_pred}</h2>
        <h2>Naive Bayes: {nb_pred}</h2>

        <br><br>
        <a href="/" style="color:#00c6ff;">Back Home</a>

    </body>
    </html>
    """

# ---------------- RUN APP ----------------
if __name__ == "__main__":
    app.run()
