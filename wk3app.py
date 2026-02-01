from flask import Flask, request, jsonify, render_template
import numpy as np
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Train a simple model at startup
def train_model():
    rng = np.random.default_rng(42)
    X, y = [], []

    for _ in range(500):
        revenue = rng.uniform(50000, 2000000)
        cost = rng.uniform(30000, 1800000)
        marketing = rng.uniform(0, 300000)
        employees = rng.integers(1, 300)

        profit = revenue - cost
        margin = profit / max(revenue, 1)

        if margin > 0.25 and profit > 200000:
            label = 0
        elif margin > 0.10:
            label = 1
        else:
            label = 2

        X.append([revenue, cost, marketing, employees])
        y.append(label)

    X = np.array(X)
    X[:, 0] /= 2000000
    X[:, 1] /= 1800000
    X[:, 2] /= 300000
    X[:, 3] /= 300

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model

MODEL = train_model()
LABELS = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    x = np.array([[ 
        float(data["revenue"]) / 2000000,
        float(data["cost"]) / 1800000,
        float(data["marketing"]) / 300000,
        float(data["employees"]) / 300
    ]])
    pred = int(MODEL.predict(x)[0])
    return jsonify({"prediction": LABELS[pred]})

if __name__ == "__main__":
    app.run()
