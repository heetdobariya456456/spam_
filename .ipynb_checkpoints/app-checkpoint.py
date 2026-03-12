from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    probability = None

    if request.method == "POST":
        text = request.form["message"]
        vector = vectorizer.transform([text])
        pred = model.predict(vector)[0]
        prob = model.predict_proba(vector)[0][pred]

        prediction = "Spam 🚫" if pred == 1 else "Not Spam ✅"
        probability = round(prob * 100, 2)

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability
    )

if __name__ == "__main__":
    app.run(debug=True)
