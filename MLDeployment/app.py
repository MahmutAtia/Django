from flask import Flask, request, render_template
import  numpy as np
import pickle


model = pickle.load(open("model.pkl","rb"))


app = Flask(__name__)
@app.route("/")
def home():
  return render_template("index.html")


@app.route("/predict", methods=["GET","POST"])
def predict():

  features = [float(x) for x in request.form.values()]
  f_array = np.array([features])
  prediction = model.predict(f_array)



  return render_template("index.html", pre = f"the prediction is {prediction}")

if __name__ == "__main__":
  app.run(debug=True)
