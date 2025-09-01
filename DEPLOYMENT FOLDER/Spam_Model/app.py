import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle
    
# load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# create flask app
app = Flask(__name__)

# # load thr pickle model
# model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def Home():
    return render_template("index.html")

# 2222222222
@app.route("/predict", methods = ["POST"])
def predict():
    try:
    
        input_data = request.form["Text"]
        if not input_data.strip():
            return render_template("index.html", prediction="Please enter a message.")
        
        # Convert to DataFrame
        input_df = pd.DataFrame([{'Text':input_data}])
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        label = "Spam" if prediction == 1 else "Ham"
        return render_template("index.html", prediction=f" The message is classified as: {label}")
        
    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")
    
if __name__ == "__main__":
    app.run(debug=True)
    
    
# 11111111111111111111111
# @app.route("/predict", methods = ["POST"])
# def predict():
#     message = request.form['message']
#     if not message.strip():
#         return render_template("index.html", prediction="Please enter a message.")
#     input_df = pd.DataFrame({"Text":[message]})
#     prediction = model.predict(input_df)[0]
#     label = "Spam" if prediction == 1 else "Ham"
#     return render_template("index.html", prediction=f" The message is classified as: {label}")
    
        
#     # float_features = [float(x) for x in request.form.values()]
#     # features = [np.array(float_features)]
#     # prediction = model.predict(features)
    
#     # return render_template("index.html", prediction_text = "The flower species is {}".format(prediction))

# if __name__ == "__main__":
#     app.run(debug=True)


