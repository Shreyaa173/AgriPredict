from flask import Flask, request, render_template
import numpy as np
import pickle

# Ensure that the paths to your model and scaler files are correct
model = pickle.load(open('crop_recommendation.pkl', 'rb'))
sc = pickle.load(open('standardscaler.pkl', 'rb'))
mx = pickle.load(open('minmaxscaler.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/home')
def home():
    return render_template("home.html")

@app.route("/predict", methods=['POST'])
def predict():
    try:
        N = float(request.form['Nitrogen'])
        P = float(request.form['Phosporus'])
        K = float(request.form['Potassium'])
        temp = float(request.form['Temperature'])
        humidity = float(request.form['Humidity'])
        ph = float(request.form['pH'])
        rainfall = float(request.form['Rainfall'])

        feature_list = [N, P, K, temp, humidity, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1, -1)

        # Transform the features using MinMax and Standard Scalers
        mx_features = mx.transform(single_pred)
        sc_mx_features = sc.transform(mx_features)

        # Make a prediction
        prediction = model.predict(sc_mx_features)

        # Crop dictionary mapping
        crop_dict = {
            1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
            8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
            14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
            19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
        }

        # Get the crop name from the prediction
        crop = crop_dict.get(prediction[0], "Unknown crop")
        result = f"{crop} is the best crop to be cultivated right there" if crop != "Unknown crop" else "Could not determine the best crop."

        return render_template('home.html', result=result)
    except Exception as e:
        return render_template('home.html', result=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
