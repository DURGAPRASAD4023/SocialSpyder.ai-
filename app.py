from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import tensorflow as tf
import gender_guesser.detector as gender
from flask_cors import CORS

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)  # Enable Cross-Origin Requests

# Load the trained model and scaler
try:
    model = tf.keras.models.load_model("saved_model.h5")
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    model, scaler = None, None

# Function to predict gender from name
def predict_sex(name):
    d = gender.Detector()
    first_name = name.split(' ')[0] if name else ''
    sex = d.get_gender(first_name)
    sex_dict = {'female': -2, 'mostly_female': -1, 'unknown': 0, 'mostly_male': 1, 'male': 2}
    return sex_dict.get(sex, 0)

# Function to extract features from input data
def extract_features(data):
    try:
        lang_dict = {'en': 0, 'es': 1, 'fr': 2, 'de': 3, 'it': 4}  
        lang_code = lang_dict.get(data.get('lang', 'unknown'), -1)
        sex_code = predict_sex(data.get('name', ''))

        features = ['statuses_count', 'followers_count', 'friends_count', 'favourites_count', 'listed_count']
        feature_values = [data.get(feat, 0) for feat in features]

        return np.array([*feature_values, sex_code, lang_code]).reshape(1, -1)
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect')
def detect():
    return render_template('detection.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/subscribe')
def subscribe():
    return render_template('subscribe.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not scaler:
        return jsonify({'error': 'Model or Scaler not loaded'}), 500
    
    try:
        data = request.json
        features = extract_features(data)
        if features is None:
            return jsonify({'error': 'Invalid input data'}), 400

        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0][0]

        return jsonify({
            'prediction': 'Genuine' if prediction > 0.5 else 'Fake',
            'probability': round(float(prediction), 4)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
