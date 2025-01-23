# app.py
from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained LSTM model and preprocessor
model = load_model('model/lstm_model.keras')
preprocessor = joblib.load('data/preprocessor.pkl')
scaler_y = joblib.load('data/scaler_y.pkl')

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for predicting laptop price
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        name = request.form['name']
        processor = request.form['processor']
        screen_size = float(request.form['screen_size'])
        color = request.form['color']
        ram = int(request.form['ram'])
        os = request.form['os']
        brand = request.form['brand']
        touchscreen = int(request.form['touchscreen'])
        ips = int(request.form['ips'])
        vga_brand = request.form['vga_brand']
        ssd = int(request.form['ssd'])
        hdd = int(request.form['hdd'])
        actual_price = float(request.form['actual_price'])

        # Combine features into DataFrame (Ensure column names match preprocessor)
        features = pd.DataFrame([{
            'Nama': name,
            'Processor': processor,
            'Ukuran Layar': screen_size,
            'Warna': color,
            'RAM': ram,
            'Sistem Operasi': os,
            'Brand': brand,
            'Touchscreen': touchscreen,
            'IPS': ips,
            'VGA Brand': vga_brand,
            'SSD': ssd,
            'HDD': hdd
        }])

        # Validate column names
        expected_columns = set(preprocessor.feature_names_in_)
        actual_columns = set(features.columns)
        if expected_columns != actual_columns:
            return jsonify({
                'error': f'columns are missing or mismatched: {expected_columns - actual_columns}'
            })

        # Preprocess dan prediksi
        features_transformed = preprocessor.transform(features)
        features_lstm = features_transformed.reshape((features_transformed.shape[0], 1, features_transformed.shape[1]))
        prediction_scaled = model.predict(features_lstm)
        predicted_price = scaler_y.inverse_transform(prediction_scaled)[0][0]

        # Hitung persentase akurasi
        error_rate = abs(predicted_price - actual_price) / actual_price
        accuracy = (1 - error_rate) * 100

        # Kirim respons JSON termasuk harga asli dan akurasi
        return jsonify({
            'predicted_price': float(predicted_price),
            'actual_price': actual_price,
            'accuracy': round(accuracy, 2)  # Akurasi dalam persen dengan 2 desimal
        })

    except Exception as e:
        return jsonify({'error': str(e)})

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)
