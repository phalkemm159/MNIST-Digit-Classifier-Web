from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, ImageOps
import io
import base64
import os
from scipy import ndimage

app = Flask(__name__)

# Load the trained model
model = None

def load_mnist_model():
    global model
    try:
        model = load_model("mnist_model.h5")
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return False
    return True

# Load model at the global level so gunicorn can access it
load_mnist_model()

def preprocess_image(image_data):
    try:
        # Decode base64 string to bytes
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image = Image.open(io.BytesIO(image_bytes)).convert('L')  # Convert to grayscale

        # Resize to 28x28
        image = ImageOps.invert(image)
        image = image.resize((28, 28), Image.LANCZOS)

        # Convert to numpy array and normalize
        image_array = np.array(image).astype('float32') / 255.0

        # Invert colors
        image_array = 1.0 - image_array

        # Reshape for model input
        image_array = image_array.reshape(1, 28, 28, 1)

        return image_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500

        data = request.json
        image_data = data['image']

        processed_image = preprocess_image(image_data)
        if processed_image is None:
            return jsonify({'error': 'Failed to process image'}), 400

        # Optional: Save processed image for debugging
        debug_img = (processed_image[0, :, :, 0] * 255).astype(np.uint8)
        Image.fromarray(debug_img).save('debug_processed.png')
        print("Saved debug image as 'debug_processed.png'")

        # Predict
        prediction = model.predict(processed_image)
        predicted_digit = int(np.argmax(prediction))
        confidence = float(np.max(prediction))
        probabilities = {str(i): float(prediction[0][i]) for i in range(10)}

        print(f"Prediction: {predicted_digit}, Confidence: {confidence:.3f}")
        print(f"All probabilities: {probabilities}")

        return jsonify({
            'prediction': predicted_digit,
            'confidence': confidence,
            'probabilities': probabilities
        })

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    # This is only used for local development
    if model is not None:
        print("Starting Flask app...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load model. Please ensure 'mnist_model.h5' exists in the current directory.")
