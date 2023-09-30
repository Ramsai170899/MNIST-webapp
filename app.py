import base64
import numpy as np
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import io
from PIL import Image

app = Flask(__name__)

# Load your trained MNIST model
model = tf.keras.models.load_model('mnist_model.h5')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Receive the image data from the client
        image_data_url = request.json['image_data']

        image_data = image_data_url.split(',')[1]
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))

        # Resize and normalize the image
        image = image.resize((28, 28))
        image = np.array(image)
        image = image[:, :, 3]  # Keep only one channel (assuming RGBA image)
        image = image.reshape((1, 28, 28, 1))
        image = image / 255.0

        # Make a prediction using the loaded model
        prediction = model.predict(image)

        # Get the predicted digit (the index with the highest probability)
        predicted_digit = np.argmax(prediction)
        print("\n predicted_digit : ", predicted_digit)

        # Convert predicted_digit to a string before sending as JSON
        return jsonify({'predicted_digit': str(predicted_digit)})

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
