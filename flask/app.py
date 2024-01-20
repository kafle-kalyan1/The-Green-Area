import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from flask import Flask, request, jsonify
from flask_cors import CORS 
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)
CORS(app)  

model = tf.keras.models.load_model('PlantDNet.h5', compile=False)
print('Model loaded. Check http://127.0.0.1:5000/')

def model_predict(img_path, model):
    img = image.load_img(img_path, grayscale=False, target_size=(64, 64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = np.array(x, 'float32') / 255.0
    preds = model.predict(x)
    return preds

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.json:
        return jsonify({'error': 'No file part'})
    
    data_url = request.json['file']
    # Split the base64 string in data URL
    _, base64_data = data_url.split(',')
    byte_data = base64.b64decode(base64_data)

    # Create a PIL Image object
    image_data = BytesIO(byte_data)
    img = Image.open(image_data)
    img.save('temp_image.jpg', 'JPEG')

    # Make prediction
    preds = model_predict('temp_image.jpg', model)
    disease_class = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight',
                     'Potato___Late_blight', 'Potato___healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight',
                     'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
                     'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot',
                     'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy']
    
    # Get the predicted class
    predicted_class = disease_class[np.argmax(preds[0])]

    # Prepare the response
    response = {
        'prediction': predicted_class,
        'confidence': float(preds[0][np.argmax(preds[0])])
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
