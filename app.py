from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import cv2

app = Flask(__name__)

model = load_model('mask.h5')

@app.route('/predict', methods=['POST'])
def predict():
    img = request.files['image'].read()
    npimg = np.fromstring(img, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    preds = model.predict(img)
    label = "Mask" if preds[0][0] > preds[0][1] else "No



if __name__ == "__main__":
    app.run(debug = True)