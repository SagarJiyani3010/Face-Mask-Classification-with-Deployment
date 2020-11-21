import numpy as np
import tensorflow as tf
import cv2
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
model = tf.keras.models.load_model('models/model.h5')

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(file.filename))
        file.save(file_path)
    
    path = os.path.join(file_path)
    print(path)

    img = cv2.imread(path)
    img = cv2.resize(img,(100,100))
    img = np.reshape(img,[1,100,100,3])

    prediction = model.predict(img)
    prediction = 'Without Mask' if prediction.item(0)==0 else 'With Mask'
    color = 'danger' if prediction=='Without Mask' else 'success'
    return render_template('index.html', prediction_text='{}' .format(prediction), color = '{}' .format(color))

if __name__ == '__main__':
    app.run(debug=True)