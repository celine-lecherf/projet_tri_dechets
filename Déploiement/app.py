from flask import Flask, render_template, request, redirect, url_for, send_from_directory, abort

from flask_wtf import FlaskForm
from flask_wtf.file import FileField
from wtforms import SubmitField
from werkzeug.utils import secure_filename
import os
import keras.models
import re
import sys 
import base64
sys.path.append(os.path.abspath("./model"))
from load import *
import tensorflow as tf
from keras.preprocessing.image import load_img
from tensorflow.keras.applications import imagenet_utils
from keras.preprocessing import image
#from PIL import Image
import io
import numpy as np


app = Flask(__name__)

app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.jpeg']
app.config['UPLOAD_PATH'] = 'uploads'

InputShape=(224,224)
categories = ['battery', 'biological', 'brown-glass', 'cardboard', 'clothes', 'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass']

model = tf.keras.models.load_model("model\model_projet")


def prepare_image(image):
    image = tf.keras.preprocessing.image.img_to_array(image)/255
    image = np.expand_dims(image, axis=0)
    # return the processed image
    return image


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_files():
    uploaded_file = request.files['file']
    filename = secure_filename(uploaded_file.filename)
    if filename != '':
        file_ext = os.path.splitext(filename)[1]
        if file_ext not in app.config['UPLOAD_EXTENSIONS']:
            abort(400)
        file = uploaded_file.save(os.path.join(app.config['UPLOAD_PATH'], filename))
        file_path = os.path.join(app.config['UPLOAD_PATH'], filename)
        image= load_img(file_path, target_size=(224,224))
        image= prepare_image(image)
        p = model.predict(image)
        prediction= np.argmax(p,axis=1)
        if prediction == 0:
            garbage = "battery"
        elif prediction == 1:
            garbage = "biological"
        elif prediction == 2:
            garbage = "brown-glass"
        elif prediction == 3:
            garbage = "cardboard"
        elif prediction == 4:
            garbage = "clothes"
        elif prediction == 5:
            garbage = "green-glass"
        elif prediction == 6:
            garbage = "metal"
        elif prediction == 7:
            garbage = "paper"
        elif prediction == 8:
            garbage = "plastic"
        elif prediction == 9:
            garbage = "shoes"
        elif prediction == 10:
            garbage = "trash"
        elif prediction == 11:
            garbage = "white-glass"
        else:
            garbage = "Other"

    return render_template("index.html", prediction=garbage,image=file_path)


@app.route('/uploads/<filename>')
def upload(filename):
    return send_from_directory(app.config['UPLOAD_PATH'], filename)


if __name__ == "__main__":
    app.run(debug=True, port=5002)

