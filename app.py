from __future__ import division, print_function

# coding=utf-8
import os
import shutil
import numpy as np

# Tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow import expand_dims
from tensorflow.nn import softmax

# Flask utils
from flask import Flask, redirect, url_for, request, render_template, session
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from flask_wtf import FlaskForm
from wtforms import RadioField, SubmitField

# Refreshing the uploads directory 
if os.path.exists("uploads"):
    shutil.rmtree('uploads')
os.makedirs('uploads')

# Define a flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecretkey'

# Flask Form
class InfoForm(FlaskForm):
    crop = RadioField('Please choose the plant:', choices=[('Corn','Corn'), ('Apple','Apple'), ('Tomato','Tomato'), ('Potato','Potato')])
    submit = SubmitField('Next')

# Load models
tomato_model = load_model('models/tomato.h5')
potato_model = load_model('models/potato.h5')
corn_model = load_model('models/CornModel.h5')
apple_model = load_model('models/AppleModel.h5')

# Class names
class_names_tomato = ['Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']
class_names_potato = ['Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight']
class_names_corn = ['Corn___Cercospora_leaf_spot Gray_leaf_spot', 'Corn___Common_rust', 'Corn___healthy', 'Corn___Northern_Leaf_Blight']
class_names_apple = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy']

# Make predictions over new image
def model_predict(img_path, model, class_names, img_dims):

    img = load_img(
        img_path, target_size=(img_dims, img_dims)
    )

    img_array = img_to_array(img)
    img_array = expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = softmax(predictions[0])

    return "This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score))


@app.route('/')
def home():
    #home page
    return render_template('home.html')


@app.route('/options', methods=['GET', 'POST'])
def options():
    form = InfoForm()
    if form.validate_on_submit():
        session['crop'] = form.crop.data
        return redirect(url_for("index"))
    #Options page
    return render_template('options.html', form=form)
    

@app.route('/demo', methods=['GET'])
def index():
    # Demo page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        if session['crop'] == 'Tomato':
            preds = model_predict(file_path, tomato_model, class_names_tomato, img_dims=180)
        elif session['crop'] == 'Potato':
            preds = model_predict(file_path, potato_model, class_names_potato, img_dims=200)
        elif session['crop'] == 'Corn':
            preds = model_predict(file_path, corn_model, class_names_corn, img_dims=200)
        elif session['crop'] == 'Apple':
            preds = model_predict(file_path, apple_model, class_names_apple, img_dims=200)

        return preds
    return None


@app.route('/team')
def team():
    return render_template('team.html')


if __name__ == '__main__':
    app.run()