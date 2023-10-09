from PIL import Image
import os
from flask import Flask, render_template, request, send_from_directory
import numpy as np
import tensorflow as tf

app = Flask(__name__)

STATIC_FOLDER = 'static'

# Path to the folder where we store the different models
MODEL_FOLDER = STATIC_FOLDER + '/models'
# Path to the folder where we'll store the upload before prediction
UPLOAD_FOLDER = STATIC_FOLDER + '/uploads'



def load__model():
    print('[INFO] : Model Loading ........')
    global model
    model = tf.keras.models.load_model(MODEL_FOLDER + '/cat_dog_classifier.h5')
    print('[INFO] : Model Loaded')

def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((128,128))
    img = np.array(img)
    img = img.astype('float')/255
    img = np.expand_dims(img,axis=0)
    return img

def predict(fullpath):
    # Load and preprocess the image

    data = preprocess_image(fullpath)

    # Make predictions
    result = model.predict(data)

    return result




# Home page
@app.route('/')
def index():
    return render_template('index.html')

# Process file and predict label
@app.route('/upload',methods=['GET','POST'])
def upload_file():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        file = request.files['image']
        fullname = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(fullname)

        result = predict(fullname)

        pred_prob = result.item()

        if pred_prob > .5:
            label = 'Dog'
            accuracy = round(pred_prob * 100, 2)
        else:
            label = 'Cat'
            accuracy = round((1 - pred_prob) * 100, 2)

        return render_template('predict.html', image_file_name=file.filename, label=label, accuracy=accuracy)


@app.route('/upload/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


def create_app():
    load__model()
    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)