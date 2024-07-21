from flask import Flask, request, render_template, redirect, flash, session
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask_sqlalchemy import SQLAlchemy
from flask_caching import Cache
import redis
from flask_migrate import Migrate

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'supersecretkey')
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max file size
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///diabetic_retinopathy.db')
app.config['CACHE_TYPE'] = 'RedisCache'
app.config['CACHE_REDIS_URL'] = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')

db = SQLAlchemy(app)
cache = Cache(app)
migrate = Migrate(app, db)

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), nullable=False, unique=True)
    email = db.Column(db.String(150), nullable=False, unique=True)
    password_hash = db.Column(db.String(150), nullable=False)
    images = db.relationship('Image', backref='user', lazy=True)

class Image(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    image_path = db.Column(db.String(150), nullable=False)
    upload_time = db.Column(db.DateTime, default=db.func.current_timestamp())
    predictions = db.relationship('Prediction', backref='image', lazy=True)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image_id = db.Column(db.Integer, db.ForeignKey('image.id'), nullable=False)
    prediction_result = db.Column(db.String(50), nullable=False)
    prediction_confidence = db.Column(db.Float, nullable=False)
    prediction_time = db.Column(db.DateTime, default=db.func.current_timestamp())

model = load_model('model/diabetes_detection_model.h5')

def preprocess_image(file_path):
    img = image.load_img(file_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        img_array = preprocess_image(file_path)

        prediction_cache_key = f"prediction_{filename}"
        cached_prediction = cache.get(prediction_cache_key)

        if cached_prediction:
            prediction, confidence = cached_prediction
        else:
            prediction = model.predict(img_array)
            confidence = np.max(prediction)
            prediction = np.argmax(prediction, axis=1)[0]
            cache.set(prediction_cache_key, (prediction, confidence), timeout=3600)  # Cache for 1 hour

        user_id = 1  # Adjust based on your authentication system.
        new_image = Image(user_id=user_id, image_path=file_path)
        db.session.add(new_image)
        db.session.commit()

        new_prediction = Prediction(image_id=new_image.id, prediction_result=str(prediction), prediction_confidence=confidence)
        db.session.add(new_prediction)
        db.session.commit()

        return render_template('index.html', prediction=prediction, confidence=confidence, filename=filename)

if __name__ == '__main__':
    db.create_all()
    app.run(debug=True)

