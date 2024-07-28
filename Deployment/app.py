from flask import Flask, request, jsonify, render_template, redirect, flash, session, url_for
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask_sqlalchemy import SQLAlchemy
from flask_caching import Cache
import redis
from flask_migrate import Migrate
from functools import wraps

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'supersecretkey')
app.config['UPLOAD_FOLDER'] = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max file size
# Use PostgreSQL for Render deployment
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'postgresql://postgres:12345678@localhost/Diabetic_Retinopathy')
app.config['CACHE_TYPE'] = 'RedisCache'
app.config['CACHE_REDIS_URL'] = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
cache = Cache(app)
migrate = Migrate(app, db)

class WebsiteUser(db.Model):
    __tablename__ = 'website_user'
    
    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(150), nullable=False)
    last_name = db.Column(db.String(150), nullable=False)
    username = db.Column(db.String(150), nullable=False, unique=True)
    email = db.Column(db.String(150), nullable=False, unique=True)
    password_hash = db.Column(db.String(150), nullable=False)
    user_type = db.Column(db.String(50), nullable=False)
    terms_accepted = db.Column(db.Boolean, nullable=False)
    profile_picture = db.Column(db.String(150))
    age = db.Column(db.Integer)
    images = db.relationship('Image', backref='website_user', lazy=True)

class Image(db.Model):
    __tablename__ = 'image'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('website_user.id'), nullable=False)
    image_path = db.Column(db.String(150), nullable=False)
    upload_time = db.Column(db.DateTime, default=db.func.current_timestamp())
    predictions = db.relationship('Prediction', backref='image', lazy=True)

class Prediction(db.Model):
    __tablename__ = 'prediction'
    
    id = db.Column(db.Integer, primary_key=True)
    image_id = db.Column(db.Integer, db.ForeignKey('image.id'), nullable=False)
    prediction_result = db.Column(db.String(50), nullable=False)
    prediction_confidence = db.Column(db.Float, nullable=False)
    prediction_time = db.Column(db.DateTime, default=db.func.current_timestamp())

@app.route('/check-db')
def check_db():
    try:
        # Performing a simple query to check the connection
        users = WebsiteUser.query.all()
        return jsonify({"status": "success", "data": [user.username for user in users]})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

model = load_model('model/diabetes_detection_model.h5')

def preprocess_image(file_path):
    img = image.load_img(file_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        username = request.form['username']
        email = request.form['email']
        password = generate_password_hash(request.form['password'])
        user_type = request.form['user_type']
        terms = 'terms' in request.form
        profile_picture = request.files['profile_picture']
        age = request.form.get('age')

        if profile_picture:
            filename = secure_filename(profile_picture.filename)
            profile_picture.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        new_user = WebsiteUser(
            first_name=first_name,
            last_name=last_name,
            username=username,
            email=email,
            password_hash=password,
            user_type=user_type,
            terms_accepted=terms,
            profile_picture=filename if profile_picture else None,
            age=age
        )

        db.session.add(new_user)
        db.session.commit()

        flash('Registered successfully!')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = WebsiteUser.query.filter_by(email=email).first()

        if user:
            if check_password_hash(user.password_hash, password):
                session['user_id'] = user.id
                flash('Logged in successfully!')
                return redirect(url_for('upload_file'))
            else:
                flash('Invalid password. Please try again.')
                return redirect(url_for('login'))
        else:
            flash('Email not registered. Please register first.')
            return redirect(url_for('register'))

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('Logged out successfully!')
    return redirect(url_for('login'))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST','GET'])
@login_required
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Retrieve the user_id from the current session
        user_id = session['user_id']

        # Preprocess the image
        img_array = preprocess_image(filepath)

        # Perform the prediction
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = prediction[0][predicted_class]

        # Create new Image record
        new_image = Image(user_id=user_id, image_path=filepath)

        # Add to the session and commit
        db.session.add(new_image)
        db.session.flush()  # Get the new image id before committing

        # Create new Prediction record
        new_prediction = Prediction(
            image_id=new_image.id,
            prediction_result=str(predicted_class),
            prediction_confidence=float(confidence)
        )

        db.session.add(new_prediction)
        db.session.commit()

        return jsonify({"prediction": predicted_class, "confidence": confidence})
    return jsonify({"error": "File not allowed"}), 400


if __name__ == '__main__':
    db.create_all()
    app.run(debug=True)