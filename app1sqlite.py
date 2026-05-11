# app.py
import os
import sqlite3
from datetime import datetime
from functools import wraps
from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, session, jsonify
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

# ============================================================
# FLASK APP CONFIGURATION
# ============================================================
app = Flask(__name__)
app.secret_key = 'brain-tumor-mri-secret-key-2024-secure'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ============================================================
# DATABASE SETUP
# ============================================================
def init_db():
    conn = sqlite3.connect('instance/users.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            filename TEXT,
            prediction TEXT,
            confidence REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def get_db():
    conn = sqlite3.connect('instance/users.db')
    conn.row_factory = sqlite3.Row
    return conn

# ============================================================
# MODEL LOADING
# ============================================================
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Disease information dictionary
DISEASE_INFO = {
    'glioma': {
        'name': 'Glioma',
        'description': 'A glioma is a type of tumor that occurs in the brain and spinal cord. Gliomas originate in the glial cells that surround and support neurons.',
        'types': ['Astrocytoma', 'Oligodendroglioma', 'Ependymoma', 'Glioblastoma'],
        'symptoms': [
            'Persistent headaches', 'Seizures', 'Memory loss',
            'Changes in personality', 'Vision problems',
            'Difficulty with balance', 'Speech difficulties'
        ],
        'precautions': [
            'Regular neurological check-ups', 'Avoid radiation exposure when possible',
            'Maintain a healthy lifestyle', 'Follow prescribed medication schedules',
            'Regular MRI monitoring',
            'Avoid smoking and excessive alcohol consumption'
        ],
        'treatment': [
            'Surgery to remove the tumor',
            'Radiation therapy',
            'Chemotherapy',
            'Targeted drug therapy',
            'Rehabilitation therapy'
        ],
        'source': 'World Health Organization (WHO) & American Brain Tumor Association'
    },
    'meningioma': {
        'name': 'Meningioma',
        'description': 'A meningioma is a tumor that forms on the membranes covering the brain and spinal cord (meninges). Most meningiomas are benign (non-cancerous).',
        'types': ['Grade I (Benign)', 'Grade II (Atypical)', 'Grade III (Anaplastic/Malignant)'],
        'symptoms': [
            'Headaches that worsen over time',
            'Changes in vision', 'Hearing loss',
            'Memory problems', 'Weakness in limbs',
            'Seizures (in some cases)'
        ],
        'precautions': [
            'Regular brain imaging (MRI/CT)',
            'Monitor for new or worsening symptoms',
            'Maintain blood pressure control',
            'Avoid head injuries',
            'Follow-up with neurosurgeon regularly'
        ],
        'treatment': [
            'Observation for small, asymptomatic tumors',
            'Surgical removal',
            'Radiation therapy (for inoperable tumors)',
            'Stereotactic radiosurgery',
            'Medication for symptom management'
        ],
        'source': 'National Institute of Neurological Disorders and Stroke (NINDS)'
    },
    'notumor': {
        'name': 'No Tumor Detected',
        'description': 'The MRI scan shows no evidence of a brain tumor. However, regular health check-ups are still recommended.',
        'types': ['N/A - No tumor present'],
        'symptoms': ['No tumor-related symptoms detected'],
        'precautions': [
            'Maintain a healthy lifestyle',
            'Regular annual health check-ups',
            'Stay physically active',
            'Eat a balanced diet rich in antioxidants',
            'Manage stress through meditation or exercise',
            'Get adequate sleep (7-8 hours)'
        ],
        'treatment': ['No treatment required', 'Continue regular health monitoring'],
        'source': 'General health guidelines from WHO'
    },
    'pituitary': {
        'name': 'Pituitary Tumor',
        'description': 'A pituitary tumor is an abnormal growth in the pituitary gland, a small gland at the base of the brain that controls hormone production.',
        'types': ['Functioning Adenomas', 'Non-functioning Adenomas', 'Microadenomas', 'Macroadenomas'],
        'symptoms': [
            'Vision problems', 'Headaches',
            'Hormonal imbalances', 'Unexplained weight changes',
            'Mood changes', 'Fatigue',
            'Irregular menstrual cycles (in women)'
        ],
        'precautions': [
            'Regular hormone level monitoring',
            'Periodic vision tests',
            'Annual MRI scans',
            'Follow endocrinologist recommendations',
            'Medication adherence',
            'Stress management techniques'
        ],
        'treatment': [
            'Medication to control hormone levels',
            'Transsphenoidal surgery',
            'Radiation therapy',
            'Hormone replacement therapy',
            'Regular monitoring'
        ],
        'source': 'American Association of Neurological Surgeons (AANS)'
    }
}

try:
    MODEL_PATH = 'models/VGG16_best.h5'
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ VGG16 model loaded successfully!")
        model_loaded = True
    else:
        print("⚠️  Model file not found. Using placeholder predictions.")
        model = None
        model_loaded = False
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    model_loaded = False

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login first.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# ============================================================
# ROUTES
# ============================================================

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/methodology')
def methodology():
    return render_template('methodology.html')

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file uploaded.', 'danger')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No file selected.', 'danger')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(f"{session['user_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            if model_loaded:
                try:
                    img_array = preprocess_image(filepath)
                    predictions = model.predict(img_array, verbose=0)[0]
                    pred_idx = np.argmax(predictions)
                    pred_class = CLASSES[pred_idx]
                    confidence = float(predictions[pred_idx] * 100)
                    
                    # Get disease info
                    disease_info = DISEASE_INFO.get(pred_class, {})
                    
                    # Save prediction to database
                    conn = get_db()
                    conn.execute(
                        'INSERT INTO predictions (user_id, filename, prediction, confidence) VALUES (?, ?, ?, ?)',
                        (session['user_id'], filename, pred_class, confidence)
                    )
                    conn.commit()
                    conn.close()
                    
                    return render_template('predict.html', 
                                         prediction=pred_class,
                                         confidence=round(confidence, 2),
                                         disease_info=disease_info,
                                         probas={cls: float(predictions[i] * 100) 
                                                for i, cls in enumerate(CLASSES)},
                                         filename=filename,
                                         model_loaded=True)
                except Exception as e:
                    flash(f'Prediction error: {str(e)}', 'danger')
                    return redirect(request.url)
            else:
                flash('Model is not loaded. Please contact administrator.', 'warning')
                return redirect(request.url)
    
    return render_template('predict.html', model_loaded=model_loaded)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        conn = get_db()
        user = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
        conn.close()
        
        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            flash(f'Welcome back, {user["username"]}!', 'success')
            return redirect(url_for('predict'))
        else:
            flash('Invalid email or password.', 'danger')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if password != confirm_password:
            flash('Passwords do not match.', 'danger')
            return redirect(url_for('register'))
        
        if len(password) < 6:
            flash('Password must be at least 6 characters.', 'danger')
            return redirect(url_for('register'))
        
        conn = get_db()
        existing_user = conn.execute(
            'SELECT * FROM users WHERE email = ? OR username = ?',
            (email, username)
        ).fetchone()
        
        if existing_user:
            flash('Email or username already exists.', 'danger')
            conn.close()
            return redirect(url_for('register'))
        
        hashed_password = generate_password_hash(password)
        conn.execute(
            'INSERT INTO users (username, email, password) VALUES (?, ?, ?)',
            (username, email, hashed_password)
        )
        conn.commit()
        conn.close()
        
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('home'))

@app.route('/history')
@login_required
def history():
    conn = get_db()
    predictions = conn.execute(
        'SELECT * FROM predictions WHERE user_id = ? ORDER BY created_at DESC',
        (session['user_id'],)
    ).fetchall()
    conn.close()
    return render_template('history.html', predictions=predictions)

# Error handlers
@app.errorhandler(404)
def not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('500.html'), 500

# ============================================================
# RUN APPLICATION
# ============================================================
if __name__ == '__main__':
    os.makedirs('instance', exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)