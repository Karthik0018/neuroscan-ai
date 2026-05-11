# app.py - NeuroScan AI with Supabase Integration
import os
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
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================
# FLASK APP CONFIGURATION
# ============================================================
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'brain-tumor-mri-secret-key-2024-secure')
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ============================================================
# SUPABASE CONFIGURATION
# ============================================================
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")  # anon key
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")  # service_role key

def get_supabase() -> Client:
    """Get standard Supabase client."""
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def get_supabase_admin() -> Client:
    """Get admin Supabase client with full privileges."""
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# Initialize Supabase clients
supabase = get_supabase()
supabase_admin = get_supabase_admin()

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

# Load TensorFlow model
try:
    MODEL_PATH = 'models/VGG16_best.h5'
    if os.path.exists(MODEL_PATH):
        print("🔄 Loading VGG16 model...")
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
    """Check if uploaded file has allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image_path):
    """Preprocess MRI image for model prediction."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def login_required(f):
    """Decorator to protect routes that require authentication."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login first.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def create_user_tables():
    """Create necessary tables in Supabase if they don't exist."""
    try:
        # Create users table
        supabase_admin.table('users').select('*').limit(1).execute()
        print("✅ Users table exists")
    except Exception:
        # If table doesn't exist, create it via SQL
        print("⚠️  Users table not found. Please create it in Supabase SQL Editor:")
        print("""
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username VARCHAR(150) UNIQUE NOT NULL,
            email VARCHAR(255) UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """)
    
    try:
        # Create predictions table
        supabase_admin.table('predictions').select('*').limit(1).execute()
        print("✅ Predictions table exists")
    except Exception:
        print("⚠️  Predictions table not found. Please create it in Supabase SQL Editor:")
        print("""
        CREATE TABLE IF NOT EXISTS predictions (
            id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
            filename TEXT,
            prediction VARCHAR(50),
            confidence REAL,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """)

# ============================================================
# ROUTES
# ============================================================

@app.route('/')
def home():
    """Home page route."""
    return render_template('home.html')

@app.route('/about')
def about():
    """About page route."""
    return render_template('about.html')

@app.route('/methodology')
def methodology():
    """Methodology page route."""
    return render_template('methodology.html')

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    """Prediction route - handles MRI image upload and classification."""
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file uploaded.', 'danger')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No file selected.', 'danger')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            # Generate unique filename
            filename = secure_filename(
                f"{session['user_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
            )
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            if model_loaded:
                try:
                    # Preprocess and predict
                    img_array = preprocess_image(filepath)
                    predictions = model.predict(img_array, verbose=0)[0]
                    pred_idx = np.argmax(predictions)
                    pred_class = CLASSES[pred_idx]
                    confidence = float(predictions[pred_idx] * 100)
                    
                    # Get disease information
                    disease_info = DISEASE_INFO.get(pred_class, {})
                    
                    # Save prediction to Supabase
                    try:
                        supabase_admin.table('predictions').insert({
                            'user_id': session['user_id'],
                            'filename': filename,
                            'prediction': pred_class,
                            'confidence': confidence
                        }).execute()
                    except Exception as e:
                        print(f"⚠️ Error saving prediction: {e}")
                        # Continue even if save fails
                    
                    # Prepare probability data for template
                    probas = {
                        cls: float(predictions[i] * 100) 
                        for i, cls in enumerate(CLASSES)
                    }
                    
                    return render_template('predict.html', 
                                         prediction=pred_class,
                                         confidence=round(confidence, 2),
                                         disease_info=disease_info,
                                         probas=probas,
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
    """Login route."""
    if 'user_id' in session:
        return redirect(url_for('predict'))
    
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        
        if not email or not password:
            flash('Please fill in all fields.', 'danger')
            return render_template('login.html')
        
        try:
            # Find user by email in Supabase
            result = supabase_admin.table('users') \
                .select('*') \
                .eq('email', email) \
                .execute()
            
            if result.data and len(result.data) > 0:
                user = result.data[0]
                
                # Check password
                if check_password_hash(user['password'], password):
                    # Store user info in session
                    session['user_id'] = user['id']
                    session['username'] = user['username']
                    
                    flash(f'Welcome back, {user["username"]}!', 'success')
                    return redirect(url_for('predict'))
            
            flash('Invalid email or password.', 'danger')
            
        except Exception as e:
            print(f"❌ Login error: {e}")
            flash('Login failed. Please try again.', 'danger')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Registration route."""
    if 'user_id' in session:
        return redirect(url_for('predict'))
    
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        
        # Validation
        if not all([username, email, password]):
            flash('All fields are required.', 'danger')
            return render_template('register.html')
        
        if password != confirm_password:
            flash('Passwords do not match.', 'danger')
            return render_template('register.html')
        
        if len(password) < 6:
            flash('Password must be at least 6 characters.', 'danger')
            return render_template('register.html')
        
        try:
            # Check if user exists in Supabase
            existing = supabase_admin.table('users') \
                .select('*') \
                .or_(f'email.eq.{email},username.eq.{username}') \
                .execute()
            
            if existing.data and len(existing.data) > 0:
                # Check which field exists
                for user in existing.data:
                    if user['email'] == email:
                        flash('Email already registered.', 'danger')
                    elif user['username'] == username:
                        flash('Username already taken.', 'danger')
                return render_template('register.html')
            
            # Hash password and create user
            hashed_password = generate_password_hash(password)
            
            new_user = supabase_admin.table('users').insert({
                'username': username,
                'email': email,
                'password': hashed_password
            }).execute()
            
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
            
        except Exception as e:
            print(f"❌ Registration error: {e}")
            flash('Registration failed. Please try again.', 'danger')
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    """Logout route."""
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('home'))

# @app.route('/history')
# @login_required
# def history():
#     """View prediction history."""
#     try:
#         result = supabase_admin.table('predictions') \
#             .select('*') \
#             .eq('user_id', session['user_id']) \
#             .order('created_at', desc=True) \
#             .execute()
        
#         predictions = result.data if result.data else []
        
#     except Exception as e:
#         print(f"❌ Error fetching history: {e}")
#         predictions = []
#         flash('Could not load prediction history.', 'warning')
    
#     return render_template('history.html', predictions=predictions)

@app.route('/health')
def health_check():
    """Health check endpoint for Render."""
    try:
        # Test Supabase connection
        supabase.table('users').select('count', count='exact').execute()
        return jsonify({
            'status': 'healthy',
            'database': 'connected',
            'model_loaded': model_loaded
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

# Replace your existing error handlers with these:
@app.errorhandler(404)
def not_found(e):
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>404 - Page Not Found | NeuroScan AI</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    </head>
    <body class="bg-light">
        <div class="container text-center py-5">
            <h1 class="display-1 text-muted">404</h1>
            <h2>Page Not Found</h2>
            <p class="lead">The page you're looking for doesn't exist.</p>
            <a href="/" class="btn btn-primary">Return Home</a>
        </div>
    </body>
    </html>
    ''', 404

@app.errorhandler(500)
def server_error(e):
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>500 - Server Error | NeuroScan AI</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    </head>
    <body class="bg-light">
        <div class="container text-center py-5">
            <h1 class="display-1 text-muted">500</h1>
            <h2>Server Error</h2>
            <p class="lead">Something went wrong. Please try again later.</p>
            <a href="/" class="btn btn-primary">Return Home</a>
        </div>
    </body>
    </html>
    ''', 500

# ============================================================
# RUN APPLICATION
# ============================================================
if __name__ == '__main__':
    print("\n🧠 NeuroScan AI - Starting up...")
    print("=" * 50)
    
    # Check environment variables
    if SUPABASE_URL and SUPABASE_KEY:
        print("✅ Supabase credentials found")
        # Test connection
        try:
            supabase.table('users').select('count', count='exact').execute()
            print("✅ Connected to Supabase successfully!")
        except Exception as e:
            print(f"⚠️  Supabase connection failed: {e}")
            print("   Make sure tables are created in Supabase SQL Editor")
    else:
        print("⚠️  Supabase credentials not set. Using local storage.")
    
    print(f"🔍 Model loaded: {model_loaded}")
    print(f"📁 Upload folder: {app.config['UPLOAD_FOLDER']}")
    print("=" * 50)
    
    os.makedirs('instance', exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)