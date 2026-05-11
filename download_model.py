import os
import sys
import requests

def is_render_environment():
    """Check if we're running on Render."""
    # Render sets these environment variables
    return os.environ.get('RENDER', False) or \
           os.environ.get('RENDER_EXTERNAL_URL') is not None or \
           os.path.exists('/opt/render')

def download_model():
    """
    Download VGG16 model from Google Drive.
    Only runs on Render - skips locally since you have the model.
    """
    
    # Skip download if running locally
    if not is_render_environment():
        print("📍 Running locally - using existing model file")
        return True
    
    print("🌐 Render environment detected - downloading model...")
    
    # Your Google Drive File ID - REPLACE THIS
    FILE_ID = "YOUR_GOOGLE_DRIVE_FILE_ID"  # <-- CHANGE THIS
    
    MODEL_DIR = "models"
    MODEL_PATH = os.path.join(MODEL_DIR, "VGG16_best.h5")
    
    # Create models directory
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Check if already downloaded
    if os.path.exists(MODEL_PATH):
        size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        print(f"✅ Model already downloaded ({size_mb:.1f} MB)")
        return True
    
    print(f"🔄 Downloading VGG16 model from Google Drive...")
    print(f"   File ID: {FILE_ID}")
    
    # Google Drive direct download URL
    url = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
    
    try:
        session = requests.Session()
        
        # First attempt - might need confirmation for large files
        response = session.get(url, stream=True, timeout=300)
        
        # Check if Google shows confirmation page
        if 'text/html' in response.headers.get('Content-Type', ''):
            print("   Handling large file confirmation...")
            
            # Extract confirmation token
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    url = f"{url}&confirm={value}"
                    response = session.get(url, stream=True, timeout=300)
                    break
        
        # Download with progress
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        print(f"   Total size: {total_size / (1024*1024):.1f} MB")
        print("   Downloading: ", end='')
        
        with open(MODEL_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r   Downloading: {percent:.0f}%", end='')
        
        print("\n   Verifying download...")
        
        # Verify file
        if os.path.exists(MODEL_PATH):
            size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
            
            if size_mb > 50:  # Should be ~111 MB
                print(f"✅ Model download complete! ({size_mb:.1f} MB)")
                return True
            else:
                print(f"⚠️  Downloaded file too small ({size_mb:.1f} MB)")
                os.remove(MODEL_PATH)
                return False
        else:
            print("❌ File not found after download")
            return False
            
    except requests.exceptions.Timeout:
        print("\n❌ Download timed out")
        return False
    except Exception as e:
        print(f"\n❌ Download error: {e}")
        return False

if __name__ == "__main__":
    print("\n🧠 NeuroScan AI - Model Setup")
    print("=" * 50)
    
    if is_render_environment():
        print("🌐 Running on Render")
        success = download_model()
        if success:
            print("✅ Ready to start application")
            sys.exit(0)
        else:
            print("❌ Model download failed")
            sys.exit(1)
    else:
        print("💻 Running locally - using existing model")
        sys.exit(0)
