"""
Test Script - Verify Backend Setup
Run this to make sure everything is working
"""

import os
import sys

print("\n" + "="*60)
print("🧪 TESTING BACKEND SETUP")
print("="*60 + "\n")

# =====================================================
# TEST 1: Check Files Exist
# =====================================================

print("TEST 1: Checking files...")

files_to_check = [
    "main.py",
    "database.py",
    "requirements.txt"
]

all_files_exist = True
for file in files_to_check:
    if os.path.exists(file):
        print(f"  ✅ {file}")
    else:
        print(f"  ❌ {file} - MISSING!")
        all_files_exist = False

# Check model
model_path = "models/mining_detector_final.h5"
alt_model_path = "../models/mining_detector_final.h5"

if os.path.exists(model_path):
    print(f"  ✅ {model_path}")
    print(f"     Size: {os.path.getsize(model_path) / (1024*1024):.1f} MB")
elif os.path.exists(alt_model_path):
    print(f"  ✅ {alt_model_path}")
    print(f"     Size: {os.path.getsize(alt_model_path) / (1024*1024):.1f} MB")
else:
    print(f"  ⚠️  Model not found at {model_path}")
    print(f"     Please download from Google Drive!")
    all_files_exist = False

if not all_files_exist:
    print("\n❌ Some files are missing! Check SETUP_GUIDE.md")
    sys.exit(1)

print("\n✅ All files present!\n")

# =====================================================
# TEST 2: Check Python Dependencies
# =====================================================

print("TEST 2: Checking Python dependencies...")

required_packages = {
    "fastapi": "0.104",
    "uvicorn": "0.24",
    "tensorflow": "2.15",
    "numpy": "1.24",
    "PIL": "10.0",  # Pillow
    "pydantic": "2.4"
}

missing_packages = []

for package, min_version in required_packages.items():
    try:
        if package == "PIL":
            import PIL
            version = PIL.__version__
        else:
            module = __import__(package)
            version = module.__version__
        print(f"  ✅ {package} (version {version})")
    except ImportError:
        print(f"  ❌ {package} - NOT INSTALLED!")
        missing_packages.append(package)
    except AttributeError:
        print(f"  ✅ {package} (version check skipped)")

if missing_packages:
    print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
    print("   Run: pip install -r requirements.txt")
    sys.exit(1)

print("\n✅ All dependencies installed!\n")

# =====================================================
# TEST 3: Test Database Module
# =====================================================

print("TEST 3: Testing database...")

try:
    from database import init_db, save_detection, get_all_detections, get_statistics
    
    # Initialize
    init_db()
    print("  ✅ Database initialized")
    
    # Save test detection
    test_id = save_detection(
        latitude=23.74,
        longitude=86.41,
        confidence=87.5,
        severity="Critical",
        mining_type="Coal",
        area_hectares=145.5,
        estimated_loss_usd=12400000,
        location_name="Test Site",
        image_filename="test.jpg",
        reasoning="Test detection"
    )
    print(f"  ✅ Saved test detection (ID: {test_id})")
    
    # Retrieve
    detections = get_all_detections(limit=10)
    print(f"  ✅ Retrieved {len(detections)} detections")
    
    # Statistics
    stats = get_statistics()
    print(f"  ✅ Statistics: {stats['total_detections']} total detections")
    
    print("\n✅ Database working!\n")
    
except Exception as e:
    print(f"\n❌ Database test failed: {e}\n")
    sys.exit(1)

# =====================================================
# TEST 4: Test Model Loading
# =====================================================

print("TEST 4: Testing model loading...")

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    
    # Try to load model
    if os.path.exists(model_path):
        model = load_model(model_path)
    elif os.path.exists(alt_model_path):
        model = load_model(alt_model_path)
    else:
        raise FileNotFoundError("Model not found")
    
    print(f"  ✅ Model loaded successfully")
    print(f"     Input shape: {model.input_shape}")
    print(f"     Output shape: {model.output_shape}")
    
    # Test prediction with dummy data
    import numpy as np
    dummy_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
    prediction = model.predict(dummy_input, verbose=0)
    print(f"  ✅ Model prediction works")
    print(f"     Test output shape: {prediction.shape}")
    
    print("\n✅ Model loading successful!\n")
    
except FileNotFoundError:
    print("  ⚠️  Model file not found")
    print("     Download from Google Drive:")
    print("     illegal_mining_detection/models/mining_detector.h5")
    print("     (Server will start but detection won't work)")
    print("")
    
except Exception as e:
    print(f"\n❌ Model loading failed: {e}\n")
    print("   Check that the model file is valid")
    sys.exit(1)

# =====================================================
# TEST 5: Test Import Main App
# =====================================================

print("TEST 5: Testing main application import...")

try:
    from main import app
    print("  ✅ FastAPI app imported successfully")
    
    # Check endpoints
    routes = [route.path for route in app.routes]
    print(f"  ✅ Found {len(routes)} endpoints:")
    for route in routes[:5]:  # Show first 5
        print(f"     - {route}")
    if len(routes) > 5:
        print(f"     ... and {len(routes) - 5} more")
    
    print("\n✅ Main application ready!\n")
    
except Exception as e:
    print(f"\n❌ App import failed: {e}\n")
    sys.exit(1)

# =====================================================
# FINAL SUMMARY
# =====================================================

print("="*60)
print("🎉 ALL TESTS PASSED!")
print("="*60)
print("\n✅ Your backend is ready to run!")
print("\n📝 Next steps:")
print("   1. Run: python main.py")
print("   2. Open: http://localhost:8000/api/docs")
print("   3. Test detection endpoint with an image")
print("   4. Connect your frontend!")
print("\n" + "="*60 + "\n")
