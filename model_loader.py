import os
import gdown

MODEL_PATH = "models/mining_detector.h5"
MODEL_URL = "https://drive.google.com/uc?id=1DIU0L36frr4P6d_Ry_JW9LMuivL-x-rY"  # replace ID

def ensure_model():
    os.makedirs("models", exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        print("Downloading ML model...")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    else:
        print("Model already exists")
