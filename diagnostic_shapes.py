import os
import pickle
import numpy as np
from sb3_contrib import RecurrentPPO

def check_file(name, path):
    print(f"\n--- Checking {name} ---")
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return
    
    if path.endswith(".zip"):
        try:
            model = RecurrentPPO.load(path)
            print(f"Model {path} Observation Space: {model.observation_space}")
        except Exception as e:
            print(f"Error loading model {path}: {e}")
    elif path.endswith(".pkl"):
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
                if isinstance(data, dict) and "obs_rms" in data:
                    print(f"Normalizer {path} Stats Shape: {data['obs_rms'].mean.shape}")
                else:
                    print(f"PKL {path} content type: {type(data)}")
        except Exception as e:
            print(f"Error loading PKL {path}: {e}")

check_file("V5 Model", "model_eurusd_titany_v5_ultimate.zip")
check_file("V3 Model", "model_eurusd_titany_v3_pro_max.zip")
check_file("Best Model", "best_models/best_model.zip")
check_file("Main Normalizer", "vec_normalize.pkl")
check_file("Backup Normalizer", "vec_normalize_backup_20260129_182026.pkl")
