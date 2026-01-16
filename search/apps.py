import os
import sys
import numpy as np
from django.apps import AppConfig
from django.conf import settings

class SearchConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'search'

    # Global variables to store the model and index
    model = None
    index_features = None
    index_paths = None

    def ready(self):
        # 1. Add project root to sys.path so we can import 'dinov2_numpy'
        root_dir = str(settings.BASE_DIR)
        if root_dir not in sys.path:
            sys.path.append(root_dir)

        print("🚀 [SearchApp] Initializing DINOv2 engine...")
        
        try:
            # Import modules from root
            from dinov2_numpy import Dinov2Numpy
            
            # 2. Load Model Weights
            weights_path = os.path.join(root_dir, "vit-dinov2-base.npz")
            if os.path.exists(weights_path):
                weights = np.load(weights_path)
                SearchConfig.model = Dinov2Numpy(weights)
                print("✅ [SearchApp] Model loaded successfully.")
            else:
                print(f"⚠️ [SearchApp] Weights not found at {weights_path}")

            # 3. Load Index
            feat_path = os.path.join(root_dir, "gallery_features.npy")
            path_path = os.path.join(root_dir, "gallery_paths.npy")

            if os.path.exists(feat_path) and os.path.exists(path_path):
                SearchConfig.index_features = np.load(feat_path)
                SearchConfig.index_paths = np.load(path_path)
                print(f"✅ [SearchApp] Index loaded. {len(SearchConfig.index_paths)} items.")
            else:
                print("⚠️ [SearchApp] Index files not found. Search will not work.")

        except Exception as e:
            print(f"❌ [SearchApp] Init failed: {e}")
            # Do not raise error to avoid blocking 'migrate' or other commands
