import os
import numpy as np
import time
from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile

from .apps import SearchConfig
# Assuming preprocess_image.py is in the project root (sys.path)
from preprocess_image import resize_short_side

def index(request):
    if request.method == 'POST':
        # 1. Check if file is uploaded
        if 'query_img' not in request.FILES:
            return render(request, 'search/index.html', {'error': 'No file selected'})
        
        uploaded_file = request.FILES['query_img']
        
        # 2. Save uploaded file temporarily to disk
        # We need a physical path for resize_short_side
        # Create media directory if it doesn't exist
        media_root = getattr(settings, 'MEDIA_ROOT', os.path.join(settings.BASE_DIR, 'media'))
        if not os.path.exists(media_root):
            os.makedirs(media_root)
            
        temp_filename = f"temp_{int(time.time())}.jpg"
        temp_file_path = os.path.join(media_root, temp_filename)
        
        with open(temp_file_path, 'wb+') as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)
                
        try:
            # 3. Preprocess the image
            # resize_short_side returns a tensor of shape (1, 3, H, W)
            input_tensor = resize_short_side(temp_file_path)
            
            # 4. Inference
            if SearchConfig.model is None:
                return render(request, 'search/index.html', {'error': 'Model not loaded'})
                
            # run model
            feature = SearchConfig.model(input_tensor)
            feature = feature.flatten()
            
            # Normalize query feature
            norm = np.linalg.norm(feature)
            if norm > 0:
                feature = feature / norm
                
            # 5. Cosine Similarity search
            # SearchConfig.index_features shape: (N, 768)
            # feature shape: (768,)
            # dot product gives cosine similarity if vectors are normalized
            scores = np.dot(SearchConfig.index_features, feature)
            
            # 6. Top-K results
            top_k = 10
            # Get indices of top k scores (descending order)
            top_indices = np.argsort(scores)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                score = scores[idx]
                original_path = SearchConfig.index_paths[idx]
                
                # 7. Convert file path to URL
                # Original path example: ./downloaded_images/001.jpg
                # Target URL: /static/gallery/001.jpg
                
                # Normalize slashes first
                clean_path = original_path.replace("\\", "/")
                
                # Replace the directory part
                if clean_path.startswith("./downloaded_images"):
                     # Remove ./downloaded_images and prepend /static/gallery
                     # path[19:] skips "./downloaded_images"
                     rel_path = clean_path[19:] 
                     if rel_path.startswith("/"):
                         rel_path = rel_path[1:]
                     url_path = f"/static/gallery/{rel_path}"
                elif "downloaded_images" in clean_path:
                    # Fallback if path is absolute or different
                    part = clean_path.split("downloaded_images")[-1]
                    if part.startswith("/"):
                        part = part[1:]
                    url_path = f"/static/gallery/{part}"
                else:
                    url_path = clean_path # Should not happen based on logic
                
                results.append({
                    'image_url': url_path,
                    'score': f"{score:.4f}"
                })
                
            return render(request, 'search/index.html', {'results': results})
            
        except Exception as e:
            print(f"Error processing request: {e}")
            return render(request, 'search/index.html', {'error': str(e)})
            
        finally:
            # Cleanup temp file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    else:
        # GET request
        return render(request, 'search/index.html')
