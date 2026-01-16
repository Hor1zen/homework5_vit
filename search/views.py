import os
import sys
import numpy as np
import time
import json

from django.shortcuts import render, redirect
from django.conf import settings
from django.contrib.auth import login, logout
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.decorators import login_required
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.apps import apps
from django.http import HttpResponse

# Ensure project root is in sys.path to import modules from root
sys.path.append(str(settings.BASE_DIR))

try:
    from preprocess_image import resize_short_side
except ImportError:
    # If it fails, we might be in a different context, but usually this works
    pass

from .models import SearchRecord

def index(request):
    """
    Search page: handles image upload, inference, search, and history saving.
    """
    context = {}
    
    # Get model and index from global AppHeader
    search_app = apps.get_app_config('search')
    model = search_app.model
    index_features = search_app.index_features
    index_paths = search_app.index_paths
    
    is_ready = (model is not None) and (index_features is not None) and (index_paths is not None)
    
    if not is_ready:
        context['error'] = "System is initializing or index files are missing. Please try again later."
        return render(request, 'search/index.html', context)

    if request.method == 'POST' and request.FILES.get('query_img'):
        started_at = time.time()
        uploaded_file = request.FILES['query_img']
        
        # 1. Save uploaded file temporarily to process it
        # We need an absolute path for the preprocessing script
        # Using default_storage to save to 'tmp' folder in media
        # default_storage.save returns the relative path inside MEDIA_ROOT
        path = default_storage.save(f"tmp/{uploaded_file.name}", ContentFile(uploaded_file.read()))
        abs_file_path = os.path.join(settings.MEDIA_ROOT, path)
        
        try:
            # 2. Preprocess
            input_tensor = resize_short_side(abs_file_path)
            
            # 3. Inference
            feature = model(input_tensor).flatten()
            
            # Normalize
            norm = np.linalg.norm(feature)
            feature = feature / (norm + 1e-8)
            
            # 4. Search (Cosine Similarity)
            scores = np.dot(index_features, feature)
            
            # 5. Top-K
            top_k = 10
            top_indices = np.argsort(scores)[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                score = scores[idx]
                original_path = index_paths[idx]
                
                # Read title from corresponding .txt file
                txt_path = os.path.splitext(original_path)[0] + '.txt'
                title = ''
                try:
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        title = f.read().strip()
                except:
                    title = os.path.basename(original_path)  # fallback to filename
                
                # 6. Path Cleaning & URL Construction
                # Convert: "./downloaded_images/foo.jpg" -> "/static/gallery/foo.jpg"
                clean_path = original_path.replace("\\", "/") # Windows fix
                
                # Handle leading dots
                if clean_path.startswith("./"):
                    clean_path = clean_path[2:]
                
                # Construct URL based on known structure
                # We assume 'downloaded_images' is mapped to 'static/gallery/' or similar
                # If path contains 'downloaded_images/', we strip it and prepend static URL
                if "downloaded_images/" in clean_path:
                    # extract relative part after downloaded_images/
                    rel_part = clean_path.split("downloaded_images/")[-1]
                    url_path = f"{settings.STATIC_URL}gallery/{rel_part}"
                else:
                    # Fallback, maybe it's just a filename
                    url_path = f"{settings.STATIC_URL}gallery/{clean_path}"
                
                results.append({
                    'image_url': url_path,
                    'score': f"{score:.4f}",
                    'path': original_path,
                    'title': title
                })
            
            context['results'] = results
            context['query_url'] = f"{settings.MEDIA_URL}{path}"
            
            # 7. Save History (if logged in)
            if request.user.is_authenticated:
                latency = (time.time() - started_at) * 1000
                
                # We need to read the file content again or copy it
                with open(abs_file_path, 'rb') as f:
                    SearchRecord.objects.create(
                        user=request.user,
                        query_image=ContentFile(f.read(), name=uploaded_file.name),
                        results_data=results,
                        latency_ms=latency
                    )
                    
        except Exception as e:
            print(f"Error processing request: {e}")
            context['error'] = str(e)
            
        # Optional: Clean up temp file? 
        # For history, we already saved a copy to 'query_images/' folder via the model.
        # So we can delete the temp file in 'media/tmp/' if we want to save space.
        # try:
        #     os.remove(abs_file_path)
        # except:
        #     pass

    return render(request, 'search/index.html', context)

def register_view(request):
    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect("index")
    else:
        form = UserCreationForm()
    return render(request, "search/register.html", {"form": form})

def login_view(request):
    if request.method == "POST":
        form = AuthenticationForm(data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            return redirect("index")
    else:
        form = AuthenticationForm()
    return render(request, "search/login.html", {"form": form})

def logout_view(request):
    logout(request)
    return redirect("index")

@login_required
def history(request):
    records = SearchRecord.objects.filter(user=request.user).order_by('-created_at')
    return render(request, 'search/history.html', {'records': records})

