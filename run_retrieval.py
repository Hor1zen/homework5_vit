import os
import numpy as np
import time
from PIL import Image

# 导入我们之前写好的模块
from dinov2_numpy import Dinov2Numpy
from preprocess_image import resize_short_side

# ================= 配置区域 =================
GALLERY_DIR = "./downloaded_images"  # 图库文件夹
INDEX_FEAT_FILE = "gallery_features.npy"  # 保存特征的文件
INDEX_PATH_FILE = "gallery_paths.npy"     # 保存路径的文件
WEIGHTS_PATH = "vit-dinov2-base.npz"      # 模型权重
TOP_K = 10
# ===========================================

def load_model():
    print("🚀 Loading DINOv2 model...")
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"❌ 找不到权重文件: {WEIGHTS_PATH}")
    weights = np.load(WEIGHTS_PATH)
    model = Dinov2Numpy(weights)
    print("✅ Model loaded.")
    return model

def build_index(model):
    """
    遍历 GALLERY_DIR，提取所有图片的特征，保存到磁盘。
    """
    print(f"📂 Scanning images in {GALLERY_DIR}...")
    
    # 1. 收集所有图片路径
    image_paths = []
    supported_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.gif')
    
    if not os.path.exists(GALLERY_DIR):
        print(f"❌ 错误: 文件夹不存在 {GALLERY_DIR}")
        return None, None

    for root, _, files in os.walk(GALLERY_DIR):
        for file in files:
            if file.lower().endswith(supported_exts):
                # 强制转换为正斜杠，确保跨平台兼容性
                raw_path = os.path.join(root, file)
                clean_path = raw_path.replace('\\', '/')
                image_paths.append(clean_path)
    
    total_imgs = len(image_paths)
    if total_imgs == 0:
        print("❌ 未找到任何图片，请检查文件夹路径或后缀名。")
        return None, None

    print(f"📊 Found {total_imgs} images. Starting feature extraction...")
    
    # 2. 逐张提取特征
    all_features = []
    valid_paths = []
    failed_count = 0 
    
    start_time = time.time()
    for i, img_path in enumerate(image_paths):
        try:
            # 预处理: resize_short_side (关键步骤!)
            # shape: (1, 3, H, W)
            input_tensor = resize_short_side(img_path)
            
            # 推理: forward
            # shape: (1, 768)
            feature = model(input_tensor) 
            
            # 转为 numpy 并展平
            feature = feature.flatten() # (768,)
            
            # 归一化特征向量 (方便后续直接算点积就是余弦相似度)
            # L2 Norm: v / ||v||
            norm = np.linalg.norm(feature)
            feature = feature / (norm + 1e-8)
            
            all_features.append(feature)
            valid_paths.append(img_path)
            
        except Exception as e:
            failed_count += 1
            # 失败时不打印冗长错误，静默跳过
            pass
        
        # 打印进度
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            print(f"   [{i+1}/{total_imgs}] Success: {len(valid_paths)}, Failed: {failed_count}, Time: {elapsed:.1f}s")

    # 3. 堆叠成大矩阵
    if len(all_features) == 0:
        print("❌ No valid features extracted.")
        return None, None

    features_matrix = np.stack(all_features) # (N, 768)
    paths_array = np.array(valid_paths)
    
    # 4. 保存到磁盘
    print(f"💾 Saving index to {INDEX_FEAT_FILE}...")
    np.save(INDEX_FEAT_FILE, features_matrix)
    np.save(INDEX_PATH_FILE, paths_array)
    
    print(f"✅ Index built! Shape: {features_matrix.shape}")
    return features_matrix, paths_array

def search_image(model, query_img_path, index_features, index_paths, top_k=10):
    """
    输入一张查询图，返回图库中最相似的 Top-K 图片
    """
    print(f"\n🔍 Searching for: {query_img_path}")
    
    # 1. 提取 Query 特征
    try:
        query_input = resize_short_side(query_img_path)
        query_feat = model(query_input).flatten()
        
        # 归一化 (重要! 使得 Dot Product == Cosine Similarity)
        query_norm = np.linalg.norm(query_feat)
        query_feat = query_feat / (query_norm + 1e-8)
        
    except Exception as e:
        print(f"❌ Error loading query image: {e}")
        return

    # 2. 计算相似度 (矩阵乘法高效计算)
    # (N, 768) @ (768,) -> (N,)
    similarities = index_features @ query_feat
    
    # 3. 排序 (从大到小)
    # argsort 返回的是从小到大的索引，所以取最后 k 个并反转
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    # 4. 打印结果
    print(f"{'Rank':<5} | {'Score':<10} | {'File Path'}")
    print("-" * 50)
    
    for rank, idx in enumerate(top_indices):
        score = similarities[idx]
        file_path = index_paths[idx]
        print(f"{rank+1:<5} | {score:.4f}     | {file_path}")

def main():
    # 1. 初始化
    model = load_model()
    
    # 2. 检查是否有现成的 Index，没有就新建
    if os.path.exists(INDEX_FEAT_FILE) and os.path.exists(INDEX_PATH_FILE):
        
        # 检查是否包含反斜杠，如果有则警告或重新构建（这里简单打印警告）
        if len(index_paths) > 0 and '\\' in str(index_paths[0]):
             print("⚠️  Warning: Index paths contain backslashes. Web display might look messy.")
             
        print("📂 Loading existing index...")
        index_features = np.load(INDEX_FEAT_FILE)
        index_paths = np.load(INDEX_PATH_FILE)
        print(f"✅ Index loaded. {len(index_paths)} images indexed.")
    else:
        print("⚠️ No index found. Building from scratch...")
        index_features, index_paths = build_index(model)

    if index_features is None:
        return

    # 3. 演示检索 (Demo)
    # 我们随机选库里的一张图作为查询图，看看能不能搜到它自己
    if len(index_paths) > 0:
        # 随便挑一张，比如第 0 张
        demo_query = index_paths[0]
        # 或者你可以手动指定一张图:
        # demo_query = "./demo_data/cat.jpg" 
        
        search_image(model, demo_query, index_features, index_paths, top_k=TOP_K)

if __name__ == "__main__":
    main()
