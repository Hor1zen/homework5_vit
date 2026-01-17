import numpy as np
from dinov2_numpy import Dinov2Numpy
from preprocess_image import resize_short_side

def compute_cosine_similarity(v1, v2):
    """
    计算两个向量的余弦相似度
    """
    v1 = v1.flatten()
    v2 = v2.flatten()
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    # 加上 1e-8 防止除以 0
    return np.dot(v1, v2) / (norm1 * norm2 + 1e-8)

class CatDogClassifier:
    def __init__(self, model_weights_path, ref_feature_path):
        """
        初始化流程：
        1. 加载 DINOv2 模型权重。
        2. 加载老师提供的标准参考特征 (猫和狗的原型向量)。
        """
        print("🚀 [Init] 正在加载模型权重...")
        weights = np.load(model_weights_path)
        self.vit = Dinov2Numpy(weights)
        
        print("📂 [Init] 正在加载参考特征 (Reference Features)...")
        # 加载形状为 (2, 768) 的 npy 文件
        # 约定：index 0 是猫, index 1 是狗
        ref_feats = np.load(ref_feature_path)
        self.ref_cat = ref_feats[0] 
        self.ref_dog = ref_feats[1]
        print("✅ 初始化完成！")

    def predict(self, image_path):
        """
        预测流程：
        1. 预处理：使用 resize_short_side 保留完整视野。
        2. 推理：调用 DINOv2 (包含 Multi-Head Attention) 提取特征。
        3. 对比：计算当前图片特征与 猫/狗 原型的相似度。
        4. 决策：输出相似度更高的类别。
        """
        # --- Step 1: Preprocess (Resize & Normalize) ---
        # 使用 resize_short_side 而不是 center_crop
        # 优势：保留了图片的长宽比和更多内容，只要长宽是14的倍数即可
        pixel_values = resize_short_side(image_path) 
        # input shape 可能为 (1, 3, 224, 294) 等非正方形

        # --- Step 2: Model Inference (Forward Pass) ---
        # 模型内部会自动调用 interpolate_pos_encoding 处理非正方形输入
        # 经过 PatchEmbed -> Transformer Blocks (Multi-Head Attn) -> Norm
        current_feat = self.vit(pixel_values) # Output shape: (1, 768)

        # --- Step 3: Similarity Calculation ---
        score_cat = compute_cosine_similarity(current_feat, self.ref_cat)
        score_dog = compute_cosine_similarity(current_feat, self.ref_dog)

        # --- Step 4: Final Decision ---
        result = {
            "scores": {"cat": score_cat, "dog": score_dog},
            "winner": "🐱 CAT" if score_cat > score_dog else "🐶 DOG",
            "confidence": max(score_cat, score_dog)
        }
        return result