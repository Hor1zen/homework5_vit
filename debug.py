import numpy as np
from dinov2_numpy import Dinov2Numpy
from preprocess_image import center_crop

def compute_cosine_similarity(v1, v2):
    # 把向量展平 (1, D) -> (D,)
    v1 = v1.flatten()
    v2 = v2.flatten()
    # 计算点积
    dot_product = np.dot(v1, v2)
    # 计算模长 (Norm)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    # 避免除以 0
    return dot_product / (norm1 * norm2 + 1e-8)

def run_debug():
    print("🚀 正在初始化模型 (Loading weights)...")
    # 1. 加载权重和模型
    weights = np.load("vit-dinov2-base.npz")
    vit = Dinov2Numpy(weights)

    # 2. 提取你的特征 (Your Implementation)
    print("📸 正在提取特征 (Extracting features)...")
    
    # 提取猫的特征
    cat_pixel_values = center_crop("./demo_data/cat.jpg")
    cat_feat = vit(cat_pixel_values)  # 你的代码算出来的
    
    # 提取狗的特征
    dog_pixel_values = center_crop("./demo_data/dog.jpg")
    dog_feat = vit(dog_pixel_values)  # 你的代码算出来的

    # 3. 加载标准答案 (Reference Data)
    # 这个文件里存的是老师/助教用完全正确的代码算好的特征
    try:
        ref_feats = np.load("./demo_data/cat_dog_feature.npy")
        ref_cat_feat = ref_feats[0:1] # 第一张是猫
        ref_dog_feat = ref_feats[1:2] # 第二张是狗
    except FileNotFoundError:
        print("❌ 错误: 找不到参考文件 ./demo_data/cat_dog_feature.npy")
        return

    # 4. 计算误差 (Compute Difference)
    # 我们用 L2 范数（欧氏距离）来看两个向量差得有多远
    # 也就是：sqrt( sum( (你的值 - 标准值)^2 ) )
    diff_cat = np.linalg.norm(cat_feat - ref_cat_feat)
    diff_dog = np.linalg.norm(dog_feat - ref_dog_feat)

    sim_cat = compute_cosine_similarity(cat_feat, ref_cat_feat)
    sim_dog = compute_cosine_similarity(dog_feat, ref_dog_feat)

    # 5. 打印结果
    print("\n" + "="*50)
    print("📊 Debug 结果报告")
    print("="*50)
    
    # 打印 L2 误差 (越小越好，理想值 < 1e-5)
    print(f"📉 [L2 Error] Cat: {diff_cat:.8f}")
    print(f"📉 [L2 Error] Dog: {diff_dog:.8f}")
    
    print("-" * 30)
    
    # 打印 余弦相似度 (越接近 1.0 越好)
    print(f"📈 [Cosine Sim] Cat: {sim_cat:.8f}")
    print(f"📈 [Cosine Sim] Dog: {sim_dog:.8f}")
    print("-" * 50)

    # 6. 自动判断 (逻辑不变，依然基于 L2 判断最为严格)
    threshold = 1e-4 
    if diff_cat < threshold and diff_dog < threshold:
        print("✅ DEBUG PASSED! 实现完美！")
    else:
        print("❌ DEBUG FAILED. 还需要检查代码。")
if __name__ == "__main__":
    run_debug()