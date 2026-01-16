import numpy as np
from PIL import Image

def center_crop(img_path, crop_size=224):
    # Step 1: load image
    image = Image.open(img_path).convert("RGB")

    # Step 2: center crop
    w, h = image.size
    left = (w - crop_size) // 2
    top = (h - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size
    image = image.crop((left, top, right, bottom))  # PIL Image, size (224, 224)

    # Step 3: to_numpy
    image = np.array(image).astype(np.float32) / 255.0  # (H, W, C)

    # Step 4: norm
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std  # (H, W, C)
    image = image.transpose(2, 0, 1) # (C, H, W)
    return image[None] # (1, C, H, W)

# ************* ToDo, resize short side *************
def resize_short_side(img_path, target_size=224, patch_size=14):
    # 1. 读取图片
    image = Image.open(img_path).convert("RGB")
    w, h = image.size

    # 2. 计算缩放比例：让短边 = target_size
    if w < h:
        scale = target_size / w
        new_w = target_size
        new_h = int(round(h * scale))
    else:
        scale = target_size / h
        new_h = target_size
        new_w = int(round(w * scale))
    
    # 3. 执行 Resize (使用 BICUBIC 插值)
    image = image.resize((new_w, new_h), Image.BICUBIC)

    # 4. 确保长边是 14 的倍数 (直接切掉多余的尾巴)
    # 比如 300 -> 变成 (300 // 14) * 14 = 294
    valid_w = (new_w // patch_size) * patch_size
    valid_h = (new_h // patch_size) * patch_size
    image = image.crop((0, 0, valid_w, valid_h))

    # 5. 转 Numpy + 归一化 (直接复用 center_crop 里的逻辑)
    image = np.array(image).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    image = image.transpose(2, 0, 1)

    return image[None] # 增加 Batch 维度 (1, C, H, W)