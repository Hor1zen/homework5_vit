import os
from PIL import Image

GALLERY_DIR = "./downloaded_images"

def count_and_verify_images():
    print(f"📊 开始扫描 {GALLERY_DIR} 目录...")
    
    if not os.path.exists(GALLERY_DIR):
        print(f"❌ 错误: 文件夹不存在 {GALLERY_DIR}")
        return

    # 统计变量
    total_files = 0
    valid_images = 0
    corrupt_images = 0
    non_image_files = 0
    
    # 只需要扫描常见的图片格式
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.gif')
    
    # 获取文件列表
    file_list = os.listdir(GALLERY_DIR)
    total_files = len(file_list)
    
    print(f"📂 总文件数: {total_files}")
    print("-" * 40)

    for i, filename in enumerate(file_list):
        file_path = os.path.join(GALLERY_DIR, filename)
        
        # 1. 检查扩展名
        if not filename.lower().endswith(exts):
            non_image_files += 1
            continue

        # 2. 尝试打开并验证
        try:
            with Image.open(file_path) as img:
                # verify() 会检查文件头，能够快速识别是否是有效的图片文件
                # 注意：verify() 不会加载图像数据，速度很快
                img.verify()
                valid_images += 1
                
        except Exception:
            corrupt_images += 1
            # 可以选择打印损坏的文件名，但我这里先只计数
            # print(f"❌ 损坏: {filename}")
        
        # 简单的进度条
        if (i + 1) % 1000 == 0:
            print(f"   已扫描 {i + 1} / {total_files} 文件...")

    print("=" * 40)
    print("📋 最终统计结果")
    print("=" * 40)
    print(f"✅ 有效图片: {valid_images}")
    print(f"❌ 损坏图片: {corrupt_images} (例如: 0KB文件, 404网页等)")
    print(f"📄 非图文件: {non_image_files} (txt, json等)")
    print("-" * 40)
    print(f"🔢 总计文件: {total_files}")
    
    if valid_images == 0:
        print("\n⚠️ 警告: 没有一张有效图片！请检查爬虫或网络。")
    elif valid_images < 100:
        print("\n⚠️ 警告: 有效图片太少，可能会影响 Top-10 检索效果。")
    else:
        print("\n🎉 数据量充足，可以运行检索程序！")

if __name__ == "__main__":
    count_and_verify_images()
