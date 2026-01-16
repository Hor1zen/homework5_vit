import os
import csv
import requests
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import time

# ================= 配置区域 =================
CSV_PATH = 'data.csv'      # CSV 文件路径
SAVE_DIR = 'gallery'       # 图片保存文件夹
MAX_WORKERS = 16           # 并发线程数 (根据网速调整，推荐 8-16)
TIMEOUT = 5                # 单张图片下载超时时间 (秒)
# ===========================================

def download_one_image(args):
    """
    下载单张图片的具体的任务函数
    """
    idx, url, save_fold = args
    
    # 1. 构造保存文件名，例如 00123.jpg
    # 简单的扩展名判断
    ext = ".jpg"
    if ".png" in url.lower(): ext = ".png"
    elif ".jpeg" in url.lower(): ext = ".jpg"
    
    filename = os.path.join(save_fold, f"{idx:05d}{ext}")
    
    # 如果文件已经存在，跳过 (支持断点续传)
    if os.path.exists(filename):
        return f"Skipped {idx}"

    try:
        # 2. 发起请求
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }
        response = requests.get(url, headers=headers, timeout=TIMEOUT)
        
        # 3. 检查状态码
        if response.status_code == 200:
            with open(filename, 'wb') as f:
                f.write(response.content)
            return "Success"
        else:
            return "Status Error"
    except Exception as e:
        return "Network Error"

def main():
    # 1. 创建保存目录
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        print(f"📁 Created directory: {SAVE_DIR}")

    # 2. 读取 CSV 文件
    tasks = []
    print(f"📖 Reading {CSV_PATH}...")
    
    try:
        with open(CSV_PATH, 'r', encoding='utf-8') as f:
            # 尝试自动推断分隔符 (逗号或制表符)
            line = f.readline()
            file_dialet = csv.Sniffer().sniff(line)
            f.seek(0) # 回到文件头
            
            reader = csv.DictReader(f, dialect=file_dialet)
            
            # 使用 enumerate 生成自增 ID
            for i, row in enumerate(reader):
                if 'image_url' in row:
                    url = row['image_url']
                    # 将任务打包: (ID, URL, 保存路径)
                    tasks.append((i, url, SAVE_DIR))
    except Exception as e:
        print(f"❌ 读取 CSV 失败: {e}")
        print("💡 提示: 请检查 data.csv 是否存在，以及列名是否包含 'image_url'")
        return

    total = len(tasks)
    print(f"🚀 Found {total} images. Starting download with {MAX_WORKERS} threads...")

    # 3. 多线程下载
    success_count = 0
    fail_count = 0
    
    t0 = time.time()
    
    # 使用 ThreadPoolExecutor 进行并行处理
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # map 会按顺序启动任务，结果也是按顺序返回
        results = executor.map(download_one_image, tasks)
        
        # 简单的进度显示
        for i, res in enumerate(results):
            if res == "Success" or "Skipped" in res:
                success_count += 1
            else:
                fail_count += 1
            
            # 每 100 张打印一次进度
            if (i + 1) % 100 == 0:
                print(f"[{i+1}/{total}] Success: {success_count}, Failed: {fail_count}")

    print("="*40)
    print(f"🎉 Done! Time cost: {time.time() - t0:.2f}s")
    print(f"✅ Downloaded: {success_count}")
    print(f"❌ Failed: {fail_count} (Links might be expired)")
    print(f"📁 Images saved to: {os.path.abspath(SAVE_DIR)}")

if __name__ == "__main__":
    main()
