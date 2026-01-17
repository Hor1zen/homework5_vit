# DINOv2 Image Retrieval System

这个项目是一个基于DINOv2 Vision Transformer的图像检索Web应用。它使用NumPy实现模型，支持上传图片查询相似图像，并进行猫狗分类。

## 要求
- Python 3.11.9
- 依赖：见requirements.txt
- 硬件：CPU环境，索引构建可能需要几小时
- 模型参数：vit-dinov2-base.npz不在仓库，你需要自行下载或训练DINOv2模型获取。

## 安装
1. 克隆仓库：`git clone https://github.com/Hor1zen/homework5_vit.git`
2. 安装依赖：`pip install -r requirements.txt`
3. 下载模型参数：将vit-dinov2-base.npz放到项目根目录。
4. 生成迁移：`python manage.py makemigrations`
5. 执行迁移：`python manage.py migrate`

## 准备数据
准备10,000+张图像，放到`downloaded_images/`文件夹。可以从data.csv使用`download_images.py`脚本下载图片（配合网络代理加速）。下载后，运行`python check_images.py`检查有效图片数量，确保数据集完整。

## 构建索引
运行`python run_retrieval.py`生成gallery_features.npy和gallery_paths.npy。CPU下耗时较长，可考虑GPU优化。

## 使用
1. 启动服务器：`python manage.py runserver`
2. 浏览器访问，注册或登录账户。
3. 上传图片查询Top-10相似结果，系统会显示分类结果（猫狗或其他）。

## 调试
运行`python debug.py`，它会使用demo_data中的cat.jpg、dog.jpg和cat_dog_feature.npy验证模型输出，显示L2误差和余弦相似度，帮助检查训练或实现的准确性。

## 许可证
MIT License