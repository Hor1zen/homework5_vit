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

## 项目结构
- `demo_data/`：示例数据文件夹，包含猫狗图片和特征文件，用于模型调试。
- `mysite/`：Django项目主配置文件夹，包括settings.py等。
- `search/`：Django应用文件夹，包含视图、模型、模板，用于Web检索界面。
- `static/js/`：静态JavaScript文件，用于前端交互。
- `.gitignore`：Git忽略文件，屏蔽临时和大数据文件。
- `README.md`：项目说明文档。
- `check_images.py`：检查下载图片有效性的脚本。
- `debug.py`：调试脚本，验证模型输出与参考特征的相似度。
- `dinov2_numpy.py`：DINOv2模型的NumPy实现核心文件。
- `download_images.py`：从CSV文件下载图片的爬虫脚本。
- `manage.py`：Django管理脚本，用于运行服务器和迁移。
- `preprocess_image.py`：图像预处理函数，如resize和归一化。
- `run_retrieval.py`：构建图像特征索引的脚本。

## 许可证
MIT License