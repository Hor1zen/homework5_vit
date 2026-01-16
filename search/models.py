from django.db import models
from django.contrib.auth.models import User

# Create your models here.

class SearchRecord(models.Model):
    """
    记录用户的搜索历史
    存储策略：
    1. 图片文件存磁盘 (Media)，数据库只存路径。
    2. 搜索结果存 JSON，避免创建大量关联表，读取速度最快。
    """
    # 关联到 Django 自带的 User 表
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='history')
    
    # 保存上传的查询图片，按年月日归档，避免文件夹爆炸
    query_image = models.ImageField(upload_to='queries/%Y/%m/%d/')
    
    # 记录精简后的结果 (JSON格式)，只存必要的 Top-10 路径和分数
    # 格式示例: [{"path": "/static/...", "score": 0.98}, ...]
    results_data = models.JSONField(default=list)
    
    # 搜索耗时 (毫秒)，用于后续性能分析
    latency_ms = models.FloatField(default=0.0)
    
    # 创建时间，加索引 (db_index=True) 以便快速查询“最近搜索”
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        ordering = ['-created_at'] # 默认按时间倒序展示

    def __str__(self):
        return f"{self.user.username} @ {self.created_at.strftime('%Y-%m-%d %H:%M')}"
