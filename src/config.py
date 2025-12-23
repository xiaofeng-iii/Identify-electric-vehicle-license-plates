# -*- coding: utf-8 -*-
"""
配置文件：存放系统参数和阈值
"""

import os

# ==================== 路径配置 ====================
# 获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 资源目录
RESOURCES_DIR = os.path.join(PROJECT_ROOT, 'resources')
TEMPLATES_DIR = os.path.join(RESOURCES_DIR, 'templates')
RAW_IMAGES_DIR = os.path.join(RESOURCES_DIR, 'raw_images')

# 输出目录
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')
DEBUG_DIR = os.path.join(OUTPUT_DIR, 'debug_steps')

# ==================== 预处理参数 ====================
# 高斯滤波核大小 (必须为奇数)
GAUSSIAN_KERNEL_SIZE = (5, 5)
# 高斯滤波标准差 (0表示自动计算)
GAUSSIAN_SIGMA = 0

# CLAHE参数 (对比度受限自适应直方图均衡化)
CLAHE_CLIP_LIMIT = 2.0  # 对比度限制阈值
CLAHE_TILE_SIZE = (8, 8)  # 分块大小

# ==================== 车牌定位参数 ====================
# 车牌长宽比范围 (用于过滤非车牌区域)
PLATE_ASPECT_RATIO_MIN = 2.0
PLATE_ASPECT_RATIO_MAX = 4.0

# 车牌面积范围 (相对于图像面积的比例)
PLATE_AREA_MIN_RATIO = 0.005
PLATE_AREA_MAX_RATIO = 0.15

# ==================== HSV颜色阈值 ====================
# 白色车牌 (电动车常见)
WHITE_LOWER = (0, 0, 180)
WHITE_UPPER = (180, 30, 255)

# 蓝色车牌
BLUE_LOWER = (100, 80, 80)
BLUE_UPPER = (130, 255, 255)

# 黄色车牌
YELLOW_LOWER = (15, 80, 80)
YELLOW_UPPER = (35, 255, 255)

# 绿色车牌 (新能源)
GREEN_LOWER = (35, 80, 80)
GREEN_UPPER = (85, 255, 255)

# ==================== 形态学操作参数 ====================
# 闭运算核大小
MORPH_CLOSE_KERNEL = (25, 10)

# ==================== 字符分割参数 ====================
# 标准字符模板大小
TEMPLATE_WIDTH = 20
TEMPLATE_HEIGHT = 40

# ==================== 调试开关 ====================
DEBUG_MODE = True  # 是否保存中间处理图片
