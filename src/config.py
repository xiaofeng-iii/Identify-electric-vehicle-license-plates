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
# # 边缘密度阈值：车牌内部应有足够的字符边缘
# PLATE_EDGE_DENSITY_MIN = 0.06

# ==================== HSV颜色阈值 ====================
# 自适应白色检测参数
ADAPTIVE_WHITE_TOP_PERCENT = 10      # 选取最亮的百分比（初始值）
ADAPTIVE_WHITE_STEP = 2              # 未检出车牌时，每次增加的百分比
ADAPTIVE_WHITE_MAX_PERCENT = 50      # 最大百分比上限，防止无限增加

ADAPTIVE_WHITE_MAX_SATURATION = 10   # 最大饱和度，排除彩色区域

# 白色车牌 (电动车常见) - 固定阈值备用
# 注意：S值不能太低，否则会把灰色地面误识别为白色
WHITE_LOWER = (0, 0, 170)
WHITE_UPPER = (180, 50, 255)

# # 蓝色车牌
# BLUE_LOWER = (100, 80, 80)
# BLUE_UPPER = (130, 255, 255)

# # 黄色车牌
# YELLOW_LOWER = (15, 80, 80)
# YELLOW_UPPER = (35, 255, 255)

# # 绿色车牌 (新能源)
# GREEN_LOWER = (35, 80, 80)
# GREEN_UPPER = (85, 255, 255)

# ==================== 形态学操作参数 ====================
# 颜色定位闭运算核大小 - 适中大小，连接字符但不连接相邻物体
MORPH_CLOSE_KERNEL_COLOR = (17, 5)
# 边缘定位闭运算核大小 - 较小以保持边缘细节
MORPH_CLOSE_KERNEL_EDGE = (15, 5)

# ==================== 轮廓筛选参数 ====================
# 矩形度阈值 (轮廓面积/外接矩形面积)，车牌应该接近矩形
PLATE_RECTANGULARITY_MIN = 0.5

# 车牌长宽比范围 (用于过滤非车牌区域)
PLATE_ASPECT_RATIO_MIN = 1.5
PLATE_ASPECT_RATIO_MAX = 3.0

# 车牌角度偏离阈值 (度数)
# 角度为0表示完全水平，该值表示允许的最大偏离角度
PLATE_ANGLE_MAX = 5  # 最大允许偏离角度（度）

# 车牌面积范围 (相对于图像面积的比例)
PLATE_AREA_MIN_RATIO = 0.015   # 最小面积
PLATE_AREA_MAX_RATIO = 0.20    # 降低最大面积，排除过大区域

# ==================== 字符分割参数 ====================
# 标准字符模板大小
TEMPLATE_WIDTH = 20
TEMPLATE_HEIGHT = 40

# ==================== 调试开关 ====================
DEBUG_MODE = True  # 是否保存中间处理图片
