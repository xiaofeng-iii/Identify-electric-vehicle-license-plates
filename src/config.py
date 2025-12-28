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
RESULT_DIR = os.path.join(OUTPUT_DIR, 'results')  # 最终识别结果目录

# ==================== 预处理参数 ====================
# 高斯滤波核大小 (必须为奇数)
GAUSSIAN_KERNEL_SIZE = (5, 5)
# 高斯滤波标准差 (0表示自动计算)
GAUSSIAN_SIGMA = 0

# CLAHE参数 (对比度受限自适应直方图均衡化)
CLAHE_CLIP_LIMIT = 2.0  # 对比度限制阈值
CLAHE_TILE_SIZE = (8, 8)  # 分块大小

# ==================== 车牌定位参数 ====================

# ==================== HSV颜色阈值 ====================
# 自适应白色检测参数
ADAPTIVE_WHITE_TOP_PERCENT = 1      # 选取最亮的百分比（初始值）
ADAPTIVE_WHITE_STEP = 2              # 未检出车牌时，每次增加的百分比
ADAPTIVE_WHITE_MAX_PERCENT = 41      # 最大百分比上限，防止无限增加

ADAPTIVE_WHITE_MAX_SATURATION = 20   # 最大饱和度，排除彩色区域

# ==================== 轮廓筛选参数 ====================
# 矩形度阈值 (轮廓面积/外接矩形面积)，车牌应该接近矩形
PLATE_RECTANGULARITY_MIN = 0.5

# 车牌长宽比范围 (用于过滤非车牌区域)
PLATE_ASPECT_RATIO_MIN = 1.7
PLATE_ASPECT_RATIO_MAX = 2.3

# 车牌角度偏离阈值 (度数)
# 角度为0表示完全水平，该值表示允许的最大偏离角度
PLATE_ANGLE_MAX = 35  # 最大允许偏离角度（度）

# 车牌面积范围 (相对于图像面积的比例)
PLATE_AREA_MIN_RATIO = 0.01   # 最小面积
PLATE_AREA_MAX_RATIO = 0.20    # 降低最大面积，排除过大区域

# 标准车牌尺寸（用于统一分辨率，便于后续处理）
PLATE_STANDARD_WIDTH = 800    # 标准宽度
PLATE_STANDARD_HEIGHT = 400   # 标准高度（2:1 比例）

# ==================== 字符分割参数 ====================
# 标准字符模板大小
TEMPLATE_WIDTH = 20
TEMPLATE_HEIGHT = 40

# 二值化参数
CHAR_BINARY_BLOCK_SIZE = 55       # 自适应二值化块大小（必须为奇数）
CHAR_BINARY_C = 7                 # 自适应二值化常数（从均值中减去的值）
CHAR_BINARY_INVERT = True         # 是否反转（True=白底黑字变黑底白字）

# 形态学参数
CHAR_MORPH_CLOSE_KERNEL = (11, 11)  # 闭运算核大小（连接断裂笔画）
CHAR_MORPH_OPEN_KERNEL = (7, 7)   # 开运算核大小（去除噪点）

# 字符筛选参数
CHAR_HEIGHT_RATIO_MIN = 0.2       # 字符高度占车牌高度的最小比例
CHAR_HEIGHT_RATIO_MAX = 0.95      # 字符高度占车牌高度的最大比例
CHAR_ASPECT_RATIO_MIN = 0.1       # 字符宽高比最小值
CHAR_ASPECT_RATIO_MAX = 1.5       # 字符宽高比最大值
CHAR_MIN_AREA = 30                # 字符最小面积（像素）
CHAR_EDGE_MARGIN = 2              # 字符距边缘最小距离（像素）

# 字符归一化缩放比例（控制字符在20x40画布中的大小）
# 0.8 = 字符缩放到16x32左右，边距较大
# 0.9 = 字符缩放到18x36左右，边距较小（推荐）
# 1.0 = 字符填满画布，无边距（不推荐，可能截断字符边缘）
CHAR_NORMALIZE_SCALE = 1

# ==================== 字符识别参数 ====================
# 最小置信度阈值，低于此值认为无法识别
OCR_MIN_CONFIDENCE = 0.62

# ==================== 调试开关 ====================
DEBUG_MODE = False  # 默认关闭，由 main.py 根据 --debug 参数动态控制

# ==================== 并行处理参数 ====================
PARALLEL_WORKERS = 10  # 并行处理图片的进程数
