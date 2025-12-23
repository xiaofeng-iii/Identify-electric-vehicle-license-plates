# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

电动自行车车牌识别系统，**仅使用传统计算机视觉算法**（禁止使用任何深度学习或机器学习模型）。识别结果以文本形式展示。

## 开发环境

- **Python**: 3.8+
- **核心库**: opencv-python, numpy, matplotlib

```bash
# 安装依赖
pip install opencv-python numpy matplotlib

# 运行
python src/main.py
```

## 功能要求

### 必做功能（全部实现，每类至少5张测试图片）

1. **正常车牌识别** - 车牌无明显畸变
2. **倾斜车牌识别** - 拍摄角度导致车牌倾斜或畸变
3. **文字干扰识别** - 车牌周围存在广告或其他文字干扰

### 选做功能（三选一，至少5张测试图片）

1. **多车牌识别** - 同时识别图片中所有车牌（数量不固定）
2. **极端天气识别** - 雨天、雾天、雪天或光线明显变化
3. **遮挡检测** - 检测车牌被遮挡，输出"车牌被遮挡"结论

## 系统架构

五阶段流水线处理：

### 1. 预处理 (`image_process.py`)
- 灰度化 (BGR → GRAY)
- 高斯滤波去噪
- 直方图均衡化 (CLAHE) 增强对比度

### 2. 车牌定位 (`image_process.py`)
两种方案：
- **颜色定位 (HSV)**: `cv2.inRange` 提取白/蓝/黄色车牌区域
- **边缘定位 (Sobel)**: 垂直边缘提取 + Otsu二值化

关键过滤逻辑：
- 形态学闭运算融合区域
- **长宽比过滤 (2:1–4:1)** - 排除广告文字干扰的关键
- `cv2.findContours` 轮廓面积筛选

### 3. 畸变校正 (`image_process.py`)
- `cv2.minAreaRect` 检测旋转角度
- 仿射变换 (`cv2.warpAffine`) 处理平面旋转
- 透视变换 (`cv2.warpPerspective`) 处理梯形畸变

### 4. 字符分割 (`char_segment.py`)
- 垂直投影法：统计每列白色像素数量
- 波峰波谷分析定位字符边界
- 边界清洗去除边框/铆钉

### 5. 字符识别 (`match_ocr.py`)
- 模板匹配（禁止AI）
- 标准字符库：0-9, A-Z, 常见汉字（20x40像素二值图）
- `cv2.matchTemplate` 或像素差值平方和 (SSD)
- 选取相似度最高的匹配结果

## 文件结构

```
src/
├── main.py           # 主入口，串联整个流程
├── image_process.py  # 预处理、定位、校正
├── char_segment.py   # 垂直投影分割
├── match_ocr.py      # 模板匹配识别
└── config.py         # 颜色阈值、长宽比参数

resources/
├── templates/        # 字符模板库
└── raw_images/       # 测试图片（按类别分文件夹）

output/
└── debug_steps/      # 中间处理过程图片
```

## 关键约束

1. **禁止AI模型** - 只能使用模板匹配，不能用神经网络
2. **单机运行** - 程序必须能独立运行
3. **代码规范** - 逻辑清晰，必要注释，可读性强
4. **文字干扰** - 通过长宽比过滤 (2:1–4:1) 排除
5. **倾斜处理** - 使用仿射/透视变换校正
