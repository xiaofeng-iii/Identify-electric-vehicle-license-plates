# 电动自行车车牌识别系统

基于传统计算机视觉算法的电动自行车车牌识别系统，**不使用任何深度学习或机器学习模型**。

## 功能特性

- **多车牌识别**：支持同一图像中识别多个车牌
- **倾斜校正**：自动检测并校正倾斜的车牌
- **抗干扰**：通过长宽比过滤排除广告文字等干扰
- **自适应检测**：梯度递增亮度检测，适应不同光照条件
- **并行处理**：支持多进程并行处理，提升批量识别效率

## 环境要求

- Python 3.8+
- opencv-python
- numpy
- matplotlib
- Pillow

## 安装

```bash
pip install opencv-python numpy matplotlib pillow
```

## 使用方法

### 快速开始

1. 将待识别图片放入 `resources/raw_images/` 目录
2. 运行识别程序：

```bash
python src/main.py
```

### 命令行参数

```bash
# 简洁模式（默认）- 并行处理，只显示结果
python src/main.py

# 详细模式 - 显示处理过程
python src/main.py --verbose

# 调试模式 - 保存中间处理图片
python src/main.py --debug

# 串行模式 - 禁用并行处理
python src/main.py --serial

# 组合使用
python src/main.py --debug --verbose
```

### 测试脚本

```bash
# 测试单张图片
python src/test_pipeline.py path/to/image.jpg

# 测试整个目录
python src/test_pipeline.py path/to/directory/

# 显示可视化窗口
python src/test_pipeline.py path/to/image.jpg --show
```

## 项目结构

```
├── src/
│   ├── main.py              # 主程序入口
│   ├── test_pipeline.py     # 测试脚本
│   ├── config.py            # 配置参数
│   ├── image_process.py     # 图像预处理与车牌定位
│   ├── char_segment.py      # 字符分割
│   ├── match_ocr.py         # 模板匹配识别
│   └── visualize_result.py  # 结果可视化
├── resources/
│   ├── templates/           # 字符模板库（20x40像素）
│   └── raw_images/          # 待识别图片
└── output/
    ├── results/             # 识别结果图片
    └── debug_steps/         # 调试中间图片
```

## 处理流程

```
原始图片 → 预处理 → 车牌定位 → 畸变校正 → 字符分割 → 模板匹配 → 输出结果
```

### 1. 图像预处理
- 灰度化
- 高斯滤波去噪
- CLAHE 直方图均衡化增强对比度

### 2. 车牌定位
- **自适应白色检测**：基于 HSV 颜色空间，使用 `whiteness = V - S` 作为白色程度指标
- **梯度递增检测**：从低亮度百分比开始，逐步递增，适应不同亮度的车牌
- **区域排除机制**：检测到车牌后排除该区域，继续检测其他车牌
- **几何筛选**：长宽比（1.7-2.3）、角度（±35°）、面积过滤

### 3. 畸变校正
- `cv2.minAreaRect` 检测旋转角度
- 仿射变换校正平面旋转
- 透视变换处理梯形畸变

### 4. 字符分割
- 自适应二值化
- 形态学处理（开运算去噪 + 闭运算连接笔画）
- 连通域分析提取字符轮廓
- 高度比例、宽高比、面积筛选

### 5. 字符识别
- **模板匹配**：使用 Jaccard 相似度（SSD 方法）
- **置信度阈值**：低于阈值的字符标记为不确定
- **结果过滤**：无有效字符的车牌不显示

## 配置参数

主要参数位于 `src/config.py`：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `ADAPTIVE_WHITE_TOP_PERCENT` | 1 | 初始白色检测百分比 |
| `ADAPTIVE_WHITE_STEP` | 2 | 检测递增步长 |
| `ADAPTIVE_WHITE_MAX_PERCENT` | 41 | 最大检测百分比 |
| `PLATE_ASPECT_RATIO_MIN/MAX` | 1.7/2.3 | 车牌长宽比范围 |
| `PLATE_ANGLE_MAX` | 35 | 最大允许倾斜角度 |
| `OCR_MIN_CONFIDENCE` | 0.62 | 字符识别最小置信度 |
| `PARALLEL_WORKERS` | 10 | 并行处理进程数 |

## 模板库

模板位于 `resources/templates/`，包含：
- 数字：0-9
- 字母：A-Z（不含 I、O）
- 汉字：广、佛、山、州、惠 等常见地区字

模板规格：20×40 像素，二值图像

## 输出说明

- **结果图片**：`output/results/{图片名}_result.jpg`
  - 原图标注车牌位置（绿色边框）
  - 识别结果文字（白色，半透明背景）

- **调试图片**（--debug 模式）：`output/debug_steps/{图片名}/`
  - 预处理各阶段
  - 白色检测掩码
  - 候选区域筛选
  - 字符分割结果
  - 模板匹配过程

## 鲁棒性设计

1. **角度过滤**：限制最大偏离角度（35°），排除背景误检
2. **梯度递增检测**：适应不同亮度车牌共存的场景
3. **置信度阈值**：不确定的字符不输出，避免误识别
4. **无效车牌过滤**：没有有效字符的候选区域不显示
5. **可调参数**：用户可根据实际场景调整配置

## 扩展支持

当前针对白色车牌优化。如需支持其他颜色车牌（蓝色、黄色、绿色），可在 `_adaptive_white_segmentation` 方法中添加 H 通道（色相）过滤条件。

## 许可证

MIT License
