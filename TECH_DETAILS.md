# 电动自行车车牌识别系统 - 技术详情

> 本文档为 Report 写作提供技术参考，重点介绍车牌定位、字符分割、字符识别三个核心模块的算法原理与鲁棒性设计。

---

## 一、系统概述

本系统采用**纯传统计算机视觉方法**实现电动自行车车牌识别，不依赖任何深度学习或机器学习模型。系统采用五阶段流水线架构：

```
图像预处理 → 车牌定位 → 畸变校正 → 字符分割 → 模板匹配识别
```

**技术栈**：Python + OpenCV + NumPy

**核心特点**：
- 纯 OpenCV 实现，无需 GPU
- 支持多车牌检测
- 自适应光照变化
- 多进程并行处理

---

## 二、图像预处理

预处理阶段对原始图像进行标准化处理，提高后续车牌定位的准确性。

### 2.1 灰度化

**实现代码**（`image_process.py:57-61`）：

```python
def to_grayscale(self, image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray
```

将 BGR 彩色图像转换为灰度图，减少数据量，简化后续处理。

### 2.2 高斯滤波去噪

**实现代码**（`image_process.py:63-67`）：

```python
def gaussian_blur(self, image):
    blurred = cv2.GaussianBlur(image, GAUSSIAN_KERNEL_SIZE, GAUSSIAN_SIGMA)
    return blurred
```

**参数**：
- `GAUSSIAN_KERNEL_SIZE = (5, 5)`：滤波核大小
- `GAUSSIAN_SIGMA = 0`：标准差（0 表示自动计算）

**作用**：
- 去除图像噪声（如传感器噪声、压缩噪声）
- 平滑图像，减少后续处理的干扰

### 2.3 直方图均衡化（CLAHE）

**实现代码**（`image_process.py:69-79`）：

```python
def histogram_equalization(self, image):
    # CLAHE: 对比度受限自适应直方图均衡化
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,      # 2.0
        tileGridSize=CLAHE_TILE_SIZE     # (8, 8)
    )
    equalized = clahe.apply(image)
    return equalized
```

**CLAHE 原理**：
- **分块处理**：将图像分成 8×8 的小块，每块独立均衡化
- **对比度限制**：限制直方图的最大高度（clipLimit=2.0），避免过度增强噪声
- **自适应**：不同区域使用不同的均衡化参数，适应局部光照变化

**相比普通直方图均衡化的优势**：
- 避免过度增强噪声
- 保留局部细节
- 适应光照不均的场景（如一侧亮一侧暗）

### 2.4 预处理流程

**实现代码**（`image_process.py:81-92`）：

```python
def preprocess(self, image):
    # 1. 灰度化
    gray = self.to_grayscale(image)

    # 2. 高斯滤波去噪
    blurred = self.gaussian_blur(gray)

    # 3. 直方图均衡化增强对比度
    enhanced = self.histogram_equalization(blurred)

    return enhanced
```

**处理顺序**：灰度化 → 高斯滤波 → CLAHE

**鲁棒性体现**：
- 标准化图像质量，减少光照、噪声对后续处理的影响
- CLAHE 自适应增强，适应不同光照条件

---

## 三、车牌定位（核心创新点）

### 3.1 基于 HSV 颜色空间的自适应白色检测

电动自行车车牌以白底黑字为主。传统固定阈值方法难以适应不同光照条件，本系统采用**自适应白色检测算法**。

#### 3.1.1 白色程度指标

在 HSV 颜色空间中，白色的特征是：
- **高亮度 (V)**：接近 255
- **低饱和度 (S)**：接近 0

本系统定义**白色程度指标**：

```python
whiteness = V - S
```

**实现代码**（`image_process.py:163-177`）：
```python
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

# 计算白色程度：whiteness = V - S
whiteness = v.astype(np.int16) - s.astype(np.int16)

# 计算白色程度阈值：取 top_percent 对应的百分位数
whiteness_threshold = np.percentile(whiteness, 100 - top_percent)

# 白色条件
white_mask = whiteness >= whiteness_threshold
mask = (white_mask * 255).astype(np.uint8)
```

该指标综合考虑亮度和饱和度：
- V 越大、S 越小 → whiteness 越大 → 越接近白色
- 相比单独使用 V 通道，能更好地排除高亮度但有颜色的区域（如黄色、蓝色区域）

#### 3.1.2 梯度递增 + 区域排除式检测算法

不同车牌在同一图像中可能具有不同的亮度（如逆光、阴影）。为解决多车牌检测问题，系统采用**梯度递增 + 区域排除式检测**：

**实现流程**（`image_process.py:326-374`）：

```python
# 初始化排除掩码（全白表示所有区域可检测）
exclusion_mask = np.ones((img_h, img_w), dtype=np.uint8) * 255

current_percent = ADAPTIVE_WHITE_TOP_PERCENT  # 1%
all_candidates = []

while current_percent <= ADAPTIVE_WHITE_MAX_PERCENT:  # 最大 41%
    # 使用当前百分比进行白色分割
    color_mask = self._adaptive_white_segmentation(image, top_percent=current_percent)

    # 应用排除掩码（已检测区域变黑，不会产生轮廓）
    color_mask = cv2.bitwise_and(color_mask, exclusion_mask)

    # 在掩码中寻找车牌候选区域
    color_contours = self.find_contours(color_mask)
    color_candidates = self.filter_candidates(color_contours, img_shape)

    if len(color_candidates) > 0:
        all_candidates.extend(color_candidates)

        # 将检测到的区域从排除掩码中移除（涂黑）
        for rect, box, score in color_candidates:
            expanded_box = self._expand_box(box, scale=1.2, img_shape=img_shape)
            cv2.fillPoly(exclusion_mask, [expanded_box], 0)

    current_percent += ADAPTIVE_WHITE_STEP  # 递增 2%
```

**鲁棒性体现**：
- 从最亮区域开始检测（1%），优先找到最明显的车牌
- 逐步降低亮度阈值（每次+2%），检测较暗的车牌
- **区域排除机制**：检测到车牌后，在排除掩码中涂黑该区域（扩大 1.2 倍），避免重复检测
- 适应同一图像中**不同亮度车牌共存**的场景
- 最大检测到 41% 百分位，覆盖大部分光照场景

### 3.2 几何特征筛选

车牌具有明确的几何特征，通过几何筛选可有效排除误检区域。

#### 3.2.1 长宽比过滤

**实现代码**（`image_process.py:285-291`）：

```python
aspect_ratio = width / height

# 长宽比过滤 (关键：排除广告文字等干扰)
if aspect_ratio < PLATE_ASPECT_RATIO_MIN or aspect_ratio > PLATE_ASPECT_RATIO_MAX:
    continue
```

电动自行车车牌实际比例约为 1.56（140mm × 90mm），考虑到拍摄角度和透视畸变，设置允许范围：

```
1.7 ≤ 长宽比 ≤ 2.3
```

**鲁棒性体现**：排除广告文字、标志等干扰物（通常长宽比不在此范围）

#### 3.2.2 角度过滤

**实现代码**（`image_process.py:245-263`）：

```python
# 计算最小外接矩形的长边方向
edge1 = np.linalg.norm(box[0] - box[1])
edge2 = np.linalg.norm(box[1] - box[2])

if edge1 >= edge2:
    long_edge_vec = box[1] - box[0]
    width, height = edge1, edge2
else:
    long_edge_vec = box[2] - box[1]
    width, height = edge2, edge1

# 计算长边与水平方向的夹角
angle = np.degrees(np.arctan2(long_edge_vec[1], long_edge_vec[0]))

# 角度偏离过滤
angle_deviation = abs(angle)
if angle_deviation > PLATE_ANGLE_MAX:  # 35°
    continue
```

正常场景下车牌基本水平放置，设置最大允许偏离角度：

```
|角度| ≤ 35°
```

**鲁棒性体现**：
- 排除背景中非水平放置的矩形区域
- 允许一定倾斜，适应手持拍摄的抖动

#### 3.2.3 矩形度过滤

**实现代码**（`image_process.py:272-282`）：

```python
rectangularity = contour_area / bounding_box_area

if rectangularity < PLATE_RECTANGULARITY_MIN:  # 0.5
    continue
```

车牌应该是规整的矩形，矩形度过低说明轮廓不规则。

#### 3.2.4 面积过滤

```python
img_area * PLATE_AREA_MIN_RATIO ≤ 车牌面积 ≤ img_area * PLATE_AREA_MAX_RATIO
# 1% ≤ 车牌面积 ≤ 20%
```

排除过小（噪点）和过大（整个背景）的区域。

### 3.3 候选区域合并（IoU 去重）

**实现代码**（`image_process.py:425-444`）：

```python
def _merge_overlapping(self, candidates, iou_threshold=0.3):
    # 按分数排序
    sorted_candidates = sorted(candidates, key=lambda x: x[2], reverse=True)
    merged = []

    while sorted_candidates:
        best = sorted_candidates.pop(0)
        merged.append(best)

        remaining = []
        for candidate in sorted_candidates:
            # IoU < 0.3 认为是不同车牌，保留
            if self._calculate_iou(best[1], candidate[1]) < iou_threshold:
                remaining.append(candidate)
        sorted_candidates = remaining

    return merged
```

**IoU (Intersection over Union)** 用于判断两个矩形是否为同一车牌：
- IoU > 0.3：重叠度高，认为是同一车牌，保留评分最高的
- IoU ≤ 0.3：重叠度低，认为是不同车牌，都保留

**鲁棒性体现**：避免同一车牌被重复检测多次

### 3.4 畸变校正

**实现代码**（`image_process.py:491-537`）：

```python
# 计算旋转角度
angle = np.degrees(np.arctan2(long_edge_vec[1], long_edge_vec[0]))

# 旋转图像使车牌水平
rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated = cv2.warpAffine(image, rotation_matrix, (img_width, img_height))

# 裁剪车牌区域
plate_region = rotated[y1:y2, x1:x2]

# 统一缩放到标准分辨率 (800×400)
plate_region = cv2.resize(plate_region, (PLATE_STANDARD_WIDTH, PLATE_STANDARD_HEIGHT))
```

**处理流程**：
1. 计算车牌旋转角度
2. 仿射变换校正倾斜
3. 裁剪车牌区域
4. **统一缩放到 800×400**（使后续参数稳定）

---

## 四、字符分割

### 4.1 预处理

**实现代码**（`char_segment.py:29-50`）：

```python
# 1. 灰度化
if len(plate_image.shape) == 3:
    gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)

# 2. 自适应二值化（反转）
binary = cv2.adaptiveThreshold(
    gray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,  # 白底黑字 → 黑底白字
    blockSize=55,
    C=7
)
```

- **自适应阈值**：处理车牌光照不均（如一侧亮一侧暗）
- **反转**：使字符为白色（前景），便于轮廓检测

### 4.2 形态学处理

**实现代码**（`char_segment.py:126-130`）：

```python
# 先开运算：去除噪点
open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel)

# 再闭运算：连接断裂笔画
close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)
```

**处理顺序的考量**：
- **先开运算**：去除孤立噪点，避免闭运算时将噪点连接到字符
- **再闭运算**：填补字符内部孔洞，连接断裂笔画（如数字 8 的两个圆）

**参数说明**：
- 开运算核 (7, 7)：去除 7×7 以下的噪点
- 闭运算核 (7, 7)：连接 7 像素内的断裂

### 4.3 边缘清除（关键创新）

**问题**：车牌边框在二值化后可能与字符连接，形成一个大的连通域，导致 `cv2.RETR_EXTERNAL` 只检测到边框而检测不到内部字符。

**解决方案**（`char_segment.py:62-67`）：

```python
# 涂黑图像边缘 30 像素，断开边框与字符的连接
edge_width = 30
binary_image[:edge_width, :] = 0   # 上边
binary_image[-edge_width:, :] = 0  # 下边
binary_image[:, :edge_width] = 0   # 左边
binary_image[:, -edge_width:] = 0  # 右边
```

**鲁棒性体现**：
- 断开边框与字符的连接后，使用 `cv2.RETR_EXTERNAL` 只检测最外层轮廓即可
- 避免复杂的轮廓层级处理（`cv2.RETR_TREE`）
- 30 像素足够覆盖车牌边框宽度（车牌统一缩放到 800×400）

### 4.4 字符轮廓筛选

**实现代码**（`char_segment.py:69-109`）：

| 筛选条件 | 配置参数 | 范围 | 说明 |
|----------|---------|------|------|
| 高度比例 | `CHAR_HEIGHT_RATIO_MIN/MAX` | 20%-95% | 字符高度占车牌高度的比例 |
| 宽高比 | `CHAR_ASPECT_RATIO_MIN/MAX` | 0.1-1.5 | 排除过宽或过窄的轮廓 |
| 最小面积 | `CHAR_MIN_AREA` | 30像素 | 排除噪点 |
| 边缘距离 | `CHAR_EDGE_MARGIN` | 2像素 | 排除靠近边缘的轮廓 |
| Y 坐标 | - | 下部 2/3 | 只识别车牌号码，忽略上部地区名 |

**Y 坐标筛选代码**（`char_segment.py:107`）：
```python
# 过滤掉车牌上部1/3区域的字符（地区名如"广州"、"东莞"等）
char_contours = [c for c in char_contours if (c[1] + c[3] / 2) > img_height / 3]
```

**排序**：按 X 坐标从左到右排序（`char_segment.py:110`）

### 4.5 字符归一化

**实现代码**（`char_segment.py:157-195`）：

```python
def _normalize_char(self, char_image):
    h, w = char_image.shape[:2]

    # 保持宽高比缩放
    scale = min(TEMPLATE_WIDTH / w, TEMPLATE_HEIGHT / h) * CHAR_NORMALIZE_SCALE
    new_w = int(w * scale)
    new_h = int(h * scale)

    # 缩放
    resized = cv2.resize(char_image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 创建目标尺寸的画布（20×35），居中放置
    result = np.zeros((TEMPLATE_HEIGHT, TEMPLATE_WIDTH), dtype=np.uint8)
    x_offset = (TEMPLATE_WIDTH - new_w) // 2
    y_offset = (TEMPLATE_HEIGHT - new_h) // 2
    result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    return result
```

**处理逻辑**：
1. 保持字符原始宽高比缩放（避免变形）
2. 放在 20×35 黑色画布中心
3. 周围留黑边，保证字符不贴边

---

## 五、字符识别（模板匹配）

### 5.1 模板库

- **规格**：20×35 像素，二值图像
- **内容**：
  - 数字：0-9（10个）
  - 字母：A-Z（不含 I、O，24个）
  - 汉字：粤、广、佛、山、州、惠 等常见地区字

**加载代码**（`match_ocr.py:29-68`）：
```python
def _load_templates(self):
    for filename in os.listdir(self.templates_dir):
        # 从文件名提取字符（去掉扩展名）
        char_name = os.path.splitext(filename)[0]

        # 读取模板图像
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)

        # 确保模板为二值图像
        _, template = cv2.threshold(template, 127, 255, cv2.THRESH_BINARY)

        # 调整模板大小到标准尺寸
        template = cv2.resize(template, (TEMPLATE_WIDTH, TEMPLATE_HEIGHT))

        self.templates[char_name] = template
```

### 5.2 匹配算法（Jaccard + 像素相似度）

**实现代码**（`match_ocr.py:79-116`）：

```python
def _match_template(self, char_image, template):
    # 1. 计算前景重叠度（Jaccard 相似度）
    char_fg = (char_image > 127).astype(np.uint8)
    template_fg = (template > 127).astype(np.uint8)

    intersection = np.sum(char_fg & template_fg)
    union = np.sum(char_fg | template_fg)

    if union == 0:
        return 1.0  # 两张图都是全黑，完全匹配

    jaccard = intersection / union

    # 2. 在交集区域计算像素值差异
    if intersection > 0:
        intersection_mask = (char_fg & template_fg).astype(bool)
        diff = cv2.absdiff(char_image[intersection_mask], template[intersection_mask])
        pixel_sim = 1.0 - (np.mean(diff) / 255.0)
    else:
        pixel_sim = 0.0

    # 综合评分：结构相似度 70% + 像素相似度 30%
    similarity = 0.7 * jaccard + 0.3 * pixel_sim

    return similarity
```

**算法优势**：
- **Jaccard 相似度**：只关注前景像素（白色字符），不受背景影响
- **像素差异**：在重叠区域进一步比较灰度值，提高精度
- **权重分配**：结构相似度占主导（70%），像素细节辅助（30%）

**相比纯 SSD 的优势**：
- 对字符大小变化更鲁棒
- 对位置偏移更宽容
- 计算简单高效

### 5.3 置信度阈值

**实现代码**（`match_ocr.py:141-144`）：

```python
# 检查置信度阈值
if best_score < OCR_MIN_CONFIDENCE:  # 0.62
    return None, best_score

return best_char, best_score
```

- 相似度 ≥ 0.62：输出识别结果
- 相似度 < 0.62：返回 `None`，在结果中标记为 `?`

**车牌级过滤**（`visualize_result.py:95-100`）：
```python
# 过滤掉置信度不够的字符（问号）
filtered_string = plate_string.replace('?', '')

# 如果没有成功识别的字符，跳过
if not filtered_string:
    continue
```

**鲁棒性体现**：
- 避免强行输出错误结果
- 如果一个车牌所有字符都不确定（全是 `?`），则不显示该车牌
- 进一步筛选假阳性，提高最终输出的准确率

---

## 六、鲁棒性设计总结

### 6.1 车牌定位阶段

| 设计 | 实现位置 | 解决的问题 |
|------|---------|-----------|
| whiteness = V - S | `image_process.py:167` | 综合亮度和饱和度，更准确判断白色 |
| 梯度递增检测 (1%→41%) | `image_process.py:326-374` | 适应不同亮度车牌，从亮到暗逐步检测 |
| 区域排除机制 | `image_process.py:369-371` | 支持多车牌识别，避免重复检测（筛选后排除） |
| 长宽比过滤 (1.7-2.3) | `image_process.py:285-291` | 排除广告文字干扰 |
| 角度过滤 (±35°) | `image_process.py:260-263` | 排除非水平背景区域 |
| 矩形度过滤 (≥0.5) | `image_process.py:277-282` | 排除不规则轮廓 |
| IoU 合并 (≤0.3) | `image_process.py:425-444` | 去除重复检测的同一车牌 |
| 统一分辨率 (800×400) | `image_process.py:534-537` | 使后续处理参数稳定 |

### 6.2 字符分割阶段

| 设计 | 实现位置 | 解决的问题 |
|------|---------|-----------|
| 自适应二值化 | `char_segment.py:42-48` | 处理车牌光照不均 |
| 先开后闭形态学 | `char_segment.py:126-130` | 去噪 + 连接断裂笔画 |
| 边缘清除 (30像素) | `char_segment.py:62-67` | 断开边框与字符的连接 |
| Y 坐标筛选 (下部2/3) | `char_segment.py:107` | 只识别车牌号码，忽略地区名 |
| 保持宽高比归一化 | `char_segment.py:157-195` | 避免字符变形，提高识别准确率 |

### 6.3 字符识别阶段

| 设计 | 实现位置 | 解决的问题 |
|------|---------|-----------|
| Jaccard + 像素相似度 | `match_ocr.py:79-116` | 对形变鲁棒，计算高效 |
| 置信度阈值 (0.62) | `match_ocr.py:141` | 避免输出不确定结果 |
| 车牌级过滤 | `visualize_result.py:84-90` | 减少假阳性输出 |

---

## 七、可配置参数

系统提供大量可配置参数（位于 `config.py`），用户可根据实际场景调整：

### 预处理参数
```python
GAUSSIAN_KERNEL_SIZE = (5, 5)    # 高斯滤波核大小
CLAHE_CLIP_LIMIT = 2.0           # CLAHE 对比度限制
CLAHE_TILE_SIZE = (8, 8)         # CLAHE 分块大小
```

### 车牌定位参数
```python
ADAPTIVE_WHITE_TOP_PERCENT = 1    # 初始白色检测百分比
ADAPTIVE_WHITE_STEP = 2           # 检测递增步长
ADAPTIVE_WHITE_MAX_PERCENT = 41   # 最大检测百分比
PLATE_ASPECT_RATIO_MIN = 1.7      # 最小长宽比
PLATE_ASPECT_RATIO_MAX = 2.3      # 最大长宽比
PLATE_ANGLE_MAX = 35              # 最大允许角度（度）
PLATE_RECTANGULARITY_MIN = 0.5    # 最小矩形度
PLATE_AREA_MIN_RATIO = 0.01       # 最小面积比例
PLATE_AREA_MAX_RATIO = 0.20       # 最大面积比例
PLATE_STANDARD_WIDTH = 800        # 标准车牌宽度
PLATE_STANDARD_HEIGHT = 400       # 标准车牌高度
```

### 字符分割参数
```python
TEMPLATE_WIDTH = 20               # 字符模板宽度
TEMPLATE_HEIGHT = 35              # 字符模板高度
CHAR_BINARY_BLOCK_SIZE = 55       # 自适应二值化块大小
CHAR_BINARY_C = 7                 # 自适应二值化常数
CHAR_MORPH_OPEN_KERNEL = (7, 7)   # 开运算核大小
CHAR_MORPH_CLOSE_KERNEL = (7, 7)  # 闭运算核大小
CHAR_HEIGHT_RATIO_MIN = 0.2       # 字符最小高度比例
CHAR_HEIGHT_RATIO_MAX = 0.95      # 字符最大高度比例
CHAR_ASPECT_RATIO_MIN = 0.1       # 字符最小宽高比
CHAR_ASPECT_RATIO_MAX = 1.5       # 字符最大宽高比
CHAR_MIN_AREA = 30                # 字符最小面积
CHAR_EDGE_MARGIN = 2              # 字符距边缘最小距离
CHAR_NORMALIZE_SCALE = 1.0        # 字符归一化缩放比例
```

### 字符识别参数
```python
OCR_MIN_CONFIDENCE = 0.62         # 最小置信度阈值
```

### 并行处理参数
```python
PARALLEL_WORKERS = 5              # 并行处理进程数
```

---

## 八、性能优化

### 8.1 多进程并行

**实现代码**（`main.py:283-286`）：

```python
with ProcessPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
    futures = {executor.submit(_process_single_image_wrapper, task): task[0]
               for task in task_args}
    for future in as_completed(futures):
        results.append(future.result())
```

- **图像级并行**：每张图片串行处理，多张图片并行
- **默认 5 进程**：可根据 CPU 核心数调整
- **显著加速**：批量处理时间大幅缩短

### 8.2 算法时间复杂度

| 阶段 | 主要操作 | 复杂度 |
|------|---------|--------|
| 预处理 | 高斯滤波、CLAHE | O(n) |
| 车牌定位 | 颜色分割、轮廓检测 | O(n) |
| 字符分割 | 形态学、轮廓检测 | O(n) |
| 字符识别 | 模板匹配（逐模板） | O(m×k) |

其中 n 为图像像素数，m 为字符数，k 为模板数。

---

## 九、准确率评估方法

建议按以下方式统计准确率：

### 车牌级别准确率
```
准确率 = 完全正确识别的车牌数 / 总测试车牌数 × 100%
```

**判定标准**：车牌号码完全一致（忽略前缀汉字）

### 字符级别准确率
```
准确率 = 正确识别的字符数 / 总字符数 × 100%
```

**计数方式**：逐字符比对，正确则计数 +1

### 测试分类

建议按场景分类统计：

| 场景类别 | 说明 | 测试要点 |
|---------|------|---------|
| 正常场景 | 车牌清晰、光照均匀、无遮挡 | 基础识别能力 |
| 倾斜场景 | 车牌有明显倾斜（10°-30°） | 畸变校正能力 |
| 干扰场景 | 有广告文字、标志等干扰 | 长宽比过滤效果 |
| 多车牌场景 | 同一图像多个车牌 | 区域排除式检测 |
| 极端光照 | 过亮/过暗/逆光 | 梯度递增检测 |

### 召回率与精确率

```
召回率 (Recall) = 检测到的车牌数 / 实际车牌数 × 100%
精确率 (Precision) = 正确识别的车牌数 / 检测到的车牌数 × 100%
F1 分数 = 2 × (精确率 × 召回率) / (精确率 + 召回率)
```

---

## 十、扩展性

### 10.1 支持其他颜色车牌

当前系统针对**白色车牌**优化。如需支持蓝色、黄色、绿色车牌，可修改 `_adaptive_white_segmentation` 方法，添加 H 通道（色相）过滤：

```python
# 示例：检测蓝色车牌
h, s, v = cv2.split(hsv)
blue_mask = (h >= 100) & (h <= 130) & (s >= 50) & (v >= 50)
```

### 10.2 增加模板字符

只需在 `resources/templates/` 目录下添加新的 20×35 二值图像，文件名为字符内容（如 `京.png`），系统会自动加载。

### 10.3 调整检测策略

通过修改 `config.py` 中的参数，可以适应不同应用场景：
- **提高召回率**：增大 `ADAPTIVE_WHITE_MAX_PERCENT`、放宽长宽比范围
- **提高精确率**：提高 `OCR_MIN_CONFIDENCE`、缩小长宽比范围

---

## 十一、已知局限性

1. **仅支持白色车牌**：需要修改颜色检测算法才能支持其他颜色
2. **模板匹配限制**：字体变化大或污损严重时识别率下降
3. **光照极限**：过度曝光或完全黑暗场景无法识别
4. **遮挡处理**：当前未实现遮挡检测，被遮挡字符会识别为 `?`
5. **近似字符混淆**：如 0 和 O、1 和 I（已从模板中移除 I、O）

---

## 十二、代码文件说明

| 文件 | 行数 | 核心功能 |
|------|------|---------|
| `config.py` | ~98 | 所有可配置参数 |
| `image_process.py` | ~733 | 预处理、车牌定位、畸变校正 |
| `char_segment.py` | ~270 | 字符分割 |
| `match_ocr.py` | ~250 | 模板匹配识别 |
| `visualize_result.py` | ~222 | 结果可视化（支持中文） |
| `main.py` | ~322 | 主程序、并行处理 |
| `test_pipeline.py` | ~279 | 测试脚本、准确率统计 |

**总代码量**：约 2200 行（不含注释）
