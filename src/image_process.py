# -*- coding: utf-8 -*-
"""
图像处理模块：包含预处理、车牌定位、畸变校正等功能
"""

import cv2
import numpy as np
import os
from config import (
    GAUSSIAN_KERNEL_SIZE, GAUSSIAN_SIGMA,
    CLAHE_CLIP_LIMIT, CLAHE_TILE_SIZE,
    DEBUG_MODE, DEBUG_DIR,
    PLATE_ASPECT_RATIO_MIN, PLATE_ASPECT_RATIO_MAX,
    PLATE_AREA_MIN_RATIO, PLATE_AREA_MAX_RATIO,
    PLATE_RECTANGULARITY_MIN, PLATE_ANGLE_MAX,
    PLATE_STANDARD_WIDTH, PLATE_STANDARD_HEIGHT,
    ADAPTIVE_WHITE_TOP_PERCENT,
    ADAPTIVE_WHITE_STEP, ADAPTIVE_WHITE_MAX_PERCENT
)


class ImagePreprocessor:
    """图像预处理类"""

    def __init__(self, debug=DEBUG_MODE):
        """
        初始化预处理器

        Args:
            debug: 是否保存调试图片
        """
        self.debug = debug
        self.debug_images = {}  # 存储中间结果

    def read_image(self, image_path):
        """
        读取图像

        Args:
            image_path: 图像文件路径

        Returns:
            BGR格式的图像数组，读取失败返回None
        """
        if not os.path.exists(image_path):
            print(f"错误: 图像文件不存在 - {image_path}")
            return None

        image = cv2.imread(image_path)
        if image is None:
            print(f"错误: 无法读取图像 - {image_path}")
            return None

        self.debug_images['original'] = image.copy()
        return image

    def to_grayscale(self, image):
        # 灰度化：将BGR图像转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.debug_images['grayscale'] = gray.copy()
        return gray

    def gaussian_blur(self, image):
        # 高斯滤波：去除图像噪声
        blurred = cv2.GaussianBlur(image, GAUSSIAN_KERNEL_SIZE, GAUSSIAN_SIGMA)
        self.debug_images['gaussian_blur'] = blurred.copy()
        return blurred

    def histogram_equalization(self, image):
        # 直方图均衡化：增强图像对比度，使用CLAHE（自适应直方图均衡化）
        # CLAHE: 对比度受限自适应直方图均衡化，避免过度增强噪声，保留局部细节
        clahe = cv2.createCLAHE(
            clipLimit=CLAHE_CLIP_LIMIT,
            tileGridSize=CLAHE_TILE_SIZE
        )
        equalized = clahe.apply(image)

        self.debug_images['histogram_equalization(final)'] = equalized.copy()
        return equalized

    def preprocess(self, image):
        # 完整预处理流程：灰度化 -> 高斯滤波 -> 直方图均衡化
        # 1. 灰度化
        gray = self.to_grayscale(image)

        # 2. 高斯滤波去噪
        blurred = self.gaussian_blur(gray)

        # 3. 直方图均衡化增强对比度
        enhanced = self.histogram_equalization(blurred)

        return enhanced

    def save_debug_images(self, output_dir=DEBUG_DIR, prefix=""):
        # 保存调试图片
        if not self.debug:
            return

        # 如果有prefix，在output_dir下创建以prefix命名的子文件夹
        if prefix:
            actual_output_dir = os.path.join(output_dir, prefix)
        else:
            actual_output_dir = output_dir

        os.makedirs(actual_output_dir, exist_ok=True)

        for idx, (name, image) in enumerate(self.debug_images.items()):
            # 使用序号前缀确保文件按处理顺序排列
            filename = f"{idx:02d}_{name}.jpg"
            filepath = os.path.join(actual_output_dir, filename)
            cv2.imwrite(filepath, image)
            print(f"已保存: {filepath}")

    def clear_debug_images(self):
        """清空调试图片缓存"""
        self.debug_images.clear()


def preprocess_image(image_path, save_debug=True, output_dir=DEBUG_DIR):
    """
    便捷函数：对单张图片进行预处理
    image_path: 图像文件路径
    save_debug: 是否保存调试图片
    output_dir: 调试图片输出目录
    """
    preprocessor = ImagePreprocessor(debug=save_debug)

    # 读取图像
    image = preprocessor.read_image(image_path)
    if image is None:
        return None, None

    # 执行预处理
    processed = preprocessor.preprocess(image)

    # 保存调试图片
    if save_debug:
        # 使用图像文件名作为前缀
        basename = os.path.splitext(os.path.basename(image_path))[0]
        preprocessor.save_debug_images(output_dir, prefix=basename)

    return image, processed


class PlateLocator:
    """车牌定位类：使用颜色方法定位车牌"""

    def __init__(self, debug=DEBUG_MODE):
        """初始化车牌定位器"""
        self.debug = debug
        self.debug_images = {}

    def _adaptive_white_segmentation(self, image, top_percent=None):
        """
        自适应白色分割：选取图像中最"白"的区域

        使用 whiteness = V - S 作为白色程度指标
        V高且S低的区域更接近白色
        """
        if top_percent is None:
            top_percent = ADAPTIVE_WHITE_TOP_PERCENT

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # 计算白色程度：whiteness = V - S，亮度高且饱和度低的更"白"
        whiteness = v.astype(np.int16) - s.astype(np.int16)

        # 计算白色程度阈值：取 top_percent 对应的百分位数
        whiteness_threshold = np.percentile(whiteness, 100 - top_percent)

        # 白色条件：只用 whiteness 阈值（已包含饱和度信息）
        white_mask = whiteness >= whiteness_threshold

        # 转换为uint8类型的掩码
        mask = (white_mask * 255).astype(np.uint8)

        if self.debug:
            # 保存白色程度图用于调试，归一化到0-255显示
            whiteness_normalized = ((whiteness - whiteness.min()) /
                                   (whiteness.max() - whiteness.min() + 1) * 255).astype(np.uint8)
            self.debug_images['adaptive_white_v_channel'] = whiteness_normalized
            self.debug_images['adaptive_white_threshold'] = mask.copy()
            print(f"    自适应白色阈值: whiteness(V-S) >= {whiteness_threshold:.0f}")

        return mask

    def color_locate(self, image):
        """颜色定位法：通过HSV颜色空间定位车牌"""
        # 白色检测：使用自适应方法
        white_mask = self._adaptive_white_segmentation(image)

        # 保存调试图片
        if self.debug:
            self.debug_images['color_white_mask'] = white_mask.copy()

        return white_mask

    def filter_candidates(self, contours, image_shape, debug_filter=False, max_angle=None,
                          collect_debug_info=None):
        """
        筛选候选区域：根据长宽比、面积、矩形度和角度过滤轮廓

        参数:
            image_shape: 图像尺寸 (height, width, ...)
            debug_filter: 是否输出过滤调试信息
            max_angle: 最大允许偏离角度，None则使用配置值
            collect_debug_info: 传入列表收集筛选信息

        返回:
            符合条件的候选矩形列表，每个元素为 (rect, box, score)
            - rect: 最小外接矩形 ，包含center, size, angle
            - box: 四个角点坐标
            - score: 评分
        """
        if max_angle is None:
            max_angle = PLATE_ANGLE_MAX

        img_height, img_width = image_shape[:2]
        img_area = img_height * img_width
        candidates = []

        for i, contour in enumerate(contours):
            # 计算轮廓面积
            area = cv2.contourArea(contour)

            # 面积过滤
            if area < img_area * PLATE_AREA_MIN_RATIO:
                if debug_filter:
                    print(f"    轮廓{i}: 面积{area:.0f}太小 (min={img_area * PLATE_AREA_MIN_RATIO:.0f})")
                continue
            if area > img_area * PLATE_AREA_MAX_RATIO:
                if debug_filter:
                    print(f"    轮廓{i}: 面积{area:.0f}太大 (max={img_area * PLATE_AREA_MAX_RATIO:.0f})")
                continue

            # 获取最小外接矩形
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int32(box)

            # 直接从 box 的四个顶点计算长边角度
            # box 顺序: 逆时针，从最低点开始
            # 计算相邻两条边的长度和角度
            edge1 = np.linalg.norm(box[0] - box[1])
            edge2 = np.linalg.norm(box[1] - box[2])

            # 确定长边和短边
            if edge1 >= edge2:
                long_edge_vec = box[1] - box[0]
                width, height = edge1, edge2
            else:
                long_edge_vec = box[2] - box[1]
                width, height = edge2, edge1

            # 计算长边与水平方向的夹角 (范围 -90° 到 90°)
            angle = np.degrees(np.arctan2(long_edge_vec[1], long_edge_vec[0]))

            # 角度偏离过滤
            angle_deviation = abs(angle)
            if angle_deviation > max_angle:
                if debug_filter:
                    print(f"    轮廓{i}: 角度偏离{angle_deviation:.1f}°超过阈值 (max={max_angle}°)")
                if collect_debug_info is not None:
                    collect_debug_info.append((contour, box, 'angle', f'角度:{angle:.1f}°'))
                continue

            # 避免除零
            if height == 0 or width == 0:
                continue

            # 计算矩形度 (轮廓面积 / 外接矩形面积)
            rect_area = width * height
            rectangularity = area / rect_area

            # 矩形度过滤：车牌应该是比较规整的矩形
            if rectangularity < PLATE_RECTANGULARITY_MIN:
                if debug_filter:
                    print(f"    轮廓{i}: 矩形度{rectangularity:.2f}太低 (min={PLATE_RECTANGULARITY_MIN})")
                if collect_debug_info is not None:
                    collect_debug_info.append((contour, box, 'rect', f'矩形度:{rectangularity:.2f}'))
                continue

            # 计算长宽比
            aspect_ratio = width / height

            # 长宽比过滤 (关键：排除广告文字等干扰)
            if aspect_ratio < PLATE_ASPECT_RATIO_MIN or aspect_ratio > PLATE_ASPECT_RATIO_MAX:
                if debug_filter:
                    print(f"    轮廓{i}: 长宽比{aspect_ratio:.2f}不符合 ({PLATE_ASPECT_RATIO_MIN}~{PLATE_ASPECT_RATIO_MAX})")
                if collect_debug_info is not None:
                    collect_debug_info.append((contour, box, 'ratio', f'比例:{aspect_ratio:.2f}'))
                continue

            # 计算评分：长宽比越接近3越好（中国车牌标准比例约为3:1）
            ratio_score = 1.0 - abs(aspect_ratio - 3.0) / 2.0
            # 矩形度越高越好
            rect_score = rectangularity
            # 角度越正越好（偏离越小越好）
            angle_score = 1.0 - angle_deviation / 45.0
            # 综合评分
            score = ratio_score * 0.5 + rect_score * 0.3 + angle_score * 0.2

            if debug_filter:
                print(f"    轮廓{i}: 通过! 面积={area:.0f}, 长宽比={aspect_ratio:.2f}, 矩形度={rectangularity:.2f}, 角度={angle:.1f}°")

            if collect_debug_info is not None:
                collect_debug_info.append((contour, box, 'PASS', f'比例:{aspect_ratio:.2f} 角度:{angle:.1f}°'))

            candidates.append((rect, box, score))

        # 按评分排序
        candidates.sort(key=lambda x: x[2], reverse=True)

        return candidates

    def find_contours(self, binary_image):
        """查找轮廓"""
        contours, _ = cv2.findContours(
            binary_image,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        return contours

    def locate(self, image):
        """
        定位车牌：使用颜色定位方法，支持自适应调整
        从配置的值开始，逐步增加白色检测百分比
        """
        img_shape = image.shape

        # 累积筛选调试信息，用于最后统一绘制
        all_filter_debug_info = [] if self.debug else None

        # 颜色定位 - 区域排除式多亮度检测
        print("  [颜色定位 - 区域排除式检测]")

        # 初始化排除掩码
        img_h, img_w = image.shape[:2]
        exclusion_mask = np.ones((img_h, img_w), dtype=np.uint8) * 255

        # 从配置的初始值开始尝试
        current_percent = ADAPTIVE_WHITE_TOP_PERCENT
        all_candidates = []

        while current_percent <= ADAPTIVE_WHITE_MAX_PERCENT:
            print(f"    尝试白色百分比: {current_percent}%")

            # 使用当前百分比进行白色分割
            color_mask = self._adaptive_white_segmentation(image, top_percent=current_percent)

            # 应用排除掩码，将已检测区域变黑
            color_mask = cv2.bitwise_and(color_mask, exclusion_mask)

            color_contours = self.find_contours(color_mask)
            print(f"    找到 {len(color_contours)} 个候选轮廓")

            color_candidates = self.filter_candidates(
                color_contours, img_shape, debug_filter=False,
                collect_debug_info=all_filter_debug_info
            )

            if len(color_candidates) > 0:
                print(f"    在 {current_percent}% 找到 {len(color_candidates)} 个候选车牌")
                all_candidates.extend(color_candidates)

                # 将检测到的区域从排除掩码中移除（涂黑）
                for rect, box, score in color_candidates:
                    expanded_box = self._expand_box(box, scale=1.2, img_shape=img_shape)
                    cv2.fillPoly(exclusion_mask, [expanded_box], 0)

            # 无论是否找到，都继续递增亮度
            current_percent += ADAPTIVE_WHITE_STEP

        if len(all_candidates) == 0:
            print(f"    警告: 达到最大百分比 {ADAPTIVE_WHITE_MAX_PERCENT}% 仍未找到车牌")
        else:
            print(f"  总共检测到 {len(all_candidates)} 个候选车牌")

        if self.debug:
            self.debug_images['locate_color_result'] = color_mask.copy()
            self.debug_images['locate_exclusion_mask'] = exclusion_mask.copy()

        # 合并重叠候选区域
        merged_candidates = self._merge_overlapping(all_candidates)

        verified_candidates = merged_candidates

        # 绘制调试图
        if self.debug:
            # 绘制筛选过程调试图（统一绘制所有筛选信息）
            if all_filter_debug_info and len(all_filter_debug_info) > 0:
                vis_img = image.copy()
                color_map = {
                    'angle': (0, 0, 255),    # 红色：角度不符
                    'rect': (0, 165, 255),   # 橙色：矩形度不符
                    'ratio': (0, 255, 255),  # 黄色：长宽比不符
                    'PASS': (0, 255, 0)      # 绿色：通过
                }
                for contour, box, reason, details in all_filter_debug_info:
                    color = color_map.get(reason, (255, 255, 255))
                    cv2.drawContours(vis_img, [box], 0, color, 2)
                    cv2.drawContours(vis_img, [contour], 0, color, 1)
                    center_x = int(np.mean(box[:, 0]))
                    center_y = int(np.mean(box[:, 1]))
                    cv2.putText(vis_img, reason, (center_x - 30, center_y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    cv2.putText(vis_img, details, (center_x - 40, center_y + 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                self.debug_images['filter_candidates_debug'] = vis_img

            # 绘制定位结果调试图
            debug_img = image.copy()
            for i, (rect, box, score) in enumerate(verified_candidates):
                color = (0, 255, 0) if i == 0 else (0, 255, 255)
                cv2.drawContours(debug_img, [box], 0, color, 2)
                center = (int(rect[0][0]), int(rect[0][1]))
                cv2.putText(debug_img, f'{score:.2f}', center,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            self.debug_images['locate_final_result'] = debug_img

        return verified_candidates

    def _merge_overlapping(self, candidates, iou_threshold=0.3):
        """合并重叠的候选区域"""
        if len(candidates) <= 1:
            return candidates

        # 按分数排序
        sorted_candidates = sorted(candidates, key=lambda x: x[2], reverse=True)
        merged = []

        while sorted_candidates:
            best = sorted_candidates.pop(0)
            merged.append(best)

            remaining = []
            for candidate in sorted_candidates:
                if self._calculate_iou(best[1], candidate[1]) < iou_threshold:
                    remaining.append(candidate)
            sorted_candidates = remaining

        return merged

    def _calculate_iou(self, box1, box2):
        """
        计算两个矩形的IoU，用于查找两个重叠矩形是否是同一个车牌
        """
        # 使用cv2.rotatedRectangleIntersection计算交集
        rect1 = cv2.minAreaRect(box1)
        rect2 = cv2.minAreaRect(box2)

        # 计算交集面积
        ret, intersection_points = cv2.rotatedRectangleIntersection(rect1, rect2)

        if ret == cv2.INTERSECT_NONE or intersection_points is None:
            return 0.0

        intersection_area = cv2.contourArea(intersection_points)

        # 计算并集面积
        area1 = rect1[1][0] * rect1[1][1]
        area2 = rect2[1][0] * rect2[1][1]
        union_area = area1 + area2 - intersection_area

        if union_area == 0:
            return 0.0

        return intersection_area / union_area

    def _expand_box(self, box, scale=1.2, img_shape=None):
        """扩大边界框，用于排除检测区域时留出边距"""
        # 计算中心点
        center = np.mean(box, axis=0)

        # 从中心向外扩大
        expanded = center + (box - center) * scale

        # 转为整数
        expanded = expanded.astype(np.int32)

        # 裁剪到图像边界内
        if img_shape is not None:
            img_h, img_w = img_shape[:2]
            expanded[:, 0] = np.clip(expanded[:, 0], 0, img_w - 1)
            expanded[:, 1] = np.clip(expanded[:, 1], 0, img_h - 1)

        return expanded

    def extract_plate_region(self, image, rect, box, padding=5):
        """提取车牌区域图像，使用最小旋转角度矫正到水平，并统一输出分辨率"""
        center = rect[0]

        # 从 box 计算长边角度（与 filter_candidates 中的逻辑一致）
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

        # 添加 padding
        width = int(width + padding * 2)
        height = int(height + padding * 2)

        # 旋转图像使车牌水平
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        img_height, img_width = image.shape[:2]
        rotated = cv2.warpAffine(image, rotation_matrix, (img_width, img_height))

        # 裁剪车牌区域
        center_x, center_y = int(center[0]), int(center[1])
        half_w, half_h = int(width / 2), int(height / 2)

        x1 = max(0, center_x - half_w)
        y1 = max(0, center_y - half_h)
        x2 = min(img_width, center_x + half_w)
        y2 = min(img_height, center_y + half_h)

        plate_region = rotated[y1:y2, x1:x2]

        # 统一缩放到标准分辨率，使后续处理参数更稳定
        if plate_region.size > 0:
            plate_region = cv2.resize(
                plate_region,
                (PLATE_STANDARD_WIDTH, PLATE_STANDARD_HEIGHT),
                interpolation=cv2.INTER_LINEAR
            )

        return plate_region

    def save_debug_images(self, output_dir=DEBUG_DIR, prefix=""):
        """保存调试图片"""
        if not self.debug:
            return

        # 如果有prefix，在output_dir下创建以prefix命名的子文件夹
        if prefix:
            actual_output_dir = os.path.join(output_dir, prefix)
        else:
            actual_output_dir = output_dir

        os.makedirs(actual_output_dir, exist_ok=True)

        for idx, (name, image) in enumerate(self.debug_images.items()):
            # 使用序号前缀确保文件按处理顺序排列，从10开始避免与预处理图片冲突
            filename = f"{10 + idx:02d}_{name}.jpg"
            filepath = os.path.join(actual_output_dir, filename)
            cv2.imwrite(filepath, image)
            print(f"已保存: {filepath}")

    def clear_debug_images(self):
        """清空调试图片缓存"""
        self.debug_images.clear()


def locate_plates(image, save_debug=True, output_dir=DEBUG_DIR, prefix=""):
    """便捷函数：定位图像中的车牌"""
    locator = PlateLocator(debug=save_debug)
    candidates = locator.locate(image)

    if save_debug:
        locator.save_debug_images(output_dir, prefix)

    return candidates, locator
