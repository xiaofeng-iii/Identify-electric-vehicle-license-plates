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
    WHITE_LOWER, WHITE_UPPER,
    ADAPTIVE_WHITE_TOP_PERCENT, ADAPTIVE_WHITE_MAX_SATURATION,
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
        """
        灰度化：将BGR图像转换为灰度图

        Args:
            image: BGR格式的图像

        Returns:
            灰度图像
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.debug_images['grayscale'] = gray.copy()
        return gray

    def gaussian_blur(self, image, kernel_size=GAUSSIAN_KERNEL_SIZE, sigma=GAUSSIAN_SIGMA):
        """
        高斯滤波：去除图像噪声

        Args:
            image: 输入图像（灰度或彩色）
            kernel_size: 高斯核大小，必须为奇数，如(5,5)
            sigma: 高斯核标准差，0表示自动计算

        Returns:
            滤波后的图像
        """
        blurred = cv2.GaussianBlur(image, kernel_size, sigma)
        self.debug_images['gaussian_blur'] = blurred.copy()
        return blurred

    def histogram_equalization(self, image, use_clahe=True):
        """
        直方图均衡化：增强图像对比度

        Args:
            image: 灰度图像
            use_clahe: 是否使用CLAHE（自适应直方图均衡化）

        Returns:
            均衡化后的图像
        """
        if use_clahe:
            # CLAHE: 对比度受限自适应直方图均衡化
            # 优点：避免过度增强噪声，保留局部细节
            clahe = cv2.createCLAHE(
                clipLimit=CLAHE_CLIP_LIMIT,
                tileGridSize=CLAHE_TILE_SIZE
            )
            equalized = clahe.apply(image)
        else:
            # 普通直方图均衡化
            equalized = cv2.equalizeHist(image)

        self.debug_images['histogram_equalization(final)'] = equalized.copy()
        return equalized

    def preprocess(self, image):
        """
        完整预处理流程：灰度化 -> 高斯滤波 -> 直方图均衡化

        Args:
            image: BGR格式的原始图像

        Returns:
            预处理后的灰度图像
        """
        # 1. 灰度化
        gray = self.to_grayscale(image)

        # 2. 高斯滤波去噪
        blurred = self.gaussian_blur(gray)

        # 3. 直方图均衡化增强对比度
        enhanced = self.histogram_equalization(blurred)

        return enhanced

    def save_debug_images(self, output_dir=DEBUG_DIR, prefix=""):
        """
        保存所有调试图片

        Args:
            output_dir: 输出目录（会在其下创建以prefix命名的子文件夹）
            prefix: 图片名称，用于创建子文件夹
        """
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

    Args:
        image_path: 图像文件路径
        save_debug: 是否保存调试图片
        output_dir: 调试图片输出目录

    Returns:
        (原始图像, 预处理后图像) 元组，失败返回 (None, None)
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
    """车牌定位类：使用颜色和边缘两种方法定位车牌"""

    def __init__(self, debug=DEBUG_MODE):
        """
        初始化车牌定位器

        Args:
            debug: 是否保存调试图片
        """
        self.debug = debug
        self.debug_images = {}

    def _color_segmentation(self, image, color_lower, color_upper):
        """
        颜色分割：在HSV空间中提取指定颜色区域

        Args:
            image: BGR格式图像
            color_lower: HSV下界 (H, S, V)
            color_upper: HSV上界 (H, S, V)

        Returns:
            二值化掩码图像
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(color_lower), np.array(color_upper))
        return mask

    def _adaptive_white_segmentation(self, image, top_percent=None, min_saturation=None):
        """
        自适应白色分割：选取图像中最亮的区域

        不使用固定阈值，而是根据图像本身的亮度分布，
        选取亮度值最高的 top_percent% 像素作为"白色"区域

        Args:
            image: BGR格式图像
            top_percent: 选取最亮的百分比，None则使用配置值
            min_saturation: 最大饱和度阈值，排除彩色区域，None则使用配置值

        Returns:
            二值化掩码图像
        """
        if top_percent is None:
            top_percent = ADAPTIVE_WHITE_TOP_PERCENT
        if min_saturation is None:
            min_saturation = ADAPTIVE_WHITE_MAX_SATURATION

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # 计算亮度阈值：取 top_percent 对应的百分位数
        v_threshold = np.percentile(v, 100 - top_percent)

        # 白色条件：亮度高 + 饱和度低（排除彩色高亮区域）
        white_mask = (v >= v_threshold) & (s <= min_saturation)

        # 转换为uint8类型的掩码
        mask = (white_mask * 255).astype(np.uint8)

        if self.debug:
            self.debug_images['adaptive_white_v_channel'] = v.copy()
            self.debug_images['adaptive_white_threshold'] = mask.copy()
            print(f"    自适应白色阈值: V >= {v_threshold:.0f}, S <= {min_saturation}")

        return mask

    def color_locate(self, image, use_adaptive_white=True):
        """
        颜色定位法：通过HSV颜色空间定位车牌

        Args:
            image: BGR格式原始图像
            use_adaptive_white: 是否使用自适应白色检测（默认True）

        Returns:
            二值化掩码
        """
        # 白色检测：使用自适应方法或固定阈值
        if use_adaptive_white:
            white_mask = self._adaptive_white_segmentation(image)
        else:
            white_mask = self._color_segmentation(image, WHITE_LOWER, WHITE_UPPER)

        # 保存调试图片
        if self.debug:
            self.debug_images['color_white_mask'] = white_mask.copy()

        return white_mask

    # def color_locate_v2(self, image):
    #     """
    #     颜色定位法v2：使用边缘辅助的颜色定位
    #
    #     先找颜色区域，再用边缘检测精确定位边框
    #
    #     Args:
    #         image: BGR格式原始图像
    #
    #     Returns:
    #         候选车牌轮廓列表
    #     """
    #     # 获取颜色掩码
    #     color_mask = self.color_locate(image)
    #
    #     # 使用较小的闭运算，仅连接相邻字符
    #     kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 3))
    #     closed = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel_small)
    #
    #     # 使用RETR_TREE获取轮廓层次结构
    #     contours, hierarchy = cv2.findContours(
    #         closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    #     )
    #
    #     if hierarchy is None:
    #         return []
    #
    #     # 找有子轮廓的区域（车牌内有字符）
    #     hierarchy = hierarchy[0]
    #     candidates_contours = []
    #
    #     for i, (contour, h) in enumerate(zip(contours, hierarchy)):
    #         # h = [next, prev, child, parent]
    #         # 检查是否有子轮廓
    #         child_idx = h[2]
    #         if child_idx == -1:
    #             # 没有子轮廓，但仍可能是车牌（被字符完全覆盖）
    #             # 使用面积和长宽比筛选
    #             area = cv2.contourArea(contour)
    #             if area < 500:  # 太小的跳过
    #                 continue
    #             candidates_contours.append(contour)
    #         else:
    #             # 有子轮廓，计算子轮廓数量
    #             child_count = 0
    #             idx = child_idx
    #             while idx != -1:
    #                 child_count += 1
    #                 idx = hierarchy[idx][0]  # next sibling
    #
    #             # 车牌通常有5-8个字符
    #             if child_count >= 3:
    #                 candidates_contours.append(contour)
    #
    #     if self.debug:
    #         self.debug_images['color_v2_closed'] = closed.copy()
    #
    #     return candidates_contours
    def filter_candidates(self, contours, image_shape, debug_filter=False, max_angle=None,
                          debug_image=None):
        """
        筛选候选区域：根据长宽比、面积、矩形度和角度过滤轮廓

        Args:
            contours: 轮廓列表
            image_shape: 图像尺寸 (height, width, ...)
            debug_filter: 是否输出过滤调试信息
            max_angle: 最大允许偏离角度，None则使用配置值
            debug_image: 用于可视化的原始图像，传入则会生成筛选过程可视化

        Returns:
            符合条件的候选矩形列表，每个元素为 (rect, box, score)
            - rect: 最小外接矩形 (center, size, angle)
            - box: 四个角点坐标
            - score: 评分 (越高越可能是车牌)
        """
        if max_angle is None:
            max_angle = PLATE_ANGLE_MAX

        img_height, img_width = image_shape[:2]
        img_area = img_height * img_width
        candidates = []

        # 用于调试可视化：记录通过面积筛选的候选块及其筛选结果
        filter_debug_info = []  # [(contour, box, reason, details)]

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

            # 标准化到 [-45, 45]，因为车牌长边接近水平时角度应接近0
            # 如果角度超出此范围，说明我们选的是"竖着"的方向，需要修正
            # while angle > 45:
            #     angle -= 90
            # while angle <= -45:
            #     angle += 90

            # 角度偏离过滤
            angle_deviation = abs(angle)
            if angle_deviation > max_angle:
                if debug_filter:
                    print(f"    轮廓{i}: 角度偏离{angle_deviation:.1f}°超过阈值 (max={max_angle}°)")
                if debug_image is not None:
                    filter_debug_info.append((contour, box, 'angle', f'角度:{angle:.1f}°'))
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
                if debug_image is not None:
                    filter_debug_info.append((contour, box, 'rect', f'矩形度:{rectangularity:.2f}'))
                continue

            # 计算长宽比
            aspect_ratio = width / height

            # 长宽比过滤 (关键：排除广告文字等干扰)
            if aspect_ratio < PLATE_ASPECT_RATIO_MIN or aspect_ratio > PLATE_ASPECT_RATIO_MAX:
                if debug_filter:
                    print(f"    轮廓{i}: 长宽比{aspect_ratio:.2f}不符合 ({PLATE_ASPECT_RATIO_MIN}~{PLATE_ASPECT_RATIO_MAX})")
                if debug_image is not None:
                    filter_debug_info.append((contour, box, 'ratio', f'比例:{aspect_ratio:.2f}'))
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

            if debug_image is not None:
                filter_debug_info.append((contour, box, 'PASS', f'比例:{aspect_ratio:.2f} 角度:{angle:.1f}°'))

            candidates.append((rect, box, score))

        # 生成筛选过程可视化图片
        if debug_image is not None and len(filter_debug_info) > 0:
            vis_img = debug_image.copy()
            # 颜色映射：不同筛选原因用不同颜色
            color_map = {
                'angle': (0, 0, 255),    # 红色：角度不符
                'rect': (0, 165, 255),   # 橙色：矩形度不符
                'ratio': (0, 255, 255),  # 黄色：长宽比不符
                'PASS': (0, 255, 0)      # 绿色：通过
            }
            for idx, (contour, box, reason, details) in enumerate(filter_debug_info):
                color = color_map.get(reason, (255, 255, 255))
                # 绘制外接矩形
                cv2.drawContours(vis_img, [box], 0, color, 2)
                # 绘制轮廓
                cv2.drawContours(vis_img, [contour], 0, color, 1)
                # 标注信息
                center_x = int(np.mean(box[:, 0]))
                center_y = int(np.mean(box[:, 1]))
                label = f"{idx}:{reason}"
                cv2.putText(vis_img, label, (center_x - 30, center_y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(vis_img, details, (center_x - 40, center_y + 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            self.debug_images['filter_candidates_debug'] = vis_img

        # 按评分排序
        candidates.sort(key=lambda x: x[2], reverse=True)

        return candidates

    def find_contours(self, binary_image):
        """
        查找轮廓

        Args:
            binary_image: 二值化图像

        Returns:
            轮廓列表
        """
        contours, _ = cv2.findContours(
            binary_image,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        return contours

    def locate(self, image, method='combined'):
        """
        定位车牌：综合使用颜色和边缘方法，支持自适应调整

        如果初始参数未检出车牌，会逐步增加白色检测百分比直到找到车牌

        Args:
            image: BGR格式原始图像
            method: 定位方法
                - 'color': 仅使用颜色定位
                - 'edge': 仅使用边缘定位
                - 'combined': 综合使用两种方法（默认）

        Returns:
            候选车牌区域列表，每个元素为 (rect, box, score)
        """
        img_shape = image.shape

        if method in ('color', 'combined'):
            # 颜色定位 - 使用自适应白色检测，支持动态调整
            print("  [颜色定位]")

            # 从配置的初始值开始尝试
            current_percent = ADAPTIVE_WHITE_TOP_PERCENT
            all_candidates = []

            while current_percent <= ADAPTIVE_WHITE_MAX_PERCENT:
                print(f"    尝试白色百分比: {current_percent}%")

                # 使用当前百分比进行白色分割
                color_mask = self._adaptive_white_segmentation(image, top_percent=current_percent)
                color_contours = self.find_contours(color_mask)
                print(f"    找到 {len(color_contours)} 个候选轮廓")

                color_candidates = self.filter_candidates(
                    color_contours, img_shape, debug_filter=False,
                    debug_image=image if self.debug else None
                )

                if len(color_candidates) > 0:
                    # 找到候选车牌，停止搜索
                    print(f"    成功! 在 {current_percent}% 时找到 {len(color_candidates)} 个候选车牌")
                    all_candidates = color_candidates
                    break
                else:
                    # 未找到，增加百分比继续尝试
                    current_percent += ADAPTIVE_WHITE_STEP

            if len(all_candidates) == 0:
                print(f"    警告: 达到最大百分比 {ADAPTIVE_WHITE_MAX_PERCENT}% 仍未找到车牌")

            if self.debug:
                self.debug_images['locate_color_result'] = color_mask.copy()

        else:
            all_candidates = []

        # 合并重叠候选区域
        merged_candidates = self._merge_overlapping(all_candidates)

        verified_candidates = merged_candidates

        # # 边缘密度验证：过滤掉内部没有足够字符特征的区域
        # verified_candidates = []
        # for rect, box, score in merged_candidates:
        #     edge_density = self._calculate_edge_density(image, rect, box)
        #     print(f"  候选区域: 中心={rect[0]}, 边缘密度={edge_density:.4f}")
        #     if edge_density >= PLATE_EDGE_DENSITY_MIN:
        #         # 将边缘密度纳入评分
        #         new_score = score * 0.7 + edge_density * 0.3
        #         verified_candidates.append((rect, box, new_score))

        # # 重新按分数排序
        # verified_candidates.sort(key=lambda x: x[2], reverse=True)

        # 绘制定位结果调试图
        if self.debug:
            debug_img = image.copy()
            for i, (rect, box, score) in enumerate(verified_candidates):
                color = (0, 255, 0) if i == 0 else (0, 255, 255)
                cv2.drawContours(debug_img, [box], 0, color, 2)
                center = (int(rect[0][0]), int(rect[0][1]))
                cv2.putText(debug_img, f'{score:.2f}', center,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            self.debug_images['locate_final_result'] = debug_img

        return verified_candidates

    def _calculate_edge_density(self, image, rect, box):
        """
        计算候选区域的边缘密度

        真正的车牌内部应有字符，会产生丰富的边缘
        而纯色地面边缘很少

        Args:
            image: 原始BGR图像
            rect: 最小外接矩形
            box: 四角点坐标

        Returns:
            边缘密度值 (0~1)
        """
        # 提取候选区域
        plate_img = self.extract_plate_region(image, rect, box, padding=0)
        if plate_img.size == 0:
            return 0.0

        # 转灰度
        if len(plate_img.shape) == 3:
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_img

        # Canny边缘检测
        edges = cv2.Canny(gray, 100, 200)

        # 计算边缘像素占比
        edge_pixels = np.count_nonzero(edges)
        total_pixels = edges.size

        if total_pixels == 0:
            return 0.0

        density = edge_pixels / total_pixels
        return density

    def _merge_overlapping(self, candidates, iou_threshold=0.3):
        """
        合并重叠的候选区域

        Args:
            candidates: 候选区域列表
            iou_threshold: IoU阈值，超过此值认为重叠

        Returns:
            合并后的候选区域列表
        """
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
        计算两个旋转矩形的IoU

        Args:
            box1, box2: 四角点坐标数组

        Returns:
            IoU值
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

    def extract_plate_region(self, image, rect, box, padding=5):
        """
        提取车牌区域图像，使用最小旋转角度矫正到水平

        Args:
            image: 原始图像
            rect: 最小外接矩形 (center, size, angle)
            box: 四个角点坐标
            padding: 边缘填充像素

        Returns:
            提取的车牌区域图像（已矫正为水平）
        """
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

        # 旋转图像使车牌水平（旋转 -angle 度）
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

        # 统一缩放到标准尺寸，使后续处理参数更稳定
        if plate_region.size > 0:
            plate_region = cv2.resize(
                plate_region,
                (PLATE_STANDARD_WIDTH, PLATE_STANDARD_HEIGHT),
                interpolation=cv2.INTER_LINEAR
            )

        return plate_region

    def save_debug_images(self, output_dir=DEBUG_DIR, prefix=""):
        """
        保存调试图片

        Args:
            output_dir: 输出目录（会在其下创建以prefix命名的子文件夹）
            prefix: 图片名称，用于创建子文件夹
        """
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


def locate_plates(image, method='combined', save_debug=True, output_dir=DEBUG_DIR, prefix=""):
    """
    便捷函数：定位图像中的车牌

    Args:
        image: BGR格式图像
        method: 定位方法 ('color', 'edge', 'combined')
        save_debug: 是否保存调试图片
        output_dir: 调试图片输出目录
        prefix: 文件名前缀

    Returns:
        候选车牌区域列表
    """
    locator = PlateLocator(debug=save_debug)
    candidates = locator.locate(image, method)

    if save_debug:
        locator.save_debug_images(output_dir, prefix)

    return candidates, locator


if __name__ == "__main__":
    # 测试代码
    import sys

    if len(sys.argv) > 1:
        test_image_path = sys.argv[1]
    else:
        # 默认测试图片路径
        test_image_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'resources', 'raw_images', 'test.jpg'
        )

    print(f"测试图片: {test_image_path}")

    # 测试预处理
    original, processed = preprocess_image(test_image_path)

    if processed is not None:
        print("预处理完成！")
        print(f"原始图像尺寸: {original.shape}")
        print(f"处理后图像尺寸: {processed.shape}")

        # 测试车牌定位
        print("\n开始车牌定位...")
        basename = os.path.splitext(os.path.basename(test_image_path))[0]
        candidates, locator = locate_plates(original, method='combined', prefix=basename)

        print(f"找到 {len(candidates)} 个候选车牌区域")
        for i, (rect, box, score) in enumerate(candidates):
            print(f"  候选 {i+1}: 分数={score:.3f}, 中心={rect[0]}, 尺寸={rect[1]}, 角度={rect[2]:.1f}°")

            # 提取车牌区域
            plate_img = locator.extract_plate_region(original, rect, box)
            if plate_img.size > 0:
                plate_dir = os.path.join(DEBUG_DIR, basename)
                os.makedirs(plate_dir, exist_ok=True)
                plate_path = os.path.join(plate_dir, f"20_plate_{i+1}.jpg")
                cv2.imwrite(plate_path, plate_img)
                print(f"  已保存车牌区域: {plate_path}")
    else:
        print("预处理失败！")
