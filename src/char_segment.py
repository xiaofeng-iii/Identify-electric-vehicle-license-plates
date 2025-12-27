# -*- coding: utf-8 -*-
"""
字符分割模块：使用连通域分析从车牌图像中分割出单个字符
"""

import cv2
import numpy as np
import os
from config import (
    DEBUG_MODE, DEBUG_DIR,
    TEMPLATE_WIDTH, TEMPLATE_HEIGHT,
    CHAR_BINARY_BLOCK_SIZE, CHAR_BINARY_C, CHAR_BINARY_INVERT,
    CHAR_MORPH_CLOSE_KERNEL, CHAR_MORPH_OPEN_KERNEL,
    CHAR_HEIGHT_RATIO_MIN, CHAR_HEIGHT_RATIO_MAX,
    CHAR_ASPECT_RATIO_MIN, CHAR_ASPECT_RATIO_MAX,
    CHAR_MIN_AREA, CHAR_EDGE_MARGIN,
    CHAR_NORMALIZE_SCALE
)


class CharacterSegmenter:
    """字符分割类：使用连通域分析分割车牌字符"""

    def __init__(self, debug=DEBUG_MODE):
        """
        初始化字符分割器

        Args:
            debug: 是否保存调试图片
        """
        self.debug = debug
        self.debug_images = {}

    def preprocess_plate(self, plate_image):
        """
        车牌图像预处理：灰度化 + 二值化

        Args:
            plate_image: BGR格式的车牌图像

        Returns:
            二值化图像
        """
        # 灰度化
        if len(plate_image.shape) == 3:
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_image.copy()

        if self.debug:
            self.debug_images['char_gray'] = gray.copy()

        # 自适应二值化（处理光照不均）
        # 白底黑字车牌：字符是黑色，背景是白色
        thresh_type = cv2.THRESH_BINARY_INV if CHAR_BINARY_INVERT else cv2.THRESH_BINARY
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresh_type,
            blockSize=CHAR_BINARY_BLOCK_SIZE,
            C=CHAR_BINARY_C
        )

        if self.debug:
            self.debug_images['char_binary'] = binary.copy()

        return binary

    def find_char_contours(self, binary_image):
        """
        使用连通域分析找到字符轮廓

        Args:
            binary_image: 二值化图像

        Returns:
            字符轮廓列表，按x坐标排序
        """
        img_height, img_width = binary_image.shape[:2]

        # 查找轮廓
        contours, _ = cv2.findContours(
            binary_image,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # 筛选字符轮廓
        char_contours = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # 字符高度应占车牌高度的一定比例
            height_ratio = h / img_height
            if height_ratio < CHAR_HEIGHT_RATIO_MIN or height_ratio > CHAR_HEIGHT_RATIO_MAX:
                continue

            # 字符宽高比应在合理范围
            aspect_ratio = w / h
            if aspect_ratio < CHAR_ASPECT_RATIO_MIN or aspect_ratio > CHAR_ASPECT_RATIO_MAX:
                continue

            # 字符不应太靠近边缘（去除边框）
            if x < CHAR_EDGE_MARGIN or y < CHAR_EDGE_MARGIN or x + w > img_width - CHAR_EDGE_MARGIN:
                continue

            # 面积过滤
            area = cv2.contourArea(contour)
            if area < CHAR_MIN_AREA:
                continue

            char_contours.append((x, y, w, h, contour))

        # 双排排序：先按y坐标分上下排，再按x坐标排序
        # 电动车车牌上排是地区名（如"广州"），下排是号码（如"QK7601"）
        if len(char_contours) > 0:
            # 计算所有字符的y坐标中心
            y_centers = [(c[1] + c[3] / 2) for c in char_contours]
            y_threshold = (min(y_centers) + max(y_centers)) / 2

            # 分为上排和下排
            top_row = [c for c in char_contours if (c[1] + c[3] / 2) < y_threshold]
            bottom_row = [c for c in char_contours if (c[1] + c[3] / 2) >= y_threshold]

            # 每排按x坐标排序（从左到右）
            top_row.sort(key=lambda c: c[0])
            bottom_row.sort(key=lambda c: c[0])

            # 合并：上排在前，下排在后
            char_contours = top_row + bottom_row

        return char_contours

    def extract_characters(self, plate_image, normalize=True):
        """
        从车牌图像中提取字符

        Args:
            plate_image: BGR格式的车牌图像
            normalize: 是否归一化到标准尺寸

        Returns:
            字符图像列表，每个元素为 (char_image, bbox)
            - char_image: 字符图像
            - bbox: 边界框 (x, y, w, h)
        """
        # 预处理
        binary = self.preprocess_plate(plate_image)

        # 形态学操作：闭运算连接断裂笔画，开运算去除噪点
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, CHAR_MORPH_CLOSE_KERNEL)
        open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, CHAR_MORPH_OPEN_KERNEL)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel)

        if self.debug:
            self.debug_images['char_morphology'] = binary.copy()

        # 找字符轮廓
        char_contours = self.find_char_contours(binary)

        # 提取字符图像
        characters = []
        for x, y, w, h, contour in char_contours:
            # 提取字符区域
            char_img = binary[y:y+h, x:x+w]

            # 归一化到标准尺寸
            if normalize:
                char_img = self._normalize_char(char_img)

            characters.append((char_img, (x, y, w, h)))

        # 调试：绘制分割结果
        if self.debug:
            debug_img = plate_image.copy()
            for i, (x, y, w, h, _) in enumerate(char_contours):
                cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 1)
                cv2.putText(debug_img, str(i), (x, y-2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
            self.debug_images['char_segment_result'] = debug_img

        return characters

    def _normalize_char(self, char_image, target_width=TEMPLATE_WIDTH,
                        target_height=TEMPLATE_HEIGHT):
        """
        将字符图像归一化到标准尺寸

        Args:
            char_image: 原始字符图像
            target_width: 目标宽度
            target_height: 目标高度

        Returns:
            归一化后的字符图像
        """
        h, w = char_image.shape[:2]

        # 保持宽高比缩放
        scale = min(target_width / w, target_height / h) * CHAR_NORMALIZE_SCALE

        new_w = int(w * scale)
        new_h = int(h * scale)

        # 缩放
        resized = cv2.resize(char_image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # 创建目标尺寸的画布，居中放置
        result = np.zeros((target_height, target_width), dtype=np.uint8)
        x_offset = (target_width - new_w) // 2
        y_offset = (target_height - new_h) // 2
        result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

        return result

    def save_debug_images(self, output_dir=DEBUG_DIR, prefix=""):
        """
        保存调试图片

        Args:
            output_dir: 输出目录
            prefix: 文件名前缀
        """
        if not self.debug:
            return

        if prefix:
            actual_output_dir = os.path.join(output_dir, prefix)
        else:
            actual_output_dir = output_dir

        os.makedirs(actual_output_dir, exist_ok=True)

        for idx, (name, image) in enumerate(self.debug_images.items()):
            filename = f"{30 + idx:02d}_{name}.jpg"
            filepath = os.path.join(actual_output_dir, filename)
            cv2.imwrite(filepath, image)
            print(f"已保存: {filepath}")

    def clear_debug_images(self):
        """清空调试图片缓存"""
        self.debug_images.clear()


def segment_characters(plate_image, save_debug=True, output_dir=DEBUG_DIR, prefix=""):
    """
    便捷函数：分割车牌字符

    Args:
        plate_image: 车牌图像
        save_debug: 是否保存调试图片
        output_dir: 调试图片输出目录
        prefix: 文件名前缀

    Returns:
        字符图像列表
    """
    segmenter = CharacterSegmenter(debug=save_debug)
    characters = segmenter.extract_characters(plate_image)

    if save_debug:
        segmenter.save_debug_images(output_dir, prefix)

    return characters, segmenter


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        plate_path = sys.argv[1]
    else:
        # 默认测试
        plate_path = os.path.join(DEBUG_DIR, "test", "20_plate_1.jpg")

    if not os.path.exists(plate_path):
        print(f"车牌图片不存在: {plate_path}")
        sys.exit(1)

    print(f"测试车牌图片: {plate_path}")

    plate_img = cv2.imread(plate_path)
    if plate_img is None:
        print("无法读取图片")
        sys.exit(1)

    basename = os.path.splitext(os.path.basename(plate_path))[0]
    characters, segmenter = segment_characters(plate_img, prefix=basename)

    print(f"分割出 {len(characters)} 个字符")
    for i, (char_img, bbox) in enumerate(characters):
        print(f"  字符 {i+1}: 位置={bbox}")
        # 保存单个字符
        char_path = os.path.join(DEBUG_DIR, basename, f"char_{i+1}.jpg")
        cv2.imwrite(char_path, char_img)
        print(f"    已保存: {char_path}")
