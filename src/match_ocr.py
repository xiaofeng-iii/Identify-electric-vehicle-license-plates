# -*- coding: utf-8 -*-
"""
字符识别模块：使用模板匹配识别分割后的字符
"""

import cv2
import numpy as np
import os
from config import (
    DEBUG_MODE, DEBUG_DIR, TEMPLATES_DIR,
    TEMPLATE_WIDTH, TEMPLATE_HEIGHT,
    OCR_MATCH_METHOD, OCR_MIN_CONFIDENCE
)


class TemplateOCR:
    """模板匹配OCR类"""

    def __init__(self, templates_dir=TEMPLATES_DIR, debug=DEBUG_MODE):
        """
        初始化模板OCR

        Args:
            templates_dir: 模板目录路径
            debug: 是否保存调试图片
        """
        self.templates_dir = templates_dir
        self.debug = debug
        self.debug_images = {}
        self.templates = {}  # {字符: 模板图像}

        # 加载模板
        self._load_templates()

    def _load_templates(self):
        """加载所有字符模板"""
        if not os.path.exists(self.templates_dir):
            print(f"警告: 模板目录不存在 - {self.templates_dir}")
            return

        # 支持的模板格式
        extensions = ('.jpg', '.jpeg', '.png', '.bmp')

        for filename in os.listdir(self.templates_dir):
            if not filename.lower().endswith(extensions):
                continue

            # 从文件名提取字符（去掉扩展名）
            char_name = os.path.splitext(filename)[0]

            # 读取模板图像
            template_path = os.path.join(self.templates_dir, filename)
            template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)

            if template is None:
                print(f"警告: 无法读取模板 - {template_path}")
                continue

            # 确保模板为二值图像
            _, template = cv2.threshold(template, 127, 255, cv2.THRESH_BINARY)

            # 调整模板大小到标准尺寸
            template = cv2.resize(template, (TEMPLATE_WIDTH, TEMPLATE_HEIGHT),
                                 interpolation=cv2.INTER_AREA)

            self.templates[char_name] = template

        print(f"已加载 {len(self.templates)} 个字符模板")

    def _preprocess_char(self, char_image):
        """
        预处理待识别的字符图像

        Args:
            char_image: 字符图像（可以是灰度或二值）

        Returns:
            预处理后的二值图像
        """
        # 确保是灰度图
        if len(char_image.shape) == 3:
            char_image = cv2.cvtColor(char_image, cv2.COLOR_BGR2GRAY)

        # 调整到标准尺寸
        resized = cv2.resize(char_image, (TEMPLATE_WIDTH, TEMPLATE_HEIGHT),
                            interpolation=cv2.INTER_AREA)

        # 二值化
        _, binary = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)

        return binary

    def _match_template(self, char_image, template):
        """
        计算字符图像与模板的匹配度

        Args:
            char_image: 预处理后的字符图像
            template: 模板图像

        Returns:
            匹配度分数 (0~1, 越高越匹配)
        """
        if OCR_MATCH_METHOD == 'ssd':
            # SSD改进版：结合XOR和像素差异
            # 1. 计算前景重叠度（Jaccard相似度）
            char_fg = (char_image > 127).astype(np.uint8)
            template_fg = (template > 127).astype(np.uint8)

            intersection = np.sum(char_fg & template_fg)
            union = np.sum(char_fg | template_fg)

            if union == 0:
                # 两张图都是全黑，完全匹配
                return 1.0

            # Jaccard相似度（交集/并集）
            jaccard = intersection / union

            # 2. 在交集区域计算像素值差异
            if intersection > 0:
                intersection_mask = (char_fg & template_fg).astype(bool)
                diff = cv2.absdiff(char_image[intersection_mask], template[intersection_mask])
                pixel_sim = 1.0 - (np.mean(diff) / 255.0)
            else:
                pixel_sim = 0.0

            # 综合评分：结构相似度(Jaccard) + 像素相似度
            similarity = 0.7 * jaccard + 0.3 * pixel_sim

        elif OCR_MATCH_METHOD == 'ncc':
            # NCC (Normalized Cross-Correlation) - 归一化互相关
            result = cv2.matchTemplate(char_image, template, cv2.TM_CCOEFF_NORMED)
            similarity = result[0][0]
            # 归一化到 0~1
            similarity = (similarity + 1) / 2

        elif OCR_MATCH_METHOD == 'xor':
            # XOR - 异或比较（只看不同的地方）
            xor_result = cv2.bitwise_xor(char_image, template)
            diff_pixels = np.count_nonzero(xor_result)
            total_pixels = char_image.size
            similarity = 1.0 - (diff_pixels / total_pixels)

        else:
            # 默认使用模板匹配 TM_CCOEFF_NORMED
            result = cv2.matchTemplate(char_image, template, cv2.TM_CCOEFF_NORMED)
            similarity = result[0][0]
            similarity = (similarity + 1) / 2

        return similarity

    def recognize_char(self, char_image):
        """
        识别单个字符

        Args:
            char_image: 字符图像

        Returns:
            (识别结果字符, 置信度) 或 (None, 0) 如果无法识别
        """
        if len(self.templates) == 0:
            print("错误: 没有加载任何模板")
            return None, 0

        # 预处理
        processed = self._preprocess_char(char_image)

        # 与所有模板匹配
        best_char = None
        best_score = -1

        for char_name, template in self.templates.items():
            score = self._match_template(processed, template)
            if score > best_score:
                best_score = score
                best_char = char_name

        # 检查置信度阈值
        if best_score < OCR_MIN_CONFIDENCE:
            return None, best_score

        return best_char, best_score

    def recognize_plate(self, char_images):
        """
        识别整个车牌的所有字符

        Args:
            char_images: 字符图像列表，每个元素为 (char_image, bbox)

        Returns:
            识别结果列表，每个元素为 (字符, 置信度, bbox)
        """
        results = []

        for char_img, bbox in char_images:
            char, confidence = self.recognize_char(char_img)
            results.append((char, confidence, bbox))

        return results

    def get_plate_string(self, results):
        """
        从识别结果生成车牌字符串

        Args:
            results: recognize_plate 返回的结果列表

        Returns:
            车牌字符串
        """
        chars = []
        for char, confidence, bbox in results:
            if char is not None:
                chars.append(char)
            else:
                chars.append('?')  # 无法识别的字符用?表示
        return ''.join(chars)

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
            filename = f"{40 + idx:02d}_{name}.jpg"
            filepath = os.path.join(actual_output_dir, filename)
            cv2.imwrite(filepath, image)
            print(f"已保存: {filepath}")

    def clear_debug_images(self):
        """清空调试图片缓存"""
        self.debug_images.clear()


def recognize_characters(char_images, save_debug=True, output_dir=DEBUG_DIR, prefix=""):
    """
    便捷函数：识别字符列表

    Args:
        char_images: 字符图像列表，每个元素为 (char_image, bbox)
        save_debug: 是否保存调试图片
        output_dir: 调试图片输出目录
        prefix: 文件名前缀

    Returns:
        (车牌字符串, 识别结果详情列表, OCR对象)
    """
    ocr = TemplateOCR(debug=save_debug)
    results = ocr.recognize_plate(char_images)
    plate_string = ocr.get_plate_string(results)

    if save_debug:
        ocr.save_debug_images(output_dir, prefix)

    return plate_string, results, ocr


if __name__ == "__main__":
    import sys

    # 测试：识别已分割的字符
    if len(sys.argv) > 1:
        char_dir = sys.argv[1]
    else:
        char_dir = os.path.join(DEBUG_DIR, "test")

    if not os.path.exists(char_dir):
        print(f"字符目录不存在: {char_dir}")
        sys.exit(1)

    print(f"测试字符目录: {char_dir}")

    # 找到所有字符图片
    char_files = sorted([f for f in os.listdir(char_dir)
                        if f.startswith("30_char_") and f.endswith(".jpg")])

    if not char_files:
        print("未找到字符图片 (30_char_*.jpg)")
        sys.exit(1)

    # 加载字符图片
    char_images = []
    for f in char_files:
        char_path = os.path.join(char_dir, f)
        char_img = cv2.imread(char_path, cv2.IMREAD_GRAYSCALE)
        if char_img is not None:
            char_images.append((char_img, None))
            print(f"  加载: {f}")

    # 识别
    ocr = TemplateOCR()
    results = ocr.recognize_plate(char_images)

    print(f"\n识别结果:")
    for i, (char, confidence, _) in enumerate(results):
        status = char if char else "?"
        print(f"  字符 {i+1}: {status} (置信度: {confidence:.3f})")

    plate_string = ocr.get_plate_string(results)
    print(f"\n车牌号: {plate_string}")
