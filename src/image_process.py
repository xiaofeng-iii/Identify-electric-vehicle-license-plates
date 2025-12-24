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
    DEBUG_MODE, DEBUG_DIR
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
            output_dir: 输出目录
            prefix: 文件名前缀
        """
        if not self.debug:
            return

        os.makedirs(output_dir, exist_ok=True)

        for name, image in self.debug_images.items():
            filename = f"{prefix}_{name}.jpg" if prefix else f"{name}.jpg"
            filepath = os.path.join(output_dir, filename)
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
    original, processed = preprocess_image(test_image_path)

    if processed is not None:
        print("预处理完成！")
        print(f"原始图像尺寸: {original.shape}")
        print(f"处理后图像尺寸: {processed.shape}")
    else:
        print("预处理失败！")
