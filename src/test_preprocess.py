# -*- coding: utf-8 -*-
"""
预处理模块测试脚本
用法: python test_preprocess.py [图片路径]
"""

import sys
import os
import cv2
import numpy as np

# 添加src目录到路径
sys.path.insert(0, os.path.dirname(__file__))

from image_process import ImagePreprocessor, preprocess_image
from config import DEBUG_DIR, RAW_IMAGES_DIR


def display_comparison(original, processed, window_name="预处理对比"):
    """
    显示原图和处理后图像的对比

    Args:
        original: 原始BGR图像
        processed: 处理后的灰度图像
        window_name: 窗口名称
    """
    # 将灰度图转为BGR以便并排显示
    processed_bgr = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

    # 调整到相同高度
    h1, w1 = original.shape[:2]
    h2, w2 = processed_bgr.shape[:2]

    # 缩放以适应屏幕（使用INTER_AREA插值，缩小时效果更好）
    max_height = 600
    if h1 > max_height:
        scale = max_height / h1
        original = cv2.resize(original, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        processed_bgr = cv2.resize(processed_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    # 并排显示
    comparison = np.hstack([original, processed_bgr])

    # 添加标签
    cv2.putText(comparison, "Original", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(comparison, "Preprocessed", (original.shape[1] + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow(window_name, comparison)
    print("按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_single_image(image_path):
    """测试单张图片的预处理"""
    print(f"\n{'='*50}")
    print(f"测试图片: {image_path}")
    print('='*50)

    # 执行预处理
    original, processed = preprocess_image(image_path, save_debug=True)

    if original is None:
        print("测试失败：无法读取图片")
        return False

    print(f"原始图像尺寸: {original.shape}")
    print(f"处理后图像尺寸: {processed.shape}")
    print(f"调试图片已保存到: {DEBUG_DIR}")

    # 显示对比
    display_comparison(original, processed)

    return True


def test_all_images_in_dir(dir_path=RAW_IMAGES_DIR):
    """测试目录下所有图片"""
    if not os.path.exists(dir_path):
        print(f"目录不存在: {dir_path}")
        return

    # 支持的图片格式
    extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    images = [f for f in os.listdir(dir_path) if f.lower().endswith(extensions)]

    if not images:
        print(f"目录中没有找到图片: {dir_path}")
        return

    print(f"找到 {len(images)} 张图片")

    for img_name in images:
        img_path = os.path.join(dir_path, img_name)
        test_single_image(img_path)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # 测试指定的图片
        test_path = sys.argv[1]
        if os.path.isdir(test_path):
            test_all_images_in_dir(test_path)
        else:
            test_single_image(test_path)
    else:
        # 默认处理 raw_images 目录下的所有图片
        print("未指定路径，默认处理 resources/raw_images/ 目录")
        test_all_images_in_dir(RAW_IMAGES_DIR)
