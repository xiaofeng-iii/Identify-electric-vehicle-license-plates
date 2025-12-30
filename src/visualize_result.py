# -*- coding: utf-8 -*-
"""
结果可视化模块：在原图上标注车牌位置和识别结果
支持中文显示（使用PIL）
"""

import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont


def get_chinese_font(size=32):
    """
    获取支持中文的字体
    size: 字体大小
    """
    # Windows 常见中文字体路径
    font_paths = [
        "C:/Windows/Fonts/msyh.ttc",      # 微软雅黑
        "C:/Windows/Fonts/simsun.ttc",    # 宋体
        "C:/Windows/Fonts/simhei.ttf",    # 黑体
        "C:/Windows/Fonts/STFANGSO.TTF",  # 华文仿宋
    ]

    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, size)
            except Exception:
                continue

    # 如果找不到中文字体，使用默认字体
    return ImageFont.load_default()


def cv2_to_pil(cv2_img):
    """OpenCV图像转PIL图像"""
    return Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))


def pil_to_cv2(pil_img):
    """PIL图像转OpenCV图像"""
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def put_chinese_text(img, text, position, font_size=32, color=(255, 255, 255)):
    """
    在OpenCV图像上绘制中文文本

    参数:
        img: OpenCV BGR图像
        text: 要绘制的文本
        position: 文本位置 (x, y)
        font_size: 字体大小
        color: 文本颜色 (B, G, R)
    """
    pil_img = cv2_to_pil(img)
    draw = ImageDraw.Draw(pil_img)
    font = get_chinese_font(font_size)

    # PIL使用RGB颜色，需要转换
    rgb_color = (color[2], color[1], color[0])
    draw.text(position, text, font=font, fill=rgb_color)

    return pil_to_cv2(pil_img)


def draw_plate_result(image, candidates, plate_strings, output_path=None):
    """
    在原图上绘制车牌定位框和识别结果（不修改原图）

    参数:
        image: 原始BGR图像
        candidates: 车牌候选区域列表
        plate_strings: 车牌识别结果列表
        output_path: 输出图片路径，None则不保存
    """
    # 创建原图的副本，不修改原图
    result_img = image.copy()

    for i, ((rect, box, score), plate_string) in enumerate(zip(candidates, plate_strings)):
        # 过滤掉置信度不够的字符（问号）
        filtered_string = plate_string.replace('?', '')

        # 如果没有成功识别的字符，跳过
        if not filtered_string:
            continue

        # 绘制车牌边框（绿色）
        color = (0, 255, 0)
        cv2.drawContours(result_img, [box], 0, color, 2)

        # 计算文本位置（车牌上方）
        center_x = int(rect[0][0])

        # 获取边界框的最高点
        top_y = int(np.min(box[:, 1]))

        # 计算文本位置
        font_size = 200
        text_width = len(filtered_string) * int(font_size * 0.7)  # 估算文本宽度
        text_x = center_x - text_width // 2
        text_y = top_y - font_size - 100

        # 确保文本不超出图像边界
        if text_y < 10:
            text_y = top_y + int(rect[1][1]) + 10  # 放到车牌下方

        text_x = max(5, text_x)

        # 绘制半透明背景
        overlay = result_img.copy()
        bg_x1 = max(0, text_x - 30)
        bg_y1 = max(0, text_y)
        bg_x2 = min(result_img.shape[1], text_x + text_width + 25)
        bg_y2 = min(result_img.shape[0], text_y + int(font_size * 1.4))
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
        result_img = cv2.addWeighted(overlay, 0.6, result_img, 0.4, 0)

        # 绘制识别结果文本（白色，支持中文）
        result_img = put_chinese_text(result_img, filtered_string,
                                      (text_x, text_y), font_size, (255, 255, 255))

    # 保存结果
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, result_img)
        print(f"\n最终结果已保存: {output_path}")

    return result_img


def create_result_summary(image, candidates, plate_strings, recognition_details,
                         output_path=None):
    """
    创建详细的结果展示图，包含原图、车牌区域、识别详情
    只显示识别成功的字符，过滤掉置信度不够字符

    参数:
        image: 原始BGR图像
        candidates: 车牌候选区域列表
        plate_strings: 车牌识别结果列表
        recognition_details: 识别详情列表 [[(char, confidence, bbox), ...], ...]
        output_path: 输出图片路径
    """
    # 计算布局
    img_h, img_w = image.shape[:2]

    # 标注后的原图
    annotated_img = draw_plate_result(image, candidates, plate_strings)

    # 如果没有识别到车牌，直接返回标注图
    if not candidates:
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, annotated_img)
            print(f"结果已保存: {output_path}")
        return annotated_img

    # 创建车牌详情区域
    plate_detail_height = 100 * len(candidates)
    canvas = np.ones((img_h + plate_detail_height, img_w, 3), dtype=np.uint8) * 255

    # 放置标注图
    canvas[:img_h, :] = annotated_img

    # 绘制每个车牌的详情
    y_offset = img_h + 25

    for i, (plate_string, details) in enumerate(zip(plate_strings, recognition_details)):
        # 过滤掉置信度不够的字符（问号）
        filtered_string = plate_string.replace('?', '')

        # 车牌序号（英文用cv2）
        cv2.putText(canvas, f"Plate {i+1}:", (20, y_offset + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

        # 识别结果（中文用PIL）
        if filtered_string:
            canvas = put_chinese_text(canvas, filtered_string,
                                     (150, y_offset - 20), 36, (0, 128, 0))

            # 详细置信度信息（只显示成功识别的字符）
            detail_parts = []
            for char, confidence, _ in details:
                if char and char != '?' and confidence >= 0.5:
                    detail_parts.append(f"{char}({confidence:.2f})")

            if detail_parts:
                detail_text = " ".join(detail_parts)
                canvas = put_chinese_text(canvas, detail_text,
                                         (150, y_offset + 20), 18, (100, 100, 100))
        else:
            cv2.putText(canvas, "No valid characters", (150, y_offset + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        y_offset += 90

    # 保存
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, canvas)
        print(f"结果已保存: {output_path}")

    return canvas
    