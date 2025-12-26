# -*- coding: utf-8 -*-
"""
结果可视化模块：在原图上标注车牌位置和识别结果
"""

import cv2
import numpy as np
import os
from config import OUTPUT_DIR


def draw_plate_result(image, candidates, plate_strings, output_path=None):
    """
    在原图上绘制车牌定位框和识别结果（不修改原图）

    Args:
        image: 原始BGR图像
        candidates: 车牌候选区域列表 [(rect, box, score), ...]
        plate_strings: 车牌识别结果列表（与candidates对应）
        output_path: 输出图片路径，None则不保存

    Returns:
        标注后的图像（新图像，不修改原图）
    """
    # 创建原图的副本，不修改原图
    result_img = image.copy()

    # 字体设置
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2

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
        center_y = int(rect[0][1])

        # 获取边界框的最高点
        top_y = int(np.min(box[:, 1]))

        # 文本背景矩形
        text_size = cv2.getTextSize(filtered_string, font, font_scale, thickness)[0]
        text_x = center_x - text_size[0] // 2
        text_y = top_y - 10

        # 确保文本不超出图像边界
        if text_y < text_size[1] + 10:
            text_y = top_y + int(rect[1][1]) + text_size[1] + 10  # 放到车牌下方

        # 绘制半透明背景
        overlay = result_img.copy()
        bg_x1 = max(0, text_x - 5)
        bg_y1 = max(0, text_y - text_size[1] - 5)
        bg_x2 = min(result_img.shape[1], text_x + text_size[0] + 5)
        bg_y2 = min(result_img.shape[0], text_y + 5)
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
        result_img = cv2.addWeighted(overlay, 0.6, result_img, 0.4, 0)

        # 绘制识别结果文本（白色）
        cv2.putText(result_img, filtered_string, (text_x, text_y),
                   font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # 保存结果
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, result_img)
        print(f"\n最终结果已保存: {output_path}")

    return result_img


def create_result_summary(image, candidates, plate_strings, recognition_details,
                         output_path=None):
    """
    创建详细的结果展示图（包含原图、车牌区域、识别详情）
    只显示识别成功的字符（过滤掉置信度不够的问号）

    Args:
        image: 原始BGR图像
        candidates: 车牌候选区域列表
        plate_strings: 车牌识别结果列表
        recognition_details: 识别详情列表 [[(char, confidence, bbox), ...], ...]
        output_path: 输出图片路径

    Returns:
        拼接后的图像
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
    y_offset = img_h + 30
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i, (plate_string, details) in enumerate(zip(plate_strings, recognition_details)):
        # 过滤掉置信度不够的字符（问号）
        filtered_string = plate_string.replace('?', '')

        # 车牌序号
        cv2.putText(canvas, f"Plate {i+1}:", (20, y_offset),
                   font, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

        # 识别结果（只显示有效字符）
        if filtered_string:
            cv2.putText(canvas, filtered_string, (150, y_offset),
                       font, 1.2, (0, 128, 0), 2, cv2.LINE_AA)

            # 详细置信度信息（只显示成功识别的字符）
            detail_parts = []
            for char, confidence, _ in details:
                if char and char != '?' and confidence >= 0.5:
                    detail_parts.append(f"{char}({confidence:.2f})")

            if detail_parts:
                detail_text = " ".join(detail_parts)
                cv2.putText(canvas, detail_text, (150, y_offset + 30),
                           font, 0.5, (100, 100, 100), 1, cv2.LINE_AA)
        else:
            cv2.putText(canvas, "No valid characters", (150, y_offset),
                       font, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        y_offset += 90

    # 保存
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, canvas)
        print(f"结果已保存: {output_path}")

    return canvas


if __name__ == "__main__":
    # 测试代码
    print("请使用 test_pipeline.py 运行完整流程")