# -*- coding: utf-8 -*-
"""
车牌识别流程测试脚本
测试完整流程：预处理 -> 车牌定位 -> 字符分割

用法: python test_pipeline.py [图片路径] [--show]
      --show: 显示处理结果窗口（默认关闭）
"""

import sys
import os
import cv2
import numpy as np
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

# 添加src目录到路径
sys.path.insert(0, os.path.dirname(__file__))

from image_process import ImagePreprocessor, preprocess_image, PlateLocator, locate_plates
from char_segment import segment_characters
from match_ocr import recognize_characters
from visualize_result import create_result_summary
from config import DEBUG_DIR, RAW_IMAGES_DIR, RESULT_DIR, PARALLEL_WORKERS


def display_result(original, plate_regions, window_name="车牌识别结果"):
    """
    显示原图和定位到的车牌区域

    Args:
        original: 原始BGR图像
        plate_regions: 车牌区域图像列表
        window_name: 窗口名称
    """
    # 缩放原图以适应屏幕
    max_height = 600
    h, w = original.shape[:2]
    if h > max_height:
        scale = max_height / h
        display_img = cv2.resize(original, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        display_img = original.copy()

    cv2.imshow(window_name, display_img)

    # 显示每个车牌区域
    for i, plate in enumerate(plate_regions):
        if plate.size > 0:
            cv2.imshow(f"车牌 {i+1}", plate)

    print("按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_single_image(image_path, show_result=False):
    """
    测试单张图片的完整处理流程

    Args:
        image_path: 图片路径
        show_result: 是否显示结果窗口（默认关闭）

    Returns:
        (成功标志, 识别到的车牌数量)
    """
    basename = os.path.splitext(os.path.basename(image_path))[0]
    output_dir = os.path.join(DEBUG_DIR, basename)

    print(f"\n{'='*60}")
    print(f"测试图片: {image_path}")
    print(f"输出目录: {output_dir}")
    print('='*60)

    # ==================== 阶段1: 预处理 ====================
    print("\n[阶段1] 图像预处理")
    print("-" * 40)

    original, processed = preprocess_image(image_path, save_debug=True)

    if original is None:
        print("  失败：无法读取图片")
        return False, 0

    print(f"  原始图像尺寸: {original.shape}")
    print(f"  预处理后尺寸: {processed.shape}")

    # ==================== 阶段2: 车牌定位 ====================
    print("\n[阶段2] 车牌定位")
    print("-" * 40)

    candidates, locator = locate_plates(
        original,
        save_debug=True,
        prefix=basename
    )

    print(f"  找到 {len(candidates)} 个候选车牌区域")

    # 提取并保存车牌区域
    plate_regions = []
    for i, (rect, box, score) in enumerate(candidates):
        # 计算真实角度（长边与水平方向的夹角）
        edge1 = np.linalg.norm(box[0] - box[1])
        edge2 = np.linalg.norm(box[1] - box[2])
        if edge1 >= edge2:
            long_edge_vec = box[1] - box[0]
        else:
            long_edge_vec = box[2] - box[1]
        real_angle = np.degrees(np.arctan2(long_edge_vec[1], long_edge_vec[0]))

        print(f"  候选 {i+1}: 分数={score:.3f}, 中心={rect[0]}, 尺寸={rect[1]}, 角度={real_angle:.1f}°")

        # 提取车牌区域
        plate_img = locator.extract_plate_region(original, rect, box)
        if plate_img.size > 0:
            plate_regions.append(plate_img)
            # 保存车牌区域图片
            os.makedirs(output_dir, exist_ok=True)
            plate_path = os.path.join(output_dir, f"20_plate_{i+1}.jpg")
            cv2.imwrite(plate_path, plate_img)
            print(f"    已保存: {plate_path}")

    # ==================== 阶段3: 字符分割 ====================
    print("\n[阶段3] 字符分割")
    print("-" * 40)

    all_characters = []
    for i, plate_img in enumerate(plate_regions):
        print(f"  处理车牌 {i+1}:")
        characters, _ = segment_characters(
            plate_img,
            save_debug=True,
            output_dir=DEBUG_DIR,
            prefix=basename,
            plate_index=i+1
        )
        print(f"    分割出 {len(characters)} 个字符")

        # 保存单个字符
        for j, (char_img, _) in enumerate(characters):
            char_path = os.path.join(output_dir, f"30_char_{i+1}_{j+1}.jpg")
            cv2.imwrite(char_path, char_img)

        all_characters.append(characters)

    # ==================== 阶段4: 字符识别 ====================
    print("\n[阶段4] 字符识别")
    print("-" * 40)

    plate_strings = []
    all_recognition_details = []
    for i, characters in enumerate(all_characters):
        if len(characters) == 0:
            print(f"  车牌 {i+1}: 无字符可识别")
            plate_strings.append("")
            all_recognition_details.append([])
            continue

        print(f"  车牌 {i+1}:")
        plate_string, results, _ = recognize_characters(
            characters,
            save_debug=True,
            output_dir=DEBUG_DIR,
            prefix=basename
        )

        # 输出每个字符的识别结果
        for j, (char, confidence, _) in enumerate(results):
            status = char if char else "?"
            print(f"    字符 {j+1}: {status} (置信度: {confidence:.3f})")

        print(f"    识别结果: {plate_string}")
        plate_strings.append(plate_string)
        all_recognition_details.append(results)

    # ==================== 阶段5: 生成最终结果 ====================
    print("\n[阶段5] 生成最终结果")
    print("-" * 40)

    # 只生成一个结果图片（包含标注的原图和识别详情）
    result_path = os.path.join(RESULT_DIR, f"{basename}_result.jpg")
    result_img = create_result_summary(
        original, candidates, plate_strings,
        all_recognition_details, result_path
    )

    # 打印最终识别结果
    print("\n最终识别结果:")
    for i, plate_string in enumerate(plate_strings):
        filtered = plate_string.replace('?', '')
        if filtered:
            print(f"  车牌 {i+1}: {filtered}")
        else:
            print(f"  车牌 {i+1}: 无有效字符")

    # ==================== 结果展示 ====================
    if show_result and len(plate_regions) > 0:
        # 显示最终结果
        cv2.imshow("识别结果", result_img)

        print("\n按任意键关闭窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print(f"\n处理完成! ")
    print(f"  调试图片: {output_dir}")
    print(f"  最终结果: {result_path}")
    return True, len(candidates)


def _test_single_image_wrapper(args):
    """
    包装函数，用于多进程调用
    """
    img_path, show_result = args
    try:
        return test_single_image(img_path, show_result=show_result)
    except Exception as e:
        print(f"处理失败 {img_path}: {e}")
        return False, 0


def test_all_images_in_dir(dir_path=RAW_IMAGES_DIR, show_result=False):
    """
    测试目录下所有图片（并行处理）

    Args:
        dir_path: 图片目录路径
        show_result: 是否显示结果窗口（默认关闭）
    """
    if not os.path.exists(dir_path):
        print(f"目录不存在: {dir_path}")
        return

    # 支持的图片格式
    extensions = ('.jpg', '.jpeg', '.png', '.bmp')

    # 递归查找所有图片
    all_images = []
    for root, dirs, files in os.walk(dir_path):
        for f in files:
            if f.lower().endswith(extensions):
                all_images.append(os.path.join(root, f))

    if not all_images:
        print(f"目录中没有找到图片: {dir_path}")
        return

    print(f"找到 {len(all_images)} 张图片")
    print(f"并行进程数: {PARALLEL_WORKERS}")
    print("=" * 60)

    # 统计结果
    success_count = 0
    total_plates = 0

    # 并行处理
    task_args = [(img_path, show_result) for img_path in all_images]
    with ProcessPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
        futures = {executor.submit(_test_single_image_wrapper, args): args[0] for args in task_args}
        for future in as_completed(futures):
            img_path = futures[future]
            try:
                success, plate_count = future.result()
                if success:
                    success_count += 1
                    total_plates += plate_count
            except Exception as e:
                print(f"处理异常 {img_path}: {e}")

    # 输出统计
    print("\n" + "=" * 60)
    print("测试统计")
    print("=" * 60)
    print(f"  处理图片数: {success_count}/{len(all_images)}")
    print(f"  定位车牌数: {total_plates}")
    print(f"  调试输出目录: {DEBUG_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="车牌识别流程测试")
    parser.add_argument("path", nargs="?", default=None,
                        help="图片路径或目录路径")
    parser.add_argument("--show", action="store_true",
                        help="显示处理结果窗口（默认关闭）")

    args = parser.parse_args()

    if args.path:
        # 测试指定的图片或目录
        if os.path.isdir(args.path):
            test_all_images_in_dir(args.path, show_result=args.show)
        else:
            test_single_image(args.path, show_result=args.show)
    else:
        # 默认处理 raw_images 目录下的所有图片
        print("未指定路径，默认处理 resources/raw_images/ 目录")
        test_all_images_in_dir(RAW_IMAGES_DIR, show_result=args.show)
