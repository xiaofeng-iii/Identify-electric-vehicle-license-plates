# -*- coding: utf-8 -*-
"""
车牌识别主程序
用户将图片放入 resources/raw_images/，运行本程序即可批量识别

用法:
    python src/main.py                      # 简洁模式
    python src/main.py --verbose            # 详细输出，不保存调试图
    python src/main.py --debug              # 保存调试图，简洁输出
    python src/main.py --debug --verbose    # 保存调试图 + 详细输出
"""

import sys
import os
import cv2
import argparse
from contextlib import contextmanager

sys.path.insert(0, os.path.dirname(__file__))

from config import RAW_IMAGES_DIR, RESULT_DIR, DEBUG_DIR


@contextmanager
def suppress_stdout():
    """上下文管理器：临时抑制 stdout 输出"""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def scan_images(directory):
    """扫描目录下所有图片文件"""
    extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    images = []
    for root, dirs, files in os.walk(directory):
        for f in files:
            if f.lower().endswith(extensions):
                images.append(os.path.join(root, f))
    return sorted(images)


def process_single_image(image_path, enable_debug, verbose):
    """
    处理单张图片

    Args:
        image_path: 图片路径
        enable_debug: 是否保存调试图片（强制设置DEBUG_MODE）
        verbose: 是否显示详细输出

    Returns:
        (plate_strings, candidate_count)
    """
    # 强制设置 DEBUG_MODE，命令行参数完全控制
    import config
    config.DEBUG_MODE = enable_debug

    from image_process import preprocess_image, locate_plates
    from char_segment import segment_characters
    from match_ocr import recognize_characters
    from visualize_result import create_result_summary

    basename = os.path.splitext(os.path.basename(image_path))[0]

    if verbose:
        print("-" * 60)

    # 阶段1: 预处理
    if verbose:
        print("\n[阶段1] 图像预处理")

    with suppress_stdout() if not verbose else open(os.devnull):
        original, processed = preprocess_image(image_path, save_debug=enable_debug)

    if original is None:
        raise ValueError("无法读取图片")
    if verbose:
        print(f"  原始图像尺寸: {original.shape}")
        print(f"  预处理后尺寸: {processed.shape}")

    # 阶段2: 车牌定位
    if verbose:
        print("\n[阶段2] 车牌定位")

    with suppress_stdout() if not verbose else open(os.devnull):
        candidates, locator = locate_plates(
            original, method='combined',
            save_debug=enable_debug, prefix=basename
        )

    if verbose:
        print(f"  找到 {len(candidates)} 个候选车牌区域")

    # 提取车牌区域
    plate_regions = []
    for i, (rect, box, score) in enumerate(candidates):
        if verbose:
            print(f"  候选 {i+1}: 分数={score:.3f}, 中心={rect[0]}, "
                  f"尺寸={rect[1]}, 角度={rect[2]:.1f}°")

        plate_img = locator.extract_plate_region(original, rect, box)
        if plate_img.size > 0:
            plate_regions.append(plate_img)
            if enable_debug:
                output_dir = os.path.join(DEBUG_DIR, basename)
                os.makedirs(output_dir, exist_ok=True)
                plate_path = os.path.join(output_dir, f"20_plate_{i+1}.jpg")
                cv2.imwrite(plate_path, plate_img)
                if verbose:
                    print(f"    已保存: {plate_path}")

    # 阶段3: 字符分割
    if verbose:
        print("\n[阶段3] 字符分割")
    all_characters = []
    for i, plate_img in enumerate(plate_regions):
        if verbose:
            print(f"  处理车牌 {i+1}:")

        with suppress_stdout() if not verbose else open(os.devnull):
            characters, _ = segment_characters(
                plate_img, save_debug=enable_debug,
                output_dir=DEBUG_DIR, prefix=basename,
                plate_index=i+1
            )

        if verbose:
            print(f"    分割出 {len(characters)} 个字符")
        all_characters.append(characters)

    # 阶段4: 字符识别
    if verbose:
        print("\n[阶段4] 字符识别")
    plate_strings = []
    all_recognition_details = []
    for i, characters in enumerate(all_characters):
        if len(characters) == 0:
            if verbose:
                print(f"  车牌 {i+1}: 无字符可识别")
            plate_strings.append("")
            all_recognition_details.append([])
            continue

        if verbose:
            print(f"  车牌 {i+1}:")

        with suppress_stdout() if not verbose else open(os.devnull):
            plate_string, results, _ = recognize_characters(
                characters, save_debug=enable_debug,
                output_dir=DEBUG_DIR, prefix=basename
            )

        if verbose:
            for j, (char, confidence, _) in enumerate(results):
                status = char if char else "?"
                print(f"    字符 {j+1}: {status} (置信度: {confidence:.3f})")
            print(f"    识别结果: {plate_string}")

        plate_strings.append(plate_string)
        all_recognition_details.append(results)

    # 阶段5: 生成结果
    if verbose:
        print("\n[阶段5] 生成最终结果")

    result_path = os.path.join(RESULT_DIR, f"{basename}_result.jpg")

    with suppress_stdout() if not verbose else open(os.devnull):
        create_result_summary(
            original, candidates, plate_strings,
            all_recognition_details, result_path
        )

    if verbose:
        print(f"  结果已保存: {result_path}")
        print("\n最终识别结果:")
        for i, plate_string in enumerate(plate_strings):
            filtered = plate_string.replace('?', '')
            if filtered:
                print(f"  车牌 {i+1}: {filtered}")
        print("-" * 60)

    return plate_strings, len(candidates)


def main():
    parser = argparse.ArgumentParser(
        description="车牌识别系统 - 批量识别",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python src/main.py                      # 简洁模式
  python src/main.py --verbose            # 详细输出
  python src/main.py --debug              # 保存调试图
  python src/main.py --debug --verbose    # 全开
        """
    )
    parser.add_argument("--debug", action="store_true",
                       help="开启调试模式（保存中间处理图片到 debug_steps/）")
    parser.add_argument("--verbose", action="store_true",
                       help="显示详细处理流程（5个阶段的详细输出）")
    parser.add_argument("--show", action="store_true",
                       help="显示识别结果窗口（暂未实现）")

    args = parser.parse_args()

    # 扫描图片
    images = scan_images(RAW_IMAGES_DIR)

    if not images:
        print(f"未找到图片: {RAW_IMAGES_DIR}")
        print("请将图片放入 resources/raw_images/ 目录")
        return

    print(f"找到 {len(images)} 张图片")
    print("=" * 50)

    # 批量处理
    success_count = 0
    total_plates = 0

    for i, img_path in enumerate(images, 1):
        basename = os.path.basename(img_path)

        if not args.verbose:
            # 简洁模式：只显示进度
            print(f"\n[{i}/{len(images)}] {basename}")
        else:
            # 详细模式：显示完整路径
            print(f"\n[{i}/{len(images)}] {img_path}")

        try:
            plates, count = process_single_image(
                img_path, args.debug, args.verbose
            )
            success_count += 1
            total_plates += count

            # 简洁模式：显示识别结果
            if not args.verbose:
                for plate in plates:
                    filtered = plate.replace('?', '')
                    if filtered:
                        print(f"  ✓ {filtered}")
                print(f"  结果: output/results/{os.path.splitext(basename)[0]}_result.jpg")

        except Exception as e:
            print(f"  ✗ 处理失败: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()

    # 统计摘要
    print("\n" + "=" * 50)
    print(f"处理完成: {success_count}/{len(images)} 张")
    print(f"识别车牌: {total_plates} 个")
    print(f"结果目录: output/results/")
    if args.debug:
        print(f"调试图片: output/debug_steps/")


if __name__ == "__main__":
    main()
