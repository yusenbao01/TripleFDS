"""
Generating data for SRNet
Copyright (c) 2019 Netease Youdao Information Technology Co.,Ltd.
Licensed under the GPL License (see LICENSE for details)
Written by Yu Qian
"""

import os
import cv2
import cfg
import numpy as np
import json
from tqdm import tqdm
from Synthtext.gen import datagen, multiprocess_datagen
import shutil


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def parse_data_mode(mode_str):
    """Parse data mode string and return dimension list"""
    if mode_str == "8":
        return [2, 2, 2]
    elif mode_str == "27":
        return [3, 3, 3]
    elif mode_str == "64":
        return [4, 4, 4]
    elif "x" in mode_str:
        try:
            dimensions = [int(dim) for dim in mode_str.split("x")]
            if len(dimensions) == 3:
                return dimensions
        except ValueError:
            pass
    # Default to 2x2x2
    print(f"Warning: Invalid data mode '{mode_str}', using default 2x2x2")
    return [2, 2, 2]


def main():
    # Build complete directory path
    data_dir = cfg.data_dir
    # Create data directory
    makedirs(data_dir)

    # Get the parent directory of data_dir
    source_file_path = os.path.join('.', 'Synthtext', 'data_cfg.py')
    parent_of_data_dir = os.path.dirname(data_dir)
    destination_file_path = os.path.join(parent_of_data_dir, 'data_cfg.py')
    # Copy file
    shutil.copyfile(source_file_path, destination_file_path)
    print(f"File '{source_file_path}' has been successfully copied to '{destination_file_path}'")

    # Parse data mode
    dimensions = parse_data_mode(cfg.data_mode)
    font_num, text_num, bg_num = dimensions
    total_combinations = font_num * text_num * bg_num

    print(f"Data generation mode: {font_num}x{text_num}x{bg_num} = {total_combinations} combinations")

    # Initialize multiprocess data generator
    mp_gen = multiprocess_datagen(cfg.process_num, cfg.data_capacity, cfg.data_mode)
    mp_gen.multiprocess_runningqueue()

    # Calculate number of digits needed for filenames
    digit_num = len(str(cfg.sample_num))

    # Generate specified number of samples
    for idx in tqdm(range(cfg.sample_num)):
        # Get a generated data sample from the queue
        data_dict = mp_gen.dequeue_data()

        # Unpack the data dictionary to get images and text information
        img_list = data_dict["data"]
        mask_list = data_dict.get("mask", [])
        glyph_list = data_dict.get("glyph", [])
        bg_list = data_dict.get("background", [])
        style_list = data_dict.get("style", [])
        content_list = data_dict.get("content", [])
        lang = data_dict.get("language", [])
        cropped_bbox_list = data_dict.get("bbox", [])
        # Check if the number of images matches expectations
        if len(img_list) != total_combinations:
            print(f"Warning: Image count ({len(img_list)}) doesn't match expected ({total_combinations})")
            continue

        # Create subdirectory for each sample
        sample_dir = os.path.join(data_dir, str(idx).zfill(digit_num))
        makedirs(sample_dir)

        # Save all combination images
        for i in range(total_combinations):
            img_path = os.path.join(sample_dir, f"{i}.png")
            cv2.imwrite(img_path, img_list[i], [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

        # Save all combination images
        for i, mask in enumerate(mask_list):
            img_path = os.path.join(sample_dir, f"mask{i}.png")
            cv2.imwrite(img_path, mask, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

        # Save all glyph images
        for i, glyph in enumerate(glyph_list):
            glyph_path = os.path.join(sample_dir, f"glyph{i}.png")
            cv2.imwrite(glyph_path, glyph, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

        # Save all background images
        for i, bg in enumerate(bg_list):
            bg_path = os.path.join(sample_dir, f"bg{i}.png")
            cv2.imwrite(bg_path, bg, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

        # remove the prefix of fonts
        style_list = [os.path.basename(font) for font in style_list]

        # Create and write information file (JSON format)
        info_data = {
            "dimensions": dimensions,
            "fonts": style_list[:-1],
            "texts": content_list,
            "standard_font": style_list[-1],
            "language": lang,
            "cropped_bbox": cropped_bbox_list,
        }

        # Write to JSON file, using ensure_ascii=False to correctly display Chinese characters
        with open(os.path.join(sample_dir, "info.json"), "w", encoding="utf-8") as f:
            json.dump(info_data, f, indent=2, ensure_ascii=False)

    # Clean up resources
    mp_gen.terminate_pool()


if __name__ == "__main__":
    main()
