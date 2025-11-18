# -*- coding: utf-8 -*-
"""
SRNet data generator.
Copyright (c) 2019 Netease Youdao Information Technology Co.,Ltd.
Licensed under the GPL License
Written by Yu Qian
"""

import os
import cv2
import math
import numpy as np
import pygame
from pygame import freetype
import random
import multiprocessing
import queue
import Augmentor
import heapq

from . import render_text_mask
from . import colorize
from . import skeletonization
from . import render_standard_text
from . import data_cfg
import pickle as cp

import sys
import cfg


def generate_realistic_text_color():
    """
    Generates a color based on a probability distribution closer to real-world text colors.
    """
    # 50% blackish, 15% whitish, 15% dark colors, 20% completely random
    choice = np.random.choice(['black', 'white', 'dark', 'random'], p=[0.50, 0.15, 0.15, 0.20])

    if choice == 'black':
        # Random in a range close to pure black, simulating minor variations from scanning, compression, etc.
        return np.random.randint(0, 40, size=3).astype(np.uint8)
    
    elif choice == 'white':
        # Random in a range close to pure white
        return np.random.randint(215, 256, size=3).astype(np.uint8)
        
    elif choice == 'dark':
        # Generate a random dark color, excluding those close to pure black
        return np.random.randint(40, 128, size=3).astype(np.uint8)
        
    else: # choice == 'random'
        # Generate a completely random color to cover all other possibilities
        return (np.random.rand(3) * 255.0).astype(np.uint8)


class datagen:

    def __init__(self):
        freetype.init()
        self.load_resources()
        color_filepath = data_cfg.color_filepath
        self.colorsRGB, self.colorsLAB = colorize.get_color_matrix(color_filepath)
        self.bg_list = []

        bg_filepath = cfg.bg_folder
        if not bg_filepath:
            print("Warning: Background image folder not specified, using an empty list.")
            self.bg_list = []
        else:

            def collect_images(directory):
                if not os.path.exists(directory):
                    print(f"Warning: Background image directory does not exist: {directory}")
                    return

                for item in os.listdir(directory):
                    item_path = os.path.join(directory, item)
                    if os.path.isfile(item_path):
                        self.bg_list.append(item_path)
                    elif os.path.isdir(item_path):
                        collect_images(item_path)

            collect_images(bg_filepath)
            print(f"Loaded {len(self.bg_list)} background images.")

        self.surf_augmentor = Augmentor.DataPipeline(None)
        self.surf_augmentor.random_distortion(
            probability=data_cfg.elastic_rate,
            grid_width=data_cfg.elastic_grid_size,
            grid_height=data_cfg.elastic_grid_size,
            magnitude=data_cfg.elastic_magnitude,
        )

        self.bg_augmentor = Augmentor.DataPipeline(None)
        self.bg_augmentor.random_brightness(
            probability=data_cfg.brightness_rate, min_factor=data_cfg.brightness_min, max_factor=data_cfg.brightness_max
        )
        self.bg_augmentor.random_color(
            probability=data_cfg.color_rate, min_factor=data_cfg.color_min, max_factor=data_cfg.color_max
        )
        self.bg_augmentor.random_contrast(
            probability=data_cfg.contrast_rate, min_factor=data_cfg.contrast_min, max_factor=data_cfg.contrast_max
        )

        self.failed_bg_list = []
        self.min_bg_list_size = data_cfg.min_bg_list_size
        self.restored_bg_num = data_cfg.restored_bg_num

    def load_resources(self):
        """
        Loads resource files (fonts and texts) for each language based on cfg.py, supporting multiple languages.
        """
        self.language_resources = {}
        self.language_list = list(cfg.lang_prob.keys())
        self.language_prob = list(cfg.lang_prob.values())

        for lang in self.language_list:
            font_dir = os.path.join(cfg.font_folder, lang)
            standard_font_dir = os.path.join(cfg.font_folder, lang, "standard")
            text_filepath = os.path.join(cfg.text_folder, f"texts_{lang}.txt")

            standard_font_path = None
            if os.path.exists(standard_font_dir) and os.path.isdir(standard_font_dir):
                standard_fonts = os.listdir(standard_font_dir)
                if standard_fonts:
                    standard_font_path = os.path.join(standard_font_dir, standard_fonts[0])
                    print(f"Found standard font for {lang}: {standard_font_path}")

            self.language_resources[lang] = {"font_list": [], "standard_font_path": standard_font_path, "text_list": []}

            if os.path.exists(font_dir) and os.path.isdir(font_dir):
                def load_fonts_recursively(directory, lang_resources):
                    for item in os.listdir(directory):
                        item_path = os.path.join(directory, item)
                        if os.path.isfile(item_path):
                            lang_resources["font_list"].append(item_path)
                        elif os.path.isdir(item_path) and item != "standard":
                            load_fonts_recursively(item_path, lang_resources)

                load_fonts_recursively(font_dir, self.language_resources[lang])
                print(f"Loaded {len(self.language_resources[lang]['font_list'])} fonts for {lang}.")

            try:
                if os.path.exists(text_filepath):
                    with open(text_filepath, "r", encoding="utf-8") as f:
                        self.language_resources[lang]["text_list"] = [
                            line.strip() for line in f.readlines() if line.strip()
                        ]
                    print(f"Loaded {len(self.language_resources[lang]['text_list'])} text lines for {lang}.")
                else:
                    print(f"Text file for {lang} not found: {text_filepath}")
            except Exception as e:
                print(f"Failed to load text for {lang}: {e}")

        self.font_objects = {}
        all_fonts = []

        for lang_resources in self.language_resources.values():
            all_fonts.extend(lang_resources["font_list"])
            all_fonts.append(lang_resources["standard_font_path"])

        all_fonts = list(set(all_fonts))

        for font_path in all_fonts:
            try:
                font = freetype.Font(font_path)
                font.antialiased = True
                font.origin = True
                self.font_objects[font_path] = font
            except Exception as e:
                if not os.path.isdir(font_path):
                    print(f"Could not load font {font_path}: {e}")

        print(f"Successfully loaded {len(self.font_objects)} fonts.")

    def _set_current_language_resources(self, lang):
        self.current_lang = lang
        self.font_list = self.language_resources[lang]["font_list"]
        self.standard_font_path = self.language_resources[lang]["standard_font_path"]
        self.text_list = self.language_resources[lang]["text_list"]

    def select_language(self):
        """
        Selects a language based on probabilities set in cfg.py and sets the corresponding resources.
        Returns the selected language type, e.g., "en".
        """
        selected_lang = random.choices(self.language_list, weights=self.language_prob, k=1)[0]
        self._set_current_language_resources(selected_lang)
        return selected_lang

    def render_text_with_effects(self, font, text, params, bg_params, apply_curve_shrink=False, longest_surf_w=None):
        """
        General text rendering function, including setting curve parameters, rendering text, and applying perspective transformations.
        """
        params["curve_center"] = len(text) // 2
        param = {
            "is_curve": params["is_curve"],
            "curve_rate": params["curve_rate"],
            "curve_center": params["curve_center"],
            "letter_spacing": params["letter_spacing"],
        }

        font.underline = params["underline"]
        font.strong = params["strong"]
        font.oblique = params["oblique"]

        surf, bbs = render_text_mask.render_text(font, text, param)
        min_h = np.min(bbs[:, 3]) if bbs.size > 0 else None
        padding = np.hstack((bg_params["padding_ud"], bg_params["padding_lr"]))

        surf = render_text_mask.perspective(
            surf, params["rotate"], params["zoom"], params["shear"], params["perspect"], padding
        )

        if longest_surf_w != None and abs(longest_surf_w - surf.shape[1]) > 5:
            new_space = int(abs(longest_surf_w - surf.shape[1]) / (len(text) + 1))
            params["letter_spacing"] += new_space - 1
            surf, bbs, min_h = self.render_text_with_effects(font, text, params, bg_params, apply_curve_shrink=True)

        return surf, bbs, min_h

    def gen_Contrastive_data(self, font_num=2, text_num=2, bg_num=2):
        """
        Generates a contrastive learning dataset, supporting any number of font, text, and background combinations.
        """
        lang = self.select_language()
        total_combinations = font_num * text_num * bg_num

        fixed_size = data_cfg.fixed_size
        min_font_size = data_cfg.font_size[0]
        max_font_size = data_cfg.font_size[1]
        min_area_requirement = data_cfg.min_area_requirement

        while True:
            try:
                texts = np.random.choice(self.text_list, text_num, replace=False)
                fonts = np.random.choice(self.font_list, font_num, replace=False)
            except ValueError:
                print(f"Warning: Not enough texts ({len(self.text_list)}) or fonts ({len(self.font_list)}). Allowing repeats.")
                texts = np.random.choice(self.text_list, text_num, replace=True)
                fonts = np.random.choice(self.font_list, font_num, replace=True)

            max_len_diff = data_cfg.max_len_diff
            min_text_len = data_cfg.min_text_len
            text_lens_raw = [len(text) for text in texts]
            text_lens_no_space = [len(text.replace(" ", "")) for text in texts]
            if max(text_lens_raw) - min(text_lens_raw) > max_len_diff or min(text_lens_no_space) < min_text_len:
                continue

            self._process_text_by_language(texts, lang)

            try:
                bg_indices = np.random.choice(len(self.bg_list), bg_num, replace=False)
            except ValueError:
                print(f"Warning: Not enough backgrounds ({len(self.bg_list)}).")

            bg_paths = [self.bg_list[i] for i in bg_indices]
            bgs_original = [cv2.imread(bg_path) for bg_path in bg_paths]

            bgs = []
            for i, bg in enumerate(bgs_original):
                if bg is None: continue
                bg_h, bg_w = bg.shape[:2]
                if bg_h < fixed_size[0] or bg_w < fixed_size[1]:
                    continue

                if bg_h > fixed_size[0] and bg_w > fixed_size[1]:
                    x = np.random.randint(0, bg_w - fixed_size[1] + 1)
                    y = np.random.randint(0, bg_h - fixed_size[0] + 1)
                    t_b = bg[y : y + fixed_size[0], x : x + fixed_size[1], :]
                else:
                    t_b = cv2.resize(bg, fixed_size)
                bgs.append(t_b)

            if len(bgs) < bg_num:
                continue

            all_fonts_loaded = all(font in self.font_objects for font in fonts)
            if not all_fonts_loaded:
                continue

            bg_params = {
                "padding_ud": np.random.randint(data_cfg.padding_ud[0], data_cfg.padding_ud[1] + 1, 2),
                "padding_lr": np.random.randint(data_cfg.padding_lr[0], data_cfg.padding_lr[1] + 1, 2),
            }

            font_params = []
            for i in range(font_num + 1):
                param = {
                    "size": None,
                    "underline": np.random.rand() < data_cfg.underline_rate,
                    "strong": np.random.rand() < data_cfg.strong_rate,
                    "oblique": np.random.rand() < data_cfg.oblique_rate,
                    "is_curve": np.random.rand() < data_cfg.is_curve_rate,
                    "curve_rate": data_cfg.curve_rate_param[0] * np.random.randn() + data_cfg.curve_rate_param[1],
                    "curve_center": None,
                    "rotate": data_cfg.rotate_param[0] * np.random.randn() + data_cfg.rotate_param[1],
                    "zoom": data_cfg.zoom_param[0] * np.random.randn(2) + data_cfg.zoom_param[1],
                    "shear": data_cfg.shear_param[0] * np.random.randn(2) + data_cfg.shear_param[1],
                    "perspect": data_cfg.perspect_param[0] * np.random.randn(2) + data_cfg.perspect_param[1],
                    "use_random_color": np.random.rand() < data_cfg.use_random_color_rate,
                    "random_fg_color": generate_realistic_text_color(),
                    "is_border": np.random.rand() < data_cfg.is_border_rate,
                    "border_color": tuple(np.random.randint(0, 256, 3)),
                    "is_shadow": np.random.rand() < data_cfg.is_shadow_rate,
                    "shadow_angle": np.pi / 4 * np.random.choice(data_cfg.shadow_angle_degree)
                    + data_cfg.shadow_angle_param[0] * np.random.randn(),
                    "shadow_shift": data_cfg.shadow_shift_param[0, :] * np.random.randn(3)
                    + data_cfg.shadow_shift_param[1, :],
                    "shadow_opacity": data_cfg.shadow_opacity_param[0] * np.random.randn()
                    + data_cfg.shadow_opacity_param[1],
                    "letter_spacing": np.random.randint(
                        data_cfg.min_letter_spacing, data_cfg.max_letter_spacing + 1
                    ),
                }
                if cfg.difficulty == "easy": temp = 0
                elif cfg.difficulty == "hard": temp = 1
                elif cfg.difficulty == "middle": temp = random.randint(0, 1)
                param["difficulty"] = temp

                if not param["use_random_color"]:
                    param["random_fg_color"], _ = colorize.get_font_color(
                        self.colorsRGB, self.colorsLAB, bgs[i % bg_num]
                    )
                font_params.append(param)
            
            font_params[-1] = { # Standard font
                "size": None, "underline": False, "strong": False, "oblique": False,
                "is_curve": False, "curve_rate": 0, "curve_center": None, "rotate": 0,
                "zoom": (1, 1), "shear": (0, 0), "perspect": (0, 0), "use_random_color": False,
                "random_fg_color": (0, 0, 0), "is_border": False, "border_color": (0, 0, 0),
                "is_shadow": False, "shadow_angle": 0, "shadow_shift": (0, 0, 0),
                "shadow_opacity": 0, "letter_spacing": 2
            }

            font_text_combinations = []
            fonts = np.append(fonts, self.standard_font_path)

            for font_idx, font in enumerate(fonts):
                for text in texts:
                    font_text_combinations.append((font, text, font_idx))

            font_sizes = {}
            font_min = {}
            optimal_sizes_found = True
            buffer = data_cfg.buffer

            for font_path, text, font_idx in font_text_combinations:
                font = self.font_objects[font_path]
                params = font_params[font_idx]

                low, high = min_font_size, max_font_size
                best_size = low

                while low < high:
                    mid = (low + high) // 2
                    font.size = mid
                    surf, _, _ = self.render_text_with_effects(font, text, params, bg_params)
                    surf_h, surf_w = surf.shape[:2]

                    if surf_h <= fixed_size[0] - buffer and surf_w <= fixed_size[1] - buffer:
                        best_size = mid
                        low = mid + 1
                    else:
                        high = mid - 1

                if best_size == min_font_size:
                    font.size = best_size
                    surf, _, _ = self.render_text_with_effects(font, text, params, bg_params)
                    surf_h, surf_w = surf.shape[:2]
                    if (
                        surf_h > fixed_size[0] - buffer
                        or surf_w > fixed_size[1] - buffer
                        or surf_h * surf_w < min_area_requirement
                    ):
                        optimal_sizes_found = False
                        break

                if font_min.get(font_path, [1000, 0])[1] < surf_w:
                    font_min[font_path] = [font_min.get(font_path, [1000, 0])[0], surf_w]
                if font_min.get(font_path, (1000, 0))[0] > best_size:
                    font_min[font_path] = [best_size, font_min.get(font_path, (1000, 0))[1]]

            for font in fonts:
                for text in texts:
                    font_sizes[(font, text)] = font_min[font]

            if not optimal_sizes_found:
                continue

            combinations = [(font, text, bg) for font in fonts[:-1] for text in texts for bg in bgs]
            font_to_idx = {font: idx for idx, font in enumerate(fonts)}

            img_list, mask_list, glyph_list, bg_aug_list, bg_col_list = [], [], [], [], []

            for i, bg in enumerate(bgs):
                self.bg_augmentor.augmentor_images = [[bg]]
                bg_aug = self.bg_augmentor.sample(1)[0][0]
                bg_aug_list.append(bg_aug)
                _, bg_col = colorize.get_font_color(self.colorsRGB, self.colorsLAB, bg_aug)
                bg_col_list.append(bg_col)

            color_diff1, color_diff2 = data_cfg.color_diff1, data_cfg.color_diff2
            for font_idx in range(font_num):
                attempts = 0
                max_attempts = 100
                while attempts < max_attempts:
                    contrast_ok = True
                    for bg_col in bg_col_list:
                        if np.sum(np.abs(bg_col - font_params[font_idx]["random_fg_color"])) < color_diff1:
                            contrast_ok = False; break
                    if contrast_ok and font_num > 1:
                        for other_idx in range(font_idx):
                            fg_diff = np.sum(np.abs(font_params[other_idx]["random_fg_color"] - font_params[font_idx]["random_fg_color"]))
                            if fg_diff < color_diff2:
                                contrast_ok = False; break
                    if contrast_ok: break
                    font_params[font_idx]["random_fg_color"] = generate_realistic_text_color()
                    attempts += 1
            if attempts >= max_attempts:
                continue

            for i, (font_path, text, bg) in enumerate(combinations):
                font_idx = font_to_idx[font_path]
                params = font_params[font_idx].copy()
                font = self.font_objects[font_path]
                params["size"], longest_surf_w = font_sizes[(font_path, text)]
                font.size = params["size"]

                surf, bbs, min_h = self.render_text_with_effects(
                    font, text, params, bg_params, apply_curve_shrink=False, longest_surf_w=longest_surf_w
                )

                bg_idx = i % bg_num
                bg_aug, bg_col = bg_aug_list[bg_idx], bg_col_list[bg_idx]

                surfs = [[surf]]
                self.surf_augmentor.augmentor_images = surfs
                surf_aug = self.surf_augmentor.sample(1)[0][0]
                surf_aug = render_text_mask.center2size(surf_aug, fixed_size)

                fg_col = params["random_fg_color"]
                param = {
                    "is_border": params["is_border"], "bordar_color": params["border_color"],
                    "is_shadow": params["is_shadow"], "shadow_angle": params["shadow_angle"],
                    "shadow_shift": params["shadow_shift"], "shadow_opacity": params["shadow_opacity"],
                    "difficulty": params["difficulty"],
                }
                _, img = colorize.colorize(
                    surf_aug, bg_aug, fg_col, bg_col, self.colorsRGB, self.colorsLAB, min_h, param
                )

                img_list.append(img)
                if i % bg_num == 0:
                    mask_list.append(surf_aug)

            font = self.font_objects[self.standard_font_path]
            params = font_params[-1]
            for text in texts:
                size, _ = font_sizes[(self.standard_font_path, text)]
                font.size = size
                font.kerning = True
                surf, bbs, min_h = self.render_text_with_effects(
                    font, text, params, bg_params, apply_curve_shrink=False
                )
                surf_aug = render_text_mask.center2size(surf, fixed_size)
                glyph_list.append(surf_aug)

            if len(img_list) == total_combinations and len(glyph_list) == text_num:
                font_max_sizes, max_h, max_w = {}, 0, 0
                keep_same_size_bbox = True
                
                for font_idx, mask in enumerate(mask_list):
                    font_idx = font_idx // text_num
                    if mask.sum() > 0:
                        binary_mask = (mask[:, :, 0] > 0).astype(np.uint8) if mask.ndim == 3 else (mask > 0).astype(np.uint8)
                        rows, cols = np.any(binary_mask, axis=1), np.any(binary_mask, axis=0)
                        y_min, y_max = np.where(rows)[0][[0, -1]] if np.any(rows) else (0, mask.shape[0] - 1)
                        x_min, x_max = np.where(cols)[0][[0, -1]] if np.any(cols) else (0, mask.shape[1] - 1)
                        mask_h, mask_w = y_max - y_min + 1, x_max - x_min + 1
                        
                        if font_idx not in font_max_sizes:
                            font_max_sizes[font_idx] = (mask_h, mask_w)
                        else:
                            curr_h, curr_w = font_max_sizes[font_idx]
                            font_max_sizes[font_idx] = (max(curr_h, mask_h), max(curr_w, mask_w))
                        if keep_same_size_bbox:
                            max_h, max_w = max(max_h, mask_h), max(max_w, mask_w)
                
                if keep_same_size_bbox:
                    for font_idx in font_max_sizes:
                        font_max_sizes[font_idx] = (max_h, max_w)
                
                padding = 0
                for font_idx in font_max_sizes:
                    h, w = font_max_sizes[font_idx]
                    font_max_sizes[font_idx] = (h + 2 * padding, w + 2 * padding)

                cropped_img_list, cropped_bbox_list, cropped_mask_list = [], [], []

                for i, img in enumerate(img_list):
                    font_idx = i // (text_num * bg_num)
                    if font_idx in font_max_sizes:
                        max_h, max_w = font_max_sizes[font_idx]
                        h, w = img.shape[:2]
                        start_y, start_x = max(0, (h - max_h) // 2), max(0, (w - max_w) // 2)
                        end_y, end_x = min(h, start_y + max_h), min(w, start_x + max_w)
                        cropped_img = img[start_y:end_y, start_x:end_x]
                        cropped_img_list.append(cropped_img)
                        cropped_bbox_list.append([int(start_y), int(end_y), int(start_x), int(end_x)])
                    else:
                        cropped_img_list.append(img)
                
                for mask_idx, mask in enumerate(mask_list):
                    font_idx = mask_idx // text_num
                    if font_idx in font_max_sizes:
                        max_h, max_w = font_max_sizes[font_idx]
                        h, w = mask.shape[:2]
                        start_y, start_x = max(0, (h - max_h) // 2), max(0, (w - max_w) // 2)
                        end_y, end_x = min(h, start_y + max_h), min(w, start_x + max_w)
                        cropped_mask = mask[start_y:end_y, start_x:end_x]
                        cropped_mask_list.append(cropped_mask)
                    else:
                        cropped_mask_list.append(mask)

                img_list, mask_list = cropped_img_list, cropped_mask_list
                
                if random.random() < data_cfg.p_blur:
                    try:
                        blur_sigma = random.uniform(data_cfg.min_blur_radius, data_cfg.max_blur_radius)
                        k_size = int(6 * blur_sigma + 1)
                        if k_size % 2 == 0: k_size += 1
                        kernel_size = (k_size, k_size)
                        
                        processed_imgs, processed_bg_aug_list = [], []
                        for img_np in img_list:
                            if isinstance(img_np, np.ndarray) and img_np.ndim == 3:
                                blurred_img_np = cv2.GaussianBlur(img_np, kernel_size, sigmaX=blur_sigma)
                                processed_imgs.append(blurred_img_np)
                            else:
                                processed_imgs.append(img_np)
                        
                        for bg_aug in bg_aug_list:
                            if isinstance(bg_aug, np.ndarray) and bg_aug.ndim == 3:
                                blurred_bg_aug = cv2.GaussianBlur(bg_aug, kernel_size, sigmaX=blur_sigma)
                                processed_bg_aug_list.append(blurred_bg_aug)
                            else:
                                processed_bg_aug_list.append(bg_aug)

                        img_list, bg_aug_list = processed_imgs, processed_bg_aug_list
                    except Exception as e:
                        print(f"An error occurred during blurring: {e}. Skipping.")
                
                break

        return {
            "data": img_list,
            "bbox": cropped_bbox_list,
            "mask": mask_list,
            "background": bg_aug_list,
            "style": fonts if isinstance(fonts, list) else fonts.tolist(),
            "content": texts if isinstance(texts, list) else texts.tolist(),
            "language": self.current_lang,
        }

    def _process_text_by_language(self, texts, lang):
        """
        Processes text based on language type, e.g., case conversion.
        """
        if lang == "en":
            upper_rand = np.random.rand()
            if upper_rand < data_cfg.capitalize_rate + data_cfg.uppercase_rate:
                for i in range(len(texts)):
                    texts[i] = texts[i].title()
            if upper_rand < data_cfg.uppercase_rate:
                for i in range(len(texts)):
                    texts[i] = texts[i].upper()

    def gen_Contrastive_data_8(self):
        return self.gen_Contrastive_data(font_num=2, text_num=2, bg_num=2)

    def gen_Contrastive_data_27(self):
        return self.gen_Contrastive_data(font_num=3, text_num=3, bg_num=3)

    def gen_Contrastive_data_64(self):
        return self.gen_Contrastive_data(font_num=4, text_num=4, bg_num=4)


def enqueue_data(queue, capacity, mode="8"):
    """
    Data generation thread function.
    """
    np.random.seed()
    gen = datagen()
    while True:
        try:
            if mode == "8": data_dict = gen.gen_Contrastive_data_8()
            elif mode == "27": data_dict = gen.gen_Contrastive_data_27()
            elif mode == "64": data_dict = gen.gen_Contrastive_data_64()
            elif "x" in mode:
                dimensions = [int(dim) for dim in mode.split("x")]
                if len(dimensions) == 3:
                    data_dict = gen.gen_Contrastive_data(
                        font_num=dimensions[0], text_num=dimensions[1], bg_num=dimensions[2]
                    )
                else:
                    print(f"Warning: Invalid dimension format: {mode}, using default (2x2x2).")
                    data_dict = gen.gen_Contrastive_data_8()
            else:
                data_dict = gen.gen_Contrastive_data_8()

        except Exception as e:
            import traceback
            print(f"Error in data generation: {e}")
            traceback.print_exc()
            continue

        if queue.qsize() < capacity:
            queue.put(data_dict)


class multiprocess_datagen:

    def __init__(self, process_num, data_capacity, data_mode="8"):
        self.process_num = process_num
        self.data_capacity = data_capacity
        self.data_mode = data_mode

    def multiprocess_runningqueue(self):
        manager = multiprocessing.Manager()
        self.queue = manager.Queue()
        self.pool = multiprocessing.Pool(processes=self.process_num)
        self.processes = []
        for _ in range(self.process_num):
            p = self.pool.apply_async(enqueue_data, args=(self.queue, self.data_capacity, self.data_mode))
            self.processes.append(p)
        self.pool.close()

    def dequeue_data(self):
        np.random.seed()
        while self.queue.empty():
            pass
        return self.queue.get()

    def dequeue_batch(self, batch_size, data_shape):
        while self.queue.qsize() < batch_size:
            pass
        
        i_t_batch, i_s_batch, t_sk_batch, t_t_batch, t_b_batch, t_f_batch, mask_t_batch = [], [], [], [], [], [], []

        for i in range(batch_size):
            i_t, i_s, t_sk, t_t, t_b, t_f, mask_t = self.dequeue_data()
            i_t_batch.append(i_t); i_s_batch.append(i_s); t_sk_batch.append(t_sk)
            t_t_batch.append(t_t); t_b_batch.append(t_b); t_f_batch.append(t_f)
            mask_t_batch.append(mask_t)

        w_sum = 0
        for t_b in t_b_batch:
            h, w = t_b.shape[:2]
            scale_ratio = data_shape[0] / h
            w_sum += int(w * scale_ratio)

        to_h = data_shape[0]
        to_w = int(round((w_sum / batch_size) / 8)) * 8
        to_size = (to_w, to_h)

        for i in range(batch_size):
            i_t_batch[i] = cv2.resize(i_t_batch[i], to_size)
            i_s_batch[i] = cv2.resize(i_s_batch[i], to_size)
            t_sk_batch[i] = cv2.resize(t_sk_batch[i], to_size, interpolation=cv2.INTER_NEAREST)
            t_t_batch[i] = cv2.resize(t_t_batch[i], to_size)
            t_b_batch[i] = cv2.resize(t_b_batch[i], to_size)
            t_f_batch[i] = cv2.resize(t_f_batch[i], to_size)
            mask_t_batch[i] = cv2.resize(mask_t_batch[i], to_size, interpolation=cv2.INTER_NEAREST)
            t_sk_batch[i] = skeletonization.skeletonization(mask_t_batch[i], 127)

        i_t_batch = np.stack(i_t_batch).astype(np.float32) / 127.5 - 1.0
        i_s_batch = np.stack(i_s_batch).astype(np.float32) / 127.5 - 1.0
        t_sk_batch = np.expand_dims(np.stack(t_sk_batch), axis=-1).astype(np.float32) / 255.0
        t_t_batch = np.stack(t_t_batch).astype(np.float32) / 127.5 - 1.0
        t_b_batch = np.stack(t_b_batch).astype(np.float32) / 127.5 - 1.0
        t_f_batch = np.stack(t_f_batch).astype(np.float32) / 127.5 - 1.0
        mask_t_batch = np.expand_dims(np.stack(mask_t_batch), axis=-1).astype(np.float32) / 255.0
        
        return [i_t_batch, i_s_batch, t_sk_batch, t_t_batch, t_b_batch, t_f_batch, mask_t_batch]

    def get_queue_size(self):
        return self.queue.qsize()

    def terminate_pool(self):
        self.pool.terminate()
