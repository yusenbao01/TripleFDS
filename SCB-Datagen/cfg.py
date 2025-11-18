"""
Configurations of data generating.
Copyright (c) 2019 Netease Youdao Information Technology Co.,Ltd.
Licensed under the GPL License (see LICENSE for details)
Written by Yu Qian
"""

import os


def name_dataset(data_mode, lang_prob, total_num, difficulty):
    lang_name = "+".join([f"{prob}*{lang}" for lang, prob in lang_prob.items()])
    total_num_name = f"{total_num//1000000}m" if total_num >= 1000000 else f"{total_num//1000}k"
    return f"{lang_name}-{total_num_name}-{data_mode}-{difficulty}"


# Subdirectory names for storing different types of image data
i_t_dir = "i_t"  # Target image (contains the target text)
i_s_dir = "i_s"  # Source image (contains the source text)
t_sk_dir = "t_sk"  # Text skeleton
t_t_dir = "t_t"  # Target text
t_b_dir = "t_b"  # Text background
t_f_dir = "t_f"  # Text foreground
s_s_dir = "s_s"  # Source style
mask_t_dir = "mask_t"  # Target text mask
mask_s_dir = "mask_s"  # Source text mask


# TODO:
process_num = 32  # Number of processes to use
data_capacity = 256  # Capacity of the data queue
type = "train"
difficulty = "middle"  # Difficulty level: "easy" or "hard"
data_mode = "8"  # "8", "27", "64", or "2x3x4" for "font, text, bg"
images_num = 3000

# --- PATHS ---
# Replace with the actual path to your datasets directory
base_path = "/path/to/your/datasets"
bg_folder = os.path.join(base_path, "Contrastive-bsc/background/bg_scenevtg_erase_new", type, "")
text_folder = "./Synthtext/data/"
font_folder = os.path.join(base_path, "Contrastive-bsc/fonts_500/")

lang_prob = {
    "en": 1.0,
    # "zh": 0.5,
}


if type == "train":
    total_num = images_num
elif type == "val":
    total_num = images_num // 10
else:
    total_num = images_num // 1

if data_mode == "8":
    sample_num = total_num // 8
elif data_mode == "27":
    sample_num = total_num // 27
elif data_mode == "64":
    sample_num = total_num // 64
else:
    nums = [int(dim) for dim in data_mode.split("x")]
    sample_num = total_num // (nums[0] * nums[1] * nums[2])

# The final data directory will be constructed based on the dataset name and type
data_dir = os.path.join(base_path, "Contrastive-bsc", name_dataset(data_mode, lang_prob, images_num, difficulty), type)
