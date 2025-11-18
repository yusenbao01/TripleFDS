"""
Some configurations.
Copyright (c) 2019 Netease Youdao Information Technology Co.,Ltd.
Licensed under the GPL License (see LICENSE for details)
Written by Yu Qian
"""

import numpy as np


# TODO:
restored_bg_num = 9000
min_bg_list_size = 1000
fixed_size = (128, 256)  # Fixed output size
min_area_requirement = 2000  # Minimum area requirement for text region
max_len_diff = 3  # Maximum allowed length difference
min_text_len = 3  # Minimum text length
color_diff1 = 100  # Color difference threshold between foreground and background
color_diff2 = 50  # Color difference threshold between foregrounds
buffer = 0

# font
font_size = [1, 500]  # Font size range
underline_rate = 0.0005  # Underline probability
strong_rate = 0.1  # Bold probability
oblique_rate = 0.1  # Italic probability

# text
capitalize_rate = 0.0  # Capitalize first letter probability
uppercase_rate = 0.0  # All uppercase probability

# background
# Background enhancement parameters
brightness_rate = 0.8  # Brightness adjustment probability
brightness_min = 0.5
brightness_max = 1.5
color_rate = 0.8  # Color adjustment probability
color_min = 0.5
color_max = 1.5
contrast_rate = 0.8  # Contrast adjustment probability
contrast_min = 0.5
contrast_max = 1.5

# curve
is_curve_rate = 0.3  # Curved text probability
curve_rate_param = [0.5, 0]  # Curve parameters (scale and offset)

# perspective (enlarged from original settings)
rotate_param = [2.0, 0]  # Rotation parameters (scale and offset)
zoom_param = [0.01, 1]  # Zoom parameters (not recommended to change)
shear_param = [2, 0]  # Shear parameters
perspect_param = [0.0005, 0]  # Perspective parameters (not recommended to change)

# render
## surf augment
elastic_rate = 0.01  # Elastic deformation probability
elastic_grid_size = 4  # Grid size for elastic deformation
elastic_magnitude = 2  # Magnitude of elastic deformation

## colorize
# Colorization parameters
padding_ud = [0, 0]  # Top/bottom padding range
padding_lr = [0, 0]  # Left/right padding range
is_border_rate = 0.05  # Border probability
is_shadow_rate = 0.000  # Shadow probability
shadow_angle_degree = [1, 3, 5, 7]  # Shadow angle offsets
shadow_angle_param = [0.5, None]  # Shadow angle parameters (scale and offset)
shadow_shift_param = np.array([[0, 1, 3], [2, 7, 15]], dtype=np.float32)  # Shadow shift parameters
shadow_opacity_param = [0.1, 0.5]  # Shadow opacity offsets
color_filepath = "./Synthtext/data/colors_new.cp"  # Color file path
use_random_color_rate = 0.4  # Random color probability

# Gaussian blur
p_blur = 0.4  # Probability of applying blur
min_blur_radius = 0.2  # Minimum blur radius (corresponds to sigmaX)
max_blur_radius = 1.5  # Maximum blur radius (corresponds to sigmaX)

max_letter_spacing = 4
min_letter_spacing = 2
