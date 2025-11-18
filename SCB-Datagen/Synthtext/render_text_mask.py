import os
import cv2
import time
import random
import math
import numpy as np
import pygame, pygame.locals
from pygame import freetype


def center2size(surf, size):
    canvas = np.zeros(size).astype(np.uint8)
    size_h, size_w = size
    surf_h, surf_w = surf.shape[:2]
    canvas[
        (size_h - surf_h) // 2 : (size_h - surf_h) // 2 + surf_h,
        (size_w - surf_w) // 2 : (size_w - surf_w) // 2 + surf_w,
    ] = surf
    return canvas


def crop_safe(arr, rect, bbs=[], pad=0):
    rect = np.array(rect)
    rect[:2] -= 2 * pad
    rect[2:] += 2 * pad
    v0 = [max(0, rect[0]), max(0, rect[1])]
    v1 = [min(arr.shape[0], rect[0] + rect[2]), min(arr.shape[1], rect[1] + rect[3])]
    arr = arr[v0[0] : v1[0], v0[1] : v1[1], ...]
    if len(bbs) > 0:
        return arr, bbs
    else:
        return arr


def render_normal(font, text, letter_spacing=2):
    lines = text.split("\n")
    lengths = [len(l) for l in lines]

    line_spacing = font.get_sized_height() + 1
    line_bounds = font.get_rect(lines[np.argmax(lengths)])
    fsize = (round(10.0 * line_bounds.width), round(10.0 * line_spacing * len(lines)))
    surf = pygame.Surface(fsize, pygame.locals.SRCALPHA, 32)

    bbs = []
    space = font.get_rect("O")
    x, y = 0, 0
    for l in lines:
        x = 0
        y += 1.25 * line_spacing

        for ch in l:
            if ch.isspace():
                x += space.width + letter_spacing
            else:
                ch_bounds = font.render_to(surf, (x, y), ch)
                x += ch_bounds.width + letter_spacing
                bbs.append(np.array(ch_bounds))

    r0 = pygame.Rect(bbs[0])
    rect_union = r0.unionall(bbs)
    
    bbs = np.array(bbs)
    surf_arr, bbs = crop_safe(pygame.surfarray.pixels_alpha(surf), rect_union, bbs, pad=10)
    surf_arr = surf_arr.swapaxes(0, 1)
    return surf_arr, bbs


def render_curved(font, text, curve_rate, curve_center=None):
    wl = len(text)
    isword = len(text.split()) == 1

    lspace = font.get_sized_height() + 1
    lbound = font.get_rect(text)
    fsize = (round(10.0 * lbound.width), round(10.0 * lspace))
    surf = pygame.Surface(fsize, pygame.locals.SRCALPHA, 32)

    if curve_center is None:
        curve_center = wl // 2
    curve_center = max(0, min(curve_center, wl - 1))
    mid_idx = curve_center
    curve = [curve_rate * (i - mid_idx) * (i - mid_idx) for i in range(wl)]
    curve[mid_idx] = -np.sum(curve) / max(wl - 1, 1)
    rots = [-int(math.degrees(math.atan(2 * curve_rate * (i - mid_idx) / (font.size / 2)))) for i in range(wl)]

    bbs = []
    rect = font.get_rect(text[mid_idx])
    rect.centerx = surf.get_rect().centerx
    rect.centery = surf.get_rect().centery + curve[mid_idx]
    ch_bounds = font.render_to(surf, rect, text[mid_idx], rotation=rots[mid_idx])
    mid_ch_bb = np.array(ch_bounds)

    last_rect = rect
    ch_idx = []
    for i in range(wl):
        if i == mid_idx:
            bbs.append(mid_ch_bb)
            ch_idx.append(i)
            continue

        if i < mid_idx:
            i = mid_idx - 1 - i
        elif i == mid_idx + 1:
            last_rect = rect

        ch_idx.append(i)
        ch = text[i]

        newrect = font.get_rect(ch)
        newrect.y = last_rect.y
        if i > mid_idx:
            newrect.topleft = (last_rect.topright[0] + 2, newrect.topleft[1])
        else:
            newrect.topright = (last_rect.topleft[0] - 2, newrect.topleft[1])
        newrect.centery = max(newrect.height, min(fsize[1] - newrect.height, newrect.centery + curve[i]))
        try:
            bbrect = font.render_to(surf, newrect, ch, rotation=rots[i])
        except ValueError:
            bbrect = font.render_to(surf, newrect, ch)
        bbs.append(np.array(bbrect))
        last_rect = newrect

    bbs_sequence_order = [None for _ in ch_idx]
    for idx, i in enumerate(ch_idx):
        bbs_sequence_order[i] = bbs[idx]
    bbs = bbs_sequence_order

    r0 = pygame.Rect(bbs[0])
    rect_union = r0.unionall(bbs)

    bbs = np.array(bbs)
    surf_arr, bbs = crop_safe(pygame.surfarray.pixels_alpha(surf), rect_union, bbs, pad=10)
    surf_arr = surf_arr.swapaxes(0, 1)
    return surf_arr, bbs


def center_warpPerspective(img, H, center, size):
    P = np.array([[1, 0, center[0]], [0, 1, center[1]], [0, 0, 1]], dtype=np.float32)
    M = P.dot(H).dot(np.linalg.inv(P))
    return cv2.warpPerspective(img, M, size, cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)


def center_pointsPerspective(points, H, center):
    P = np.array([[1, 0, center[0]], [0, 1, center[1]], [0, 0, 1]], dtype=np.float32)
    M = P.dot(H).dot(np.linalg.inv(P))
    return M.dot(points)


def perspective(img, rotate_angle, zoom, shear_angle, perspect, pad):
    rotate_angle = rotate_angle * math.pi / 180.0
    shear_x_angle = shear_angle[0] * math.pi / 180.0
    shear_y_angle = shear_angle[1] * math.pi / 180.0
    scale_w, scale_h = zoom
    perspect_x, perspect_y = perspect

    H_scale = np.array([[scale_w, 0, 0], [0, scale_h, 0], [0, 0, 1]], dtype=np.float32)
    H_rotate = np.array(
        [
            [math.cos(rotate_angle), math.sin(rotate_angle), 0],
            [-math.sin(rotate_angle), math.cos(rotate_angle), 0],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )
    H_shear = np.array([[1, math.tan(shear_x_angle), 0], [math.tan(shear_y_angle), 1, 0], [0, 0, 1]], dtype=np.float32)
    H_perspect = np.array([[1, 0, 0], [0, 1, 0], [perspect_x, perspect_y, 1]], dtype=np.float32)
    H = H_rotate.dot(H_shear).dot(H_scale).dot(H_perspect)

    img_h, img_w = img.shape[:2]
    img_center = (img_w / 2, img_h / 2)
    points = np.ones((3, 4), dtype=np.float32)
    points[:2, 0] = np.array([0, 0], dtype=np.float32).T
    points[:2, 1] = np.array([img_w, 0], dtype=np.float32).T
    points[:2, 2] = np.array([img_w, img_h], dtype=np.float32).T
    points[:2, 3] = np.array([0, img_h], dtype=np.float32).T
    
    perspected_points = center_pointsPerspective(points, H, img_center)
    perspected_points[0, :] /= perspected_points[2, :]
    perspected_points[1, :] /= perspected_points[2, :]
    
    max_x = max(img_center[0], img_center[0] - np.min(perspected_points[0, :]), np.max(perspected_points[0, :]) - img_center[0])
    max_y = max(img_center[1], img_center[1] - np.min(perspected_points[1, :]), np.max(perspected_points[1, :]) - img_center[1])
    canvas_w = int(2 * max_x) + 10
    canvas_h = int(2 * max_y) + 10

    canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    tly, tlx = (canvas_h - img_h) // 2, (canvas_w - img_w) // 2
    canvas[tly : tly + img_h, tlx : tlx + img_w] = img
    
    canvas_center = (canvas_w // 2, canvas_h // 2)
    canvas_size = (canvas_w, canvas_h)
    canvas = center_warpPerspective(canvas, H, canvas_center, canvas_size)
    
    loc = np.where(canvas > 127)
    miny, minx = np.min(loc[0]), np.min(loc[1])
    maxy, maxx = np.max(loc[0]), np.max(loc[1])
    text_w, text_h = maxx - minx + 1, maxy - miny + 1
    
    resimg = np.zeros((text_h + pad[2] + pad[3], text_w + pad[0] + pad[1])).astype(np.uint8)
    resimg[pad[2] : pad[2] + text_h, pad[0] : pad[0] + text_w] = canvas[miny : maxy + 1, minx : maxx + 1]
    return resimg


def render_text(font, text, param):
    if param["is_curve"]:
        return render_curved(font, text, param["curve_rate"], param["curve_center"])
    else:
        return render_normal(font, text, letter_spacing=param["letter_spacing"])
