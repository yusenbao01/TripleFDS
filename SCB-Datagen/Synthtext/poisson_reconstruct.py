from __future__ import division
import numpy as np
import scipy.fftpack
import scipy.ndimage
import cv2
import matplotlib.pyplot as plt


def DST(x):
    return scipy.fftpack.dst(x, type=1, axis=0) / 2.0


def IDST(X):
    n = X.shape[0]
    return np.real(scipy.fftpack.idst(X, type=1, axis=0)) / (n + 1.0)


def get_grads(im):
    [H, W] = im.shape
    Dx, Dy = np.zeros((H, W), dtype=np.float32), np.zeros((H, W), dtype=np.float32)
    j, k = np.atleast_2d(np.arange(0, H - 1)).T, np.arange(0, W - 1)
    Dx[j, k] = im[j, k + 1] - im[j, k]
    Dy[j, k] = im[j + 1, k] - im[j, k]
    return Dx, Dy


def get_laplacian(Dx, Dy):
    [H, W] = Dx.shape
    Dxx, Dyy = np.zeros((H, W)), np.zeros((H, W))
    j, k = np.atleast_2d(np.arange(0, H - 1)).T, np.arange(0, W - 1)
    Dxx[j, k + 1] = Dx[j, k + 1] - Dx[j, k]
    Dyy[j + 1, k] = Dy[j + 1, k] - Dy[j, k]
    return Dxx + Dyy


def poisson_solve(gx, gy, bnd):
    gx, gy, bnd = gx.astype(np.float32), gy.astype(np.float32), bnd.astype(np.float32)
    H, W = bnd.shape
    L = get_laplacian(gx, gy)

    bnd[1:-1, 1:-1] = 0
    L_bp = np.zeros_like(L)
    L_bp[1:-1, 1:-1] = -4 * bnd[1:-1, 1:-1] + bnd[1:-1, 2:] + bnd[1:-1, 0:-2] + bnd[2:, 1:-1] + bnd[0:-2, 1:-1]
    L = L - L_bp
    L = L[1:-1, 1:-1]

    L_dst = DST(DST(L).T).T
    [xx, yy] = np.meshgrid(np.arange(1, W - 1), np.arange(1, H - 1))
    D = (2 * np.cos(np.pi * xx / (W - 1)) - 2) + (2 * np.cos(np.pi * yy / (H - 1)) - 2)
    L_dst /= D

    img_interior = IDST(IDST(L_dst).T).T
    img = bnd.copy()
    img[1:-1, 1:-1] = img_interior
    return img


def poisson_blit_images(im_top, im_back, scale_grad_range=(0.8, 1.5), mode="max"):
    """
    Optimized Poisson blending function with randomized gradient scaling.
    This creates a continuous spectrum of blending effects from subtle to strong,
    significantly increasing data diversity.
    """
    if not np.all(im_top.shape == im_back.shape):
        print(f"Foreground shape: {im_top.shape}, Background shape: {im_back.shape}")
    assert np.all(im_top.shape == im_back.shape)

    im_top = im_top.copy().astype(np.float32)
    im_back = im_back.copy().astype(np.float32)
    im_res = np.zeros_like(im_top)

    scale_grad = np.random.uniform(scale_grad_range[0], scale_grad_range[1])

    for ch in range(im_top.shape[2]):
        ims, imd = im_top[:, :, ch], im_back[:, :, ch]
        gxs, gys = get_grads(ims)
        gxd, gyd = get_grads(imd)

        gxs *= scale_grad
        gys *= scale_grad
        
        gx = gxs.copy()
        gxm = np.abs(gxd) > np.abs(gxs)
        gx[gxm] = gxd[gxm]

        gy = gys.copy()
        gym = np.abs(gyd) > np.abs(gys)
        gy[gym] = gyd[gym]

        im_res[:, :, ch] = np.clip(poisson_solve(gx, gy, imd), 0, 255)

    return im_res.astype("uint8")


def alpha_blend_images(im_top, im_back, mask):
    """
    Smooth, natural edge blending using a Gaussian blurred alpha channel.
    This is a great alternative to more complex Poisson blending.
    """
    im_top = im_top.astype(np.float32)
    im_back = im_back.copy().astype(np.float32)

    alpha = cv2.GaussianBlur(mask.astype(np.float32), (7, 7), 0)
    alpha = np.clip(alpha / 255.0, 0, 1)[..., None]

    im_back = im_top * alpha + im_back * (1.0 - alpha)
    return im_back.astype("uint8")
