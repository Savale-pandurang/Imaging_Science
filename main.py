import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import os
import sys

"""Read an image and return it as a normalized numpy array (0–1)."""
def read_image(path, color=True):
    img = Image.open(path)
    img = img.convert("RGB") if color else img.convert("L")
    return np.asarray(img, dtype=np.float32) / 255.0

"""Save a numpy array (0–1) as an image file."""
def save_image(arr, path):
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)

"""Display a list of (title, image) pairs side by side."""
def show_images(images):
    n = len(images)
    plt.figure(figsize=(4 * n, 4))
    for i, (title, img) in enumerate(images, 1):
        plt.subplot(1, n, i)
        plt.imshow(img if img.ndim == 3 else img, cmap='gray')
        plt.title(title)
        plt.axis('off')
    plt.show()

"""Create a 2D Gaussian kernel."""
def make_gaussian_kernel(size, sigma):
    if size % 2 == 0:
        raise ValueError("Kernel size must be odd.")
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel /= np.sum(kernel)
    return kernel.astype(np.float32)

"""Perform manual 2D convolution."""
def convolve2d_manual(img, kernel):
    H, W = img.shape
    kH, kW = kernel.shape
    padH, padW = kH // 2, kW // 2

    padded = np.pad(img, ((padH, padH), (padW, padW)), mode='reflect')
    output = np.zeros_like(img, dtype=np.float32)

    for i in range(H):
        for j in range(W):
            region = padded[i:i + kH, j:j + kW]
            output[i, j] = np.sum(region * kernel)
    return output

"""Apply Single-Scale Retinex to a grayscale image."""
def retinex_gray(img_gray, kernel_size=31, sigma=8.0, eps=1e-6):
    I = np.clip(img_gray, eps, 1.0)
    logI = np.log(I)

    kernel = make_gaussian_kernel(kernel_size, sigma)
    logL = convolve2d_manual(logI, kernel)
    logR = logI - logL

    R = np.exp(logR)
    L = np.exp(logL)

    R_vis = (R - R.min()) / (R.max() - R.min() + 1e-9)
    L_vis = (L - L.min()) / (L.max() - L.min() + 1e-9)

    return R_vis, L_vis, logL

"""Compute luminance (Y channel) from RGB."""
def rgb_to_luminance(rgb):
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return 0.299 * R + 0.587 * G + 0.114 * B


def srgb_to_linear(img):
    """Convert sRGB to linear color space."""
    return np.where(img <= 0.04045, img / 12.92, ((img + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(img):
    """Convert linear color space to sRGB."""
    return np.where(img <= 0.0031308, img * 12.92, 1.055 * (img ** (1 / 2.4)) - 0.055)


def estimate_color_scales(img_linear, L):
    """Estimate per-channel scaling factors to preserve color balance."""
    eps = 1e-8
    ratios = img_linear / (L[..., None] + eps)
    cR, cG, cB = np.median(ratios[..., 0]), np.median(ratios[..., 1]), np.median(ratios[..., 2])
    return np.array([max(cR, eps), max(cG, eps), max(cB, eps)], dtype=np.float32)

"""Improved color Retinex with tone preservation."""
def retinex_color(img_rgb, kernel_size=31, sigma=8.0, eps=1e-6):
    img = np.clip(img_rgb, 0.0, 1.0)
    img_linear = srgb_to_linear(img)

    Y = np.clip(rgb_to_luminance(img_linear), eps, 1.0)
    logY = np.log(Y)
    kernel = make_gaussian_kernel(kernel_size, sigma)
    logL = convolve2d_manual(logY, kernel)
    L = np.exp(logL)

    color_scales = estimate_color_scales(img_linear, L)
    denom = (L[..., None] * color_scales[None, None, :]) + eps
    R_lin = img_linear / denom

    mean_in = np.mean(rgb_to_luminance(img_linear))
    mean_out = np.mean(rgb_to_luminance(R_lin))
    gain = mean_in / (mean_out + eps)
    R_lin *= gain

    R_lin = np.power(np.clip(R_lin, 0.0, 1.0), 0.9)
    R_srgb = np.clip(linear_to_srgb(R_lin), 0.0, 1.0)
    L_vis = (L - L.min()) / (L.max() - L.min() + 1e-9)

    return R_srgb, L_vis, logL

"""Main driver: handles reading, processing, and saving images."""
def main(args):
    if args.mode == 'gray':
        is_color = False
    elif args.mode == 'color':
        is_color = True
    else:
        try:
            img = Image.open(args.input)
            is_color = (img.mode != 'L')
            img.close()
        except Exception as e:
            print("Error reading image:", e)
            sys.exit(1)

    if is_color:
        img = read_image(args.input, color=True)
        print(f"Loaded color image: {img.shape}")
        R_vis, L_vis, _ = retinex_color(img, args.kernel, args.sigma)

        base = os.path.splitext(os.path.basename(args.input))[0]
        save_image(R_vis, f"{base}_reflectance_color.png")
        save_image(L_vis, f"{base}_illumination_color.png")

        print("Saved reflectance and illumination maps (color).")
        if args.show:
            show_images([("Original", img), ("Reflectance", R_vis), ("Illumination", L_vis)])
    else:
        img = read_image(args.input, color=False)
        print(f"Loaded grayscale image: {img.shape}")
        R_vis, L_vis, _ = retinex_gray(img, args.kernel, args.sigma)

        base = os.path.splitext(os.path.basename(args.input))[0]
        save_image(R_vis, f"{base}_reflectance_gray.png")
        save_image(L_vis, f"{base}_illumination_gray.png")

        print("Saved reflectance and illumination maps (grayscale).")
        if args.show:
            show_images([("Original", img), ("Reflectance", R_vis), ("Illumination", L_vis)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simple Retinex-based illumination correction (manual Gaussian + convolution)."
    )
    parser.add_argument("--kernel", type=int, default=31, help="Size of Gaussian kernel (odd).")
    parser.add_argument("--sigma", type=float, default=8.0, help="Standard deviation for Gaussian.")
    parser.add_argument("--mode", choices=['auto', 'gray', 'color'], default='auto', help="Force gray/color mode.")
    parser.add_argument("--show", action='store_true', help="Display images after processing.")
    args = parser.parse_args()

    args.input = input("Enter image path: ").strip()
    main(args)
